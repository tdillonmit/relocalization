#!/usr/bin/env python3.9
"""Drive ParticleFilter over a real IVUS/EM replay dataset for relocalization.

Standalone companion to run_relocalization.py: rather than trusting the
EM-measured extrinsic (``TW_EM @ TEM_C``) directly, this treats it only as
the noisy "control input" (Section 4.4's u) driving the particle filter's
motion model, and localizes the catheter's extrinsic pose purely from the
image correlation between the real DeepLumen segmentation of each recorded
frame and each particle's simulated segmentation
(``VesselUltrasoundSimulator.simulate_segmentation``).

Expected dataset directory layout (matches PointCloudUpdater's replay
loader in run_relocalization.py):

    <dataset_dir>/
        calibration_parameters_ivus.yaml
        centerline_pc.ply
        fixed_branch_mesh.ply          (watertight registered CT mesh)
        no_branch_mesh.ply             (used internally by the simulator)
        side_branch_centrelines.ply    (used internally by the simulator)
        grayscale_images/*.npy
        transform_data/*.npy           (TW_EM, one 4x4 per frame)

Example
-------
    python3.9 run_particle_filter_relocalization.py \\
        --dataset-dir /home/tdillon/datasets/k8_relocalization \\
        --num-particles 300 --max-frames 150 --visualize
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import tensorflow as tf

from particle_filter import Centerline, ParticleCloudVisualizer, ParticleFilter, ParticleFilterConfig
from reconstruction_helpers_runtime import get_transform_inverse, load_default_values
from segmentation_helpers_runtime import build_mldr_drn, deeplumen_segmentation, post_process_deeplumen
from ultrasound_mesh_simulator import VesselUltrasoundSimulator, keep_largest_component


def _natural_key(path: Path):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def load_frame_pairs(dataset_dir: Path):
    image_dir = dataset_dir / "grayscale_images"
    transform_dir = dataset_dir / "transform_data"
    image_files = sorted(image_dir.glob("*.npy"), key=_natural_key)
    transform_files = sorted(transform_dir.glob("*.npy"), key=_natural_key)
    if len(image_files) != len(transform_files):
        raise ValueError(
            f"Image/transform count mismatch: {len(image_files)} images, "
            f"{len(transform_files)} transforms."
        )
    return image_files, transform_files


def build_tem_c(calibration):
    """Fixed EM-sensor-to-catheter calibration transform, matching
    run_relocalization.py's ``TEM_C`` construction exactly."""
    angle = calibration["/angle"]
    translation = calibration["/translation"]
    radial_offset = calibration["/radial_offset"]
    oclock = calibration["/oclock"]
    return np.array(
        [
            [1, 0, 0, translation],
            [0, np.cos(angle), -np.sin(angle), radial_offset * np.cos(oclock)],
            [0, np.sin(angle), np.cos(angle), radial_offset * np.sin(oclock)],
            [0, 0, 0, 1],
        ]
    )


def load_deeplumen_model(weights_path):
    model = build_mldr_drn(
        input_shape=(224, 224, 3), num_classes=3, base=64,
        blocks_per_stage=(2, 2, 3, 3, 3), dilations=(1, 2, 4),
        dropout=0.2, upsample_stride=8, return_pyramid=False, name="MLDR_DRN_Large",
    )
    model.load_weights(str(weights_path))
    return tf.function(model, jit_compile=True)


def segment_observed_frame(grayscale_image, model, conf_threshold=0.92):
    """Real DeepLumen segmentation of one recorded frame, matching
    PointCloudUpdater.append_image_transform_pair's deeplumen branch."""
    
    image = cv2.resize(grayscale_image, (224, 224))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    
    pred, conf_class2 = deeplumen_segmentation(image_rgb, model)
    
    conf_class2 = conf_class2.numpy()
    raw_data = pred[0].numpy()
    mask_1, mask_2, _, _ = post_process_deeplumen(raw_data, conf_class2, conf_threshold, hybrid_seg=0)
    mask_2 = (keep_largest_component(mask_2) * 255).astype(np.uint8)
    
    return mask_1, mask_2


def pose_error(pose_a, pose_b):
    """Translation error (m), forward-axis ("aim") error (deg), and total
    rotation error (deg) between two SE(3) poses.

    Forward-axis error only compares the probe's x axis (its imaging-plane
    normal) and ignores roll about that axis. Total rotation error folds
    roll in too. A lumen cross-section is often close to rotationally
    symmetric about the probe axis unless a side branch is in view, so the
    image correlation score may not meaningfully constrain roll -- expect
    total rotation error to be noisier than forward-axis error for that
    reason, not necessarily because localization is failing.
    """
    translation_error = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))

    cos_forward = np.clip(np.dot(pose_a[:3, 0], pose_b[:3, 0]), -1.0, 1.0)
    forward_error_deg = float(np.degrees(np.arccos(cos_forward)))

    relative_rotation = pose_a[:3, :3].T @ pose_b[:3, :3]
    cos_angle = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = float(np.degrees(np.arccos(cos_angle)))

    return translation_error, forward_error_deg, rotation_error_deg


def resolve_dataset_dir(dataset_dir_arg, data_root):
    """Accept either a full path or a bare dataset name.

    Mirrors PointCloudUpdater's ``self.write_folder = "/home/tdillon/datasets/"
    + str(dataset_path)`` convention in run_relocalization.py: a bare name
    (no existing relative/absolute path) is resolved under ``data_root``.
    """
    dataset_dir_arg = Path(dataset_dir_arg).expanduser()
    if dataset_dir_arg.is_absolute() or dataset_dir_arg.exists():
        return dataset_dir_arg.resolve()
    return (data_root / dataset_dir_arg).expanduser().resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", required=True, type=str,
        help=(
            "Either a full path, or a bare dataset name resolved under "
            "--data-root (e.g. 'k8_relocalization')."
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/tdillon/datasets"),
        help="Parent directory bare --dataset-dir names are resolved under.",
    )
    parser.add_argument("--num-particles", type=int, default=300)
    parser.add_argument("--vessel-diameter-m", type=float, default=0.02)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-near-em",
        action="store_true",
        help=(
            "Initialize particles around the first frame's EM-measured pose "
            "instead of blind global localization along the whole centerline."
        ),
    )
    parser.add_argument("--visualize", action="store_true", help="Open a live Open3D window.")
    parser.add_argument(
        "--reference-frame-size", type=float, default=0.01,
        help=(
            "Size (m) of the coordinate-frame triad drawn at the raw "
            "EM-measured extrinsic (TW_EM @ TEM_C, no particle-filter "
            "correction) -- a fixed reference to visually confirm the "
            "particle cloud does/doesn't converge onto it."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "deeplumen_model" / "weights.weights.h5",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.92)
    parser.add_argument(
        "--mesh-decimation-fraction", type=float, default=None,
        help=(
            "Decimate the mesh(es) simulate_segmentation_fast sections, as a "
            "fraction of original triangle count (e.g. 0.5). Off by default. "
            "~2x faster particle scoring at 0.5 with mean branch-mask Dice "
            "0.94 vs undecimated; more aggressive fractions trade further "
            "speed for branch-detection reliability -- see "
            "VesselUltrasoundSimulator.simulate_segmentation_fast's docstring."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.data_root)
    print(f"Using dataset directory: {dataset_dir}")

    calibration = load_default_values(str(dataset_dir / "calibration_parameters_ivus.yaml"))
    tem_c = build_tem_c(calibration)
    scaling = calibration["/scaling"]

    print("Loading registered mesh + centerline...")
    registered_mesh = o3d.io.read_triangle_mesh(str(dataset_dir / "fixed_branch_mesh.ply"))
    sim = VesselUltrasoundSimulator(
        str(dataset_dir), registered_mesh, image_size=224, fov_mm=scaling * 224 * (781 / 224),
        fast_decimation_fraction=args.mesh_decimation_fraction,
    )
    centerline = Centerline.from_ply(dataset_dir / "centerline_pc.ply")

    print("Loading DeepLumen segmentation model...")
    tf.keras.backend.clear_session()
    model = load_deeplumen_model(args.model_path)

    image_files, transform_files = load_frame_pairs(dataset_dir)
    start = args.start_frame
    stop = len(image_files) if args.max_frames is None else min(len(image_files), start + args.max_frames * args.frame_stride)
    frame_indices = list(range(start, stop, args.frame_stride))
    print(f"Running {len(frame_indices)} frames from {dataset_dir.name} "
          f"(frames {start}:{stop}:{args.frame_stride})")

    config = ParticleFilterConfig(
        num_particles=args.num_particles, vessel_diameter_m=args.vessel_diameter_m,
    )
    pf = ParticleFilter(centerline, config, seed=args.seed)

    first_em = np.load(transform_files[frame_indices[0]]) @ tem_c
    pf.initialize(seed_pose=first_em if args.seed_near_em else None, sim=sim)

    vis = None
    cloud_viz = None
    mesh_viz = None
    reference_frame = None
    reference_prev_pose = np.eye(4)

    triangles = np.asarray(registered_mesh.triangles)
    registered_mesh.triangles = o3d.utility.Vector3iVector(triangles[:, ::-1])
    registered_mesh.compute_vertex_normals()

    if args.visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Particle filter relocalization", width=1200, height=900)
        vis.get_render_option().mesh_show_back_face = False
        mesh_viz = o3d.geometry.TriangleMesh(registered_mesh)
        mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_viz.triangles)[:, ::-1])
        mesh_viz.compute_vertex_normals()
        mesh_viz.paint_uniform_color([0.75, 0.75, 0.78])
        vis.add_geometry(mesh_viz)
        cloud_viz = ParticleCloudVisualizer(vis)

        # Raw EM-measured extrinsic (TW_EM @ TEM_C), no particle-filter
        # correction -- a fixed reference to check the particle cloud
        # against: if the EM registration is trustworthy, particles should
        # converge onto this frame; if not, the filter's estimate (green
        # cylinder) should visibly separate from it instead.
        reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=args.reference_frame_size
        )
        vis.add_geometry(reference_frame)

    previous_extrinsic = first_em
    log_rows = []

    for step_number, frame_idx in enumerate(frame_indices):
        t_start = time.perf_counter()

        grayscale_image = np.load(image_files[frame_idx])
        tw_em = np.load(transform_files[frame_idx])
        measured_extrinsic = tw_em @ tem_c

        observed_mask_1, observed_mask_2 = segment_observed_frame(
            grayscale_image, model, conf_threshold=args.conf_threshold
        )

        estimate = pf.step(
            previous_extrinsic, measured_extrinsic, sim, observed_mask_1, observed_mask_2,
        )
        previous_extrinsic = measured_extrinsic

        translation_err, forward_err_deg, rotation_err_deg = pose_error(estimate.pose, measured_extrinsic)
        elapsed = time.perf_counter() - t_start
        log_rows.append((frame_idx, translation_err, forward_err_deg, rotation_err_deg, estimate.confidence, estimate.num_particles))

        print(
            f"[{step_number:04d}] frame={frame_idx} n={estimate.num_particles:4d} "
            f"conf={estimate.confidence:.3f} p_new={pf.last_p_new:.2f} "
            f"err_pos={translation_err * 1000:6.2f} mm err_aim={forward_err_deg:6.2f} deg "
            f"err_rot={rotation_err_deg:6.2f} deg ({elapsed:.2f}s/frame)"
        )

        if vis is not None:
            cloud_viz.update(pf, estimate_pose=estimate.pose)
            reference_frame.transform(get_transform_inverse(reference_prev_pose))
            reference_frame.transform(measured_extrinsic)
            reference_prev_pose = measured_extrinsic.copy()
            vis.update_geometry(reference_frame)
            vis.poll_events()
            vis.update_renderer()

    if vis is not None:
        print("Close the Open3D window to exit.")
        vis.run()
        vis.destroy_window()

    log_rows = np.array([(r[1], r[2], r[3]) for r in log_rows])
    print(
        f"\nDone. Median vs EM-measured pose: "
        f"position={np.median(log_rows[:, 0]) * 1000:.2f} mm, "
        f"aim (forward-axis only)={np.median(log_rows[:, 1]):.2f} deg, "
        f"total rotation (incl. roll)={np.median(log_rows[:, 2]):.2f} deg.\n"
        "Note: the EM-measured pose is only a proxy reference (what the "
        "non-relocalization pipeline trusts directly), not independent "
        "ground truth -- sustained disagreement is expected exactly where "
        "relocalization is doing its job (drifted/incorrect EM "
        "registration), not necessarily a filter failure. Total rotation "
        "error is usually noisier than aim error because a roughly "
        "circular lumen cross-section doesn't strongly constrain roll "
        "about the probe axis unless a side branch is in view."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3.9
"""Demo/smoke-test for VesselUltrasoundSimulator.

Samples random poses inside K1_1_branches.stl, simulates the DeepLumen-style
(mask_1, mask_2) segmentation at each, and writes preview overlays + raw
masks to ./outputs/ultrasound_sim_demo/ for visual inspection.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultrasound_mesh_simulator import NoiseParams, VesselUltrasoundSimulator

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=str(SCRIPT_DIR / "K1_1_branches.stl"))
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--fov-mm",
        type=float,
        default=0.05,
        help=(
            "Imaging-plane field of view, in the mesh's native coordinate "
            "units (ct_slicer_mesh.ply is stored in meters, so 0.05 = a "
            "50mm-equivalent FOV)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_DIR / "outputs" / "ultrasound_sim_demo")
    )
    parser.add_argument(
        "--noisy", action="store_true", help="Apply mild sensor-noise simulation."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Instead of saving PNGs, open an Open3D window showing the mesh "
            "and probe pose plus two OpenCV windows with the live masks. "
            "Requires a display."
        ),
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=800,
        help="Milliseconds each pose is shown for in --interactive mode (0 = wait for keypress).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sim = VesselUltrasoundSimulator(
        args.mesh, image_size=args.image_size, fov_mm=args.fov_mm
    )

    noise = None
    if args.noisy:
        noise = NoiseParams(
            boundary_noise_std_mm=0.3,
            branch_dropout_prob=0.15,
            lumen_dropout_prob=0.02,
            dilate_px=1,
        )

    if args.interactive:
        poses = [sim.sample_random_pose(rng) for _ in range(args.num_poses)]
        sim.visualize_poses(
            poses,
            wait_ms=args.wait_ms,
            noise=noise,
            rng=rng,
            mesh_show_back_face=False,
        )
        return

    empty_lumen_count = 0
    branch_present_count = 0

    for i in range(args.num_poses):
        pose = sim.sample_random_pose(rng)
        mask_1, mask_2 = sim.simulate_segmentation(pose, noise=noise, rng=rng)

        if mask_1.sum() == 0:
            empty_lumen_count += 1
        if mask_2.sum() > 0:
            branch_present_count += 1

        preview = sim.render_preview(mask_1, mask_2)
        cv2.imwrite(str(output_dir / f"pose_{i:03d}_preview.png"), preview)
        cv2.imwrite(str(output_dir / f"pose_{i:03d}_mask1.png"), mask_1)
        cv2.imwrite(str(output_dir / f"pose_{i:03d}_mask2.png"), mask_2)
        np.save(output_dir / f"pose_{i:03d}_pose.npy", pose)

        print(
            f"pose {i:03d}: position={np.round(pose[:3, 3], 2)} "
            f"lumen_px={int(mask_1.sum() // 255)} branch_px={int(mask_2.sum() // 255)}"
        )

    print(
        f"\nSaved {args.num_poses} previews to {output_dir}\n"
        f"empty-lumen poses: {empty_lumen_count}/{args.num_poses} "
        f"(random orientation means the plane sometimes clips outside the FOV)\n"
        f"poses with a visible branch: {branch_present_count}/{args.num_poses}"
    )


if __name__ == "__main__":
    main()

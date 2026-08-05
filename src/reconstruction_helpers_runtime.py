"""Runtime reconstruction utilities used by ``run_mapping.py``."""

from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


__all__ = [
    "load_default_values",
    "get_box",
    "get_transform_inverse",
    "update_tsdf_mesh",
    "WireframeGenerator",
    "get_point_cloud_from_masks",
    "get_single_point_cloud_from_mask",
    "get_single_point_cloud_from_pixels",
]


def load_default_values(file_path=None):
    """Load the default IVUS calibration parameters from YAML.

    When no path is supplied, the calibration file is expected next to this
    module. This replaces the original developer-specific absolute path while
    preserving the existing ``load_default_values()`` call in ``run_mapping``.
    """

    print("file_path is", file_path)
    if file_path is None:
        file_path = Path(__file__).resolve().parent / "calibration_parameters_ivus.yaml"
    else:
        file_path = Path(file_path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Default IVUS calibration file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as stream:
        try:
            all_parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Could not parse IVUS calibration YAML: {file_path}"
            ) from exc

    if not isinstance(all_parameters, dict):
        raise ValueError(
            f"IVUS calibration file must contain a YAML mapping: {file_path}"
        )

    supported_keys = (
        "/angle",
        "/threshold",
        "/translation",
        "/scaling",
        "/radial_offset",
        "/oclock",
    )
    return {
        key: all_parameters[key]
        for key in supported_keys
        if key in all_parameters
    }



def get_box(min_bounds, max_bounds):
    """Create the axis-aligned bounding box displayed by ``run_mapping``."""
    return o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)


def get_transform_inverse(transform):
    """Return the inverse of a rigid 4 x 4 homogeneous transform."""
    displacement = transform[:3, 3]
    rotation_transpose = np.transpose(transform[:3, :3])

    inverse = np.eye(4)
    inverse[:3, :3] = rotation_transpose
    inverse[:3, 3] = -rotation_transpose @ displacement
    return inverse


def update_tsdf_mesh(
    vis,
    tsdf_volume,
    mesh,
    three_d_points,
    extrinsic_matrix,
    color,
    keep_largest=True,
):
    """Integrate points into the TSDF and update an Open3D triangle mesh.

    ``vis`` and ``color`` are retained for compatibility with the existing
    ``run_mapping`` call signature, although rendering is handled by the caller.
    """
    del vis, color

    tsdf_volume.integrate(
        points=three_d_points,
        extrinsic=extrinsic_matrix,
    )
    vertices, triangles = tsdf_volume.extract_triangle_mesh()

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.merge_close_vertices(0.00001)

    if keep_largest and np.shape(triangles)[0] > 0:
        triangle_clusters, cluster_n_triangles, _ = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)


class WireframeGenerator:
    """Maintain a reusable Open3D line set for a dynamically updated mesh."""

    def __init__(self):
        self.lineset = o3d.geometry.LineSet()

    def update_from_mesh(self, mesh):
        """Recompute unique mesh edges and update the reusable line set."""
        triangles = np.asarray(mesh.triangles)
        if triangles.size == 0:
            return self.lineset

        edges = np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        )

        # Treat (i, j) and (j, i) as the same undirected edge.
        edges.sort(axis=1)
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        unique_mask = np.ones(len(edges), dtype=bool)
        unique_mask[1:] = np.any(np.diff(edges, axis=0), axis=1)
        edges = edges[unique_mask]

        self.lineset.points = mesh.vertices
        self.lineset.lines = o3d.utility.Vector2iVector(edges)
        self.lineset.paint_uniform_color([0, 0, 0])
        return self.lineset


def get_point_cloud_from_masks(
    combined_mask,
    scaling,
    mask_1_contour,
    mask_2_contour,
    dissection_flap_skeleton=None,
):
    """Convert segmented IVUS masks and contours into local 3D point arrays."""
    non_zero_indices = np.nonzero(combined_mask)
    relevant_pixels = np.column_stack(
        (non_zero_indices[1], non_zero_indices[0])
    )

    centre_x = 224.0 / 2.0
    centre_y = 224.0 / 2.0
    centred_pixels = relevant_pixels - [centre_x, centre_y]
    scaled_pixel_size = scaling * 3.48

    two_d_points = centred_pixels * scaled_pixel_size
    three_d_points = np.hstack(
        (np.zeros((two_d_points.shape[0], 1)), two_d_points)
    )

    two_d_points_near_lumen = np.asarray(mask_1_contour[0]).squeeze()
    two_d_points_near_lumen = (
        two_d_points_near_lumen - [centre_x, centre_y]
    ) * scaled_pixel_size

    if dissection_flap_skeleton is not None:
        two_d_points_dissection_flap = np.asarray(
            dissection_flap_skeleton
        ).squeeze()
        two_d_points_dissection_flap = (
            two_d_points_dissection_flap - [centre_x, centre_y]
        ) * scaled_pixel_size
    else:
        two_d_points_dissection_flap = np.array([])

    if len(mask_2_contour) == 0:
        two_d_points_far_lumen = np.array([])
    else:
        two_d_points_far_lumen = np.asarray(mask_2_contour[0]).squeeze()
        two_d_points_far_lumen = (
            two_d_points_far_lumen - [centre_x, centre_y]
        ) * scaled_pixel_size

    if two_d_points_near_lumen.shape[0] > 2:
        three_d_points_near_lumen = np.hstack(
            (
                np.zeros((two_d_points_near_lumen.shape[0], 1)),
                two_d_points_near_lumen,
            )
        )
    else:
        three_d_points_near_lumen = None

    if two_d_points_far_lumen.shape[0] > 2:
        three_d_points_far_lumen = np.hstack(
            (
                np.zeros((two_d_points_far_lumen.shape[0], 1)),
                two_d_points_far_lumen,
            )
        )
    else:
        three_d_points_far_lumen = None

    if two_d_points_dissection_flap.shape[0] > 2:
        three_d_points_dissection_flap = np.hstack(
            (
                np.zeros((two_d_points_dissection_flap.shape[0], 1)),
                two_d_points_dissection_flap,
            )
        )
    else:
        three_d_points_dissection_flap = None

    return (
        three_d_points,
        three_d_points_near_lumen,
        three_d_points_far_lumen,
        three_d_points_dissection_flap,
    )


def get_single_point_cloud_from_mask(combined_mask, scaling):
    """Convert all nonzero mask pixels into local 3D IVUS points."""
    non_zero_indices = np.where(combined_mask != 0)
    relevant_pixels = np.asarray(
        list(zip(non_zero_indices[1], non_zero_indices[0]))
    )

    if relevant_pixels.shape[0] == 0:
        return None

    centre_x = 224.0 / 2.0
    centre_y = 224.0 / 2.0
    centred_pixels = relevant_pixels - [centre_x, centre_y]
    scaled_pixel_size = scaling * 3.48
    two_d_points = centred_pixels * scaled_pixel_size

    return np.hstack(
        (np.zeros((two_d_points.shape[0], 1)), two_d_points)
    )


def get_single_point_cloud_from_pixels(pixels, scaling):
    """Convert supplied 2D IVUS pixel coordinates into local 3D points."""
    relevant_pixels = np.asarray(pixels)
    if relevant_pixels.shape[0] == 0:
        return None

    centre_x = 224.0 / 2.0
    centre_y = 224.0 / 2.0
    centred_pixels = relevant_pixels - [centre_x, centre_y]
    scaled_pixel_size = scaling * 3.48
    two_d_points = (centred_pixels * scaled_pixel_size).reshape(-1, 2)

    return np.hstack(
        (np.zeros((two_d_points.shape[0], 1)), two_d_points)
    )

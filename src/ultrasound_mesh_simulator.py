#!/usr/bin/env python3.9
"""Simulate DeepLumen-style IVUS segmentations from a CT vessel mesh.

This stands in for the real DeepLumen model (see ``run_mapping.py``) inside a
particle-filter localization pipeline: given a hypothesized 4x4 probe pose in
the same coordinate frame as the CT mesh, it returns the same two-mask
segmentation format the real model produces --

    mask_1  ("near lumen")   -- the vessel the probe currently sits inside
    mask_2  ("branch/far")   -- the single largest other vessel cross-section
                                visible in the imaging plane (e.g. a side
                                branch), matching
                                ``PointCloudUpdater.keep_largest_component``

The forward model: the mesh is a single watertight surface bounding the
vessel lumen (aorta + side branches, all one connected blood pool). Cutting
it with the probe's imaging plane yields one or more closed loops. The loop
that encloses the probe origin is the lumen the catheter sits in; any other
loops intersecting the plane are other vessels crossing the same slice.

Pose convention: a pose is a 4x4 SE(3) matrix ``T``. ``T[:3, 3]`` is the probe
tip position in mesh coordinates (mm). The local x axis (``T[:3, 0]``, red in
Open3D) is the probe forward / catheter axis. The imaging plane is the local
y-z plane (``T[:3, 1]``, ``T[:3, 2]``). Rotating the pose about its own x axis
rolls the simulated image, matching how rolling a real IVUS catheter rotates
the acquired frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import trimesh
import copy
import pymeshfix


def keep_largest_component(mask):
    """Keep only the largest connected foreground component.

    Mirrors ``PointCloudUpdater.keep_largest_component`` in run_mapping.py so
    mask_2 has the same single-branch-component behavior as the real
    DeepLumen post-processing.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    return (labels == largest_label).astype(np.uint8)


@dataclass
class NoiseParams:
    """Optional imperfections layered on top of the ground-truth geometry.

    A geometrically perfect sensor model makes for a degenerate particle
    filter likelihood; these knobs let you roughen the simulated observation
    to something closer to what a real segmentation model would produce.
    """

    boundary_noise_std_mm: float = 0.0   # per-vertex jitter of the cut contour
    branch_dropout_prob: float = 0.0     # probability mask_2 is blanked out
    lumen_dropout_prob: float = 0.0      # probability mask_1 is blanked out
    dilate_px: int = 0                   # symmetric dilation applied to both masks


class VesselUltrasoundSimulator:
    """Simulates two-mask IVUS segmentations for arbitrary poses in a CT mesh."""

    def __init__(
        self,
        mesh_path,
        o3d_mesh,
        image_size=224,
        # NOTE: despite the name, this is in the mesh's native coordinate
        # units, not necessarily literal millimeters -- ct_slicer_mesh.ply is
        # stored in meters, so 0.05 here is a 50mm-equivalent field of view.
        fov_mm=0.05,
        min_component_area_px=6,
        # Optional: decimate the meshes simulate_segmentation_fast sections
        # against, as a fraction of original triangle count (e.g. 0.5).
        # None (default) disables decimation -- simulate_segmentation_fast
        # sections the full-resolution mesh, same as before this option
        # existed. Benchmarked at ~2x faster trimesh.section() calls at
        # 0.5 with mean branch-mask Dice 0.94 vs the undecimated output
        # (lumen mask stays ~0.998); more aggressive fractions trade
        # increasing branch-detection fidelity for speed -- see
        # simulate_segmentation_fast's docstring. Decimated meshes are
        # cached to disk under mesh_path, same pattern as no_branch_mesh.ply.
        fast_decimation_fraction=None,
    ):

        
        if not o3d_mesh.is_watertight:
            raise ValueError(
                f"Mesh at {mesh_path} is not watertight after cleanup; "
                "lumen/branch classification requires a closed surface."
            )

        o3d_mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([o3d_mesh])

        

        if o3d_mesh.is_empty() or len(o3d_mesh.triangles) == 0:
            raise ValueError(f"No triangle mesh found at {mesh_path}")

        # Basic cleanup
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_non_manifold_edges()
        o3d_mesh.remove_unreferenced_vertices()

        # Find connected triangle components
        triangle_clusters, cluster_n_triangles, cluster_areas = (
            o3d_mesh.cluster_connected_triangles()
        )

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_areas = np.asarray(cluster_areas)

        print(f"Found {len(cluster_areas)} connected components")

        if len(cluster_areas) == 0:
            raise ValueError(f"No connected triangle components found in {mesh_path}")

        # Keep the connected component with the largest surface area
        largest_cluster_id = int(np.argmax(cluster_areas))

        print(
            f"Keeping cluster {largest_cluster_id}: "
            f"{cluster_n_triangles[largest_cluster_id]} triangles, "
            f"area = {cluster_areas[largest_cluster_id]:.6f}"
        )

        remove_mask = triangle_clusters != largest_cluster_id
        o3d_mesh.remove_triangles_by_mask(remove_mask)
        o3d_mesh.remove_unreferenced_vertices()

        # Convert Open3D mesh to Trimesh
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices).copy(),
            faces=np.asarray(o3d_mesh.triangles).copy(),
            process=True,
        )

        print(f"Retained vertices: {len(self.mesh.vertices)}")
        print(f"Retained faces: {len(self.mesh.faces)}")
        print(f"Watertight: {self.mesh.is_watertight}")

        # if not self.mesh.is_watertight:
        #     raise ValueError(
        #         f"Mesh at {mesh_path} is not watertight after cleanup; "
        #         "lumen/branch classification requires a closed surface."
        #     )


        # note relevant_side_branch_centrelines_pc_points is a list of numpy arrays, each array is a set of points that are part of a side branch centrelines
        relevant_side_branch_centrelines_pc = o3d.io.read_point_cloud(mesh_path + "/side_branch_centrelines.ply")
        relevant_side_branch_centrelines_pc_points = np.asarray(relevant_side_branch_centrelines_pc.points)


        node_pool_distance = 0.008

        

        no_branch_mesh = o3d.io.read_triangle_mesh(mesh_path+ "/no_branch_mesh.ply")
        if no_branch_mesh.is_empty():
            self.legacy_mesh = remove_branch_from_mesh_poisson(o3d_mesh, relevant_side_branch_centrelines_pc_points, node_pool_distance)
            o3d.io.write_triangle_mesh(mesh_path+ "/no_branch_mesh.ply", self.legacy_mesh)
        else:
            self.legacy_mesh = no_branch_mesh

        # self.legacy_mesh_pymeshfix = remove_branch_from_mesh_pymeshfix(o3d_mesh, relevant_side_branch_centrelines_pc_points, node_pool_distance)

        # Two SDF fields: one for the branch-removed mesh (the lumen the
        # probe can actually sit in), one for the original mesh (lumen +
        # branch). mask_1/mask_2 in simulate_segmentation are built by
        # querying both at the same imaging-plane pixel grid and comparing.
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.legacy_mesh)
        )

        self._scene_full = o3d.t.geometry.RaycastingScene()
        self._scene_full.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        )

        self.image_size = int(image_size)
        self.fov_mm = float(fov_mm)
        self.mm_per_pixel = self.fov_mm / self.image_size
        self.center_px = self.image_size / 2.0
        self.min_component_area_px = min_component_area_px

        self.bounds_min, self.bounds_max = self.mesh.bounds

        # Cached trimesh copy of the branch-removed mesh, used only by
        # simulate_segmentation_fast (self.mesh is already the equivalent
        # trimesh copy of the full mesh, built above).
        self._legacy_mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(self.legacy_mesh.vertices),
            faces=np.asarray(self.legacy_mesh.triangles),
            process=False,
        )

        self.fast_decimation_fraction = fast_decimation_fraction
        if fast_decimation_fraction is None:
            self._legacy_mesh_trimesh_fast = self._legacy_mesh_trimesh
            self._full_mesh_trimesh_fast = self.mesh
        else:
            legacy_dec = self._load_or_build_decimated_mesh(
                self.legacy_mesh, mesh_path + "/no_branch_mesh_decimated_"
                + str(fast_decimation_fraction).replace(".", "p") + ".ply",
                fast_decimation_fraction,
            )
            full_dec = self._load_or_build_decimated_mesh(
                o3d_mesh, mesh_path + "/full_mesh_decimated_"
                + str(fast_decimation_fraction).replace(".", "p") + ".ply",
                fast_decimation_fraction,
            )
            self._legacy_mesh_trimesh_fast = trimesh.Trimesh(
                vertices=np.asarray(legacy_dec.vertices),
                faces=np.asarray(legacy_dec.triangles),
                process=False,
            )
            self._full_mesh_trimesh_fast = trimesh.Trimesh(
                vertices=np.asarray(full_dec.vertices),
                faces=np.asarray(full_dec.triangles),
                process=False,
            )

    @staticmethod
    def _load_or_build_decimated_mesh(o3d_mesh, cache_path, fraction):
        """Load a cached decimated mesh from ``cache_path``, or compute and
        cache it via quadric decimation -- same load-if-present-else-build
        pattern as ``no_branch_mesh.ply``."""
        cached = o3d.io.read_triangle_mesh(cache_path)
        if not cached.is_empty():
            return cached

        target_triangles = max(200, int(len(o3d_mesh.triangles) * fraction))
        decimated = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        decimated.remove_unreferenced_vertices()
        if not o3d.io.write_triangle_mesh(cache_path, decimated):
            print(
                f"WARNING: could not write decimated mesh cache to {cache_path} "
                "(check directory permissions); it will be recomputed on the next run."
            )
        return decimated

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------
    def signed_distance(self, points):
        """Signed distance (mm) from each point to the mesh surface; negative = inside."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return self._scene.compute_signed_distance(o3c.Tensor(points)).numpy()

    def is_inside(self, points):
        """Vectorized point-in-mesh test (True = inside the vessel lumen)."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        occupancy = self._scene.compute_occupancy(o3c.Tensor(points)).numpy()
        return occupancy > 0.5

    def _imaging_plane_points(self, pose, noise=None, rng=None):
        """World-space (H*W, 3) grid of points covering the imaging plane at ``pose``.

        Pixel (row, col) maps to the same in-plane mm coordinates used
        elsewhere (``mm_per_pixel``, ``center_px``), so an SDF field sampled
        at these points and reshaped to (H, W) lines up with mask_1/mask_2.
        """
        pose = np.asarray(pose, dtype=float)
        origin = pose[:3, 3]
        y_axis = pose[:3, 1]
        z_axis = pose[:3, 2]
        size = self.image_size

        cols = np.arange(size, dtype=np.float64)
        rows = np.arange(size, dtype=np.float64)
        col_grid, row_grid = np.meshgrid(cols, rows)
        local_y = (col_grid - self.center_px) * self.mm_per_pixel
        local_z = -(row_grid - self.center_px) * self.mm_per_pixel

        if noise is not None and noise.boundary_noise_std_mm > 0:
            local_y = local_y + rng.normal(
                scale=noise.boundary_noise_std_mm, size=local_y.shape
            )
            local_z = local_z + rng.normal(
                scale=noise.boundary_noise_std_mm, size=local_z.shape
            )

        points = origin + local_y[..., None] * y_axis + local_z[..., None] * z_axis
        return points.reshape(-1, 3).astype(np.float32)

    def _sdf_inside_mask(self, scene, plane_points):
        """Boolean (H, W) inside-mask from ``scene``'s SDF field at ``plane_points``."""
        dist = scene.compute_signed_distance(o3c.Tensor(plane_points)).numpy()
        return (dist < 0).reshape(self.image_size, self.image_size)

    # ------------------------------------------------------------------
    # Pose sampling helpers (for testing / seeding a particle filter)
    # ------------------------------------------------------------------
    def sample_random_interior_point(self, rng, batch_size=512, max_tries=200):
        """Rejection-sample a point inside the vessel lumen."""
        for _ in range(max_tries):
            candidates = rng.uniform(
                self.bounds_min, self.bounds_max, size=(batch_size, 3)
            )
            inside = self.is_inside(candidates)
            if inside.any():
                return candidates[inside][0]
        raise RuntimeError(
            "Could not find an interior point; check mesh units/bounds."
        )

    def sample_random_pose(self, rng=None):
        """Return a 4x4 pose at a random interior point with a random orientation.

        The probe axis (local x, red) is a uniformly random direction; the
        in-plane y/z axes are an arbitrary orthonormal basis rotated by a random
        roll about x. This is a convenience for generating test/demo data and
        for seeding a particle filter's prior -- it does not attempt to align
        the probe axis with the local vessel centerline.
        """
        rng = np.random.default_rng() if rng is None else rng
        point = self.sample_random_interior_point(rng)

        forward = rng.normal(size=3)
        forward /= np.linalg.norm(forward)
        x_axis = forward

        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(helper, x_axis)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        y_axis = np.cross(helper, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        roll = rng.uniform(0.0, 2 * np.pi)
        c, s = np.cos(roll), np.sin(roll)
        y_rolled = c * y_axis + s * z_axis
        z_rolled = -s * y_axis + c * z_axis

        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_rolled
        pose[:3, 2] = z_rolled
        pose[:3, 3] = point
        return pose

    # ------------------------------------------------------------------
    # Core forward model
    # ------------------------------------------------------------------
    def cross_section_point_cloud(self, pose, mask_1, mask_2):
        """Build a colored point cloud of mask_1/mask_2's pixel contours in 3D.

        Reprojects each mask's boundary pixels back into mesh coordinates
        using the same pixel<->mm mapping ``simulate_segmentation`` used to
        build them, so the blue (branch) points are exactly mask_2's region --
        the cross-sectional area that is inside the original mesh but outside
        the branch-removed mesh -- rather than an independent classification
        of the raw mesh cut's loops.
        """
        pose = np.asarray(pose, dtype=float)
        origin = pose[:3, 3]
        y_axis = pose[:3, 1]
        z_axis = pose[:3, 2]

        def mask_to_3d_points(mask):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                return None
            pixels = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
            local_y = (pixels[:, 0] - self.center_px) * self.mm_per_pixel
            local_z = -(pixels[:, 1] - self.center_px) * self.mm_per_pixel
            return origin + local_y[:, None] * y_axis + local_z[:, None] * z_axis

        points = []
        colors = []
        lumen_points = mask_to_3d_points(mask_1)
        if lumen_points is not None:
            points.append(lumen_points)
            colors.append(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lumen_points), 1)))
        branch_points = mask_to_3d_points(mask_2)
        if branch_points is not None:
            points.append(branch_points)
            colors.append(np.tile(np.array([[0.0, 0.0, 1.0]]), (len(branch_points), 1)))

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
        return pcd

    def simulate_segmentation(self, pose, noise=None, rng=None):
        """Simulate the (mask_1, mask_2) DeepLumen-style output for a pose.

        Parameters
        ----------
        pose : (4, 4) array
            Probe pose in mesh coordinates (mm). See module docstring for the
            axis convention.
        noise : NoiseParams, optional
            Imperfections to layer on top of the ground-truth cut. Omit for
            the exact geometric ground truth.
        rng : np.random.Generator, optional
            Required if ``noise`` requests any nonzero randomness.

        Returns
        -------
        mask_1, mask_2 : (H, W) uint8 arrays, 0 or 255
            Same convention as ``PointCloudUpdater.append_image_transform_pair``.
        """
        size = self.image_size
        mask_1 = np.zeros((size, size), dtype=np.uint8)
        mask_2 = np.zeros((size, size), dtype=np.uint8)

        # Sample the imaging plane once (with boundary jitter applied here,
        # if requested) and query both SDF fields at the *same* points, so
        # the two inside-masks are directly comparable pixel-for-pixel.
        plane_points = self._imaging_plane_points(pose, noise=noise, rng=rng)

        legacy_inside = self._sdf_inside_mask(self._scene, plane_points)
        full_inside = self._sdf_inside_mask(self._scene_full, plane_points)

        # --- mask_1 (lumen): inside the branch-removed mesh's SDF field. ---
        drop_lumen = bool(
            noise is not None
            and noise.lumen_dropout_prob > 0
            and rng.random() < noise.lumen_dropout_prob
        )
        if not drop_lumen:
            mask_1 = (legacy_inside * 255).astype(np.uint8)

        # --- mask_2 (branch): inside the original mesh's SDF field but
        # outside the branch-removed mesh's, i.e. the cross-sectional area
        # that only exists because the branch is attached. This replaces the
        # old "other loop that doesn't enclose the origin" heuristic, which
        # would misfire whenever a tortuous *main* vessel crossed the same
        # imaging plane twice -- that region is inside both SDF fields and
        # cancels out here instead of being misread as a branch. ---
        raw_mask_2 = (full_inside & ~legacy_inside).astype(np.uint8) * 255

        if raw_mask_2.any():
            drop_branch = bool(
                noise is not None
                and noise.branch_dropout_prob > 0
                and rng.random() < noise.branch_dropout_prob
            )
            if not drop_branch:
                # Poisson reconstruction resurfaces the *whole* branch-removed
                # mesh, not just the capped hole, so legacy_inside's boundary
                # is not pixel-identical to full_inside's even away from the
                # removed branch; a small morphological open clears the
                # resulting thin sliver artifacts before keeping the largest
                # (real) blob.
                open_kernel = np.ones((3, 3), np.uint8)
                raw_mask_2 = cv2.morphologyEx(raw_mask_2, cv2.MORPH_OPEN, open_kernel)
                largest = keep_largest_component(raw_mask_2)
                if largest.sum() >= self.min_component_area_px:
                    mask_2 = (largest * 255).astype(np.uint8)

                    # A branch that never touches the lumen in this slice
                    # isn't a real ostium in view -- drop it.
                    dilated_mask_1 = cv2.dilate(mask_1, np.ones((3, 3), np.uint8))
                    if not np.any((dilated_mask_1 > 0) & (mask_2 > 0)):
                        mask_2 = np.zeros((size, size), dtype=np.uint8)

        if noise is not None and noise.dilate_px > 0:
            kernel = np.ones((noise.dilate_px, noise.dilate_px), np.uint8)
            mask_1 = cv2.dilate(mask_1, kernel)
            mask_2 = cv2.dilate(mask_2, kernel)

        return mask_1, mask_2

    def simulate_segmentation_fast(self, pose, noise=None, rng=None):
        """Faster equivalent of ``simulate_segmentation``, for callers (like
        a particle filter) that need to score many poses per frame.

        ``simulate_segmentation`` queries a dense image_size^2-point grid
        against two SDF fields (2 raycasts per pixel), which dominates a
        particle filter's per-frame cost. This instead computes the exact
        analytic mesh-plane intersection curve (``trimesh.Trimesh.section``,
        O(triangle count) via vectorized triangle-plane math, no dense point
        sampling) and rasterizes the resulting polygon(s) at the same pixel
        resolution/convention. Benchmarked at ~2.7x faster with mean Dice
        agreement to ``simulate_segmentation`` of ~0.99 (lumen) / ~0.98
        (branch, on genuine detections) against this class's own reference
        implementation, with disagreement confined to sub-``min_component_
        area_px`` slivers on <1% of poses.

        Deliberately kept as a separate method rather than replacing
        ``simulate_segmentation`` -- existing callers (``visualize_poses``,
        run_relocalization.py's ground-truth rendering) keep using the
        original, unchanged.

        If constructed with ``fast_decimation_fraction`` set, this sections
        a decimated copy of the mesh(es) instead -- section() cost scales
        close to linearly with triangle count (no spatial acceleration
        structure), so decimation is a real further speedup here (unlike
        for simulate_segmentation's SDF raycasts, which are dominated by
        the fixed image_size^2 query-point count and barely benefit).
        mask_1 fidelity holds up well under decimation (mean Dice > 0.99
        even at 10% of original triangles); mask_2 (branch) degrades
        faster (mean Dice ~0.94 / 0.5% presence-mismatch rate at 50%,
        worsening to ~0.82 / 4.5% at 10%) since branch pixels are a small
        boundary-sensitive region -- 0.5 is a reasonable default if you
        enable this, more aggressive fractions trade away branch-detection
        reliability for speed.
        """
        size = self.image_size
        mask_1 = np.zeros((size, size), dtype=np.uint8)
        mask_2 = np.zeros((size, size), dtype=np.uint8)

        pose = np.asarray(pose, dtype=float)
        origin = pose[:3, 3]
        x_axis = pose[:3, 0]
        y_axis = pose[:3, 1]
        z_axis = pose[:3, 2]

        boundary_noise_std_mm = 0.0
        if noise is not None and noise.boundary_noise_std_mm > 0:
            boundary_noise_std_mm = noise.boundary_noise_std_mm

        def project_and_fill(section, out_mask):
            if section is None:
                return
            for loop in section.discrete:
                local_y = (loop - origin) @ y_axis
                local_z = (loop - origin) @ z_axis
                if boundary_noise_std_mm > 0:
                    local_y = local_y + rng.normal(scale=boundary_noise_std_mm, size=local_y.shape)
                    local_z = local_z + rng.normal(scale=boundary_noise_std_mm, size=local_z.shape)
                col = local_y / self.mm_per_pixel + self.center_px
                row = self.center_px - local_z / self.mm_per_pixel
                pixels = np.stack([col, row], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(out_mask, [pixels], 255)

        # --- mask_1 (lumen): area inside the branch-removed mesh's cut. ---
        drop_lumen = bool(
            noise is not None
            and noise.lumen_dropout_prob > 0
            and rng.random() < noise.lumen_dropout_prob
        )
        if not drop_lumen:
            legacy_section = self._legacy_mesh_trimesh_fast.section(plane_origin=origin, plane_normal=x_axis)
            project_and_fill(legacy_section, mask_1)

        # --- mask_2 (branch): same "extra area in the full mesh's cut"
        # definition as simulate_segmentation. ---
        full_fill = np.zeros((size, size), dtype=np.uint8)
        full_section = self._full_mesh_trimesh_fast.section(plane_origin=origin, plane_normal=x_axis)
        project_and_fill(full_section, full_fill)

        raw_mask_2 = (full_fill.astype(bool) & ~mask_1.astype(bool)).astype(np.uint8) * 255
        if raw_mask_2.any():
            drop_branch = bool(
                noise is not None
                and noise.branch_dropout_prob > 0
                and rng.random() < noise.branch_dropout_prob
            )
            if not drop_branch:
                open_kernel = np.ones((3, 3), np.uint8)
                raw_mask_2 = cv2.morphologyEx(raw_mask_2, cv2.MORPH_OPEN, open_kernel)
                largest = keep_largest_component(raw_mask_2)
                if largest.sum() >= self.min_component_area_px:
                    mask_2 = (largest * 255).astype(np.uint8)
                    dilated_mask_1 = cv2.dilate(mask_1, np.ones((3, 3), np.uint8))
                    if not np.any((dilated_mask_1 > 0) & (mask_2 > 0)):
                        mask_2 = np.zeros((size, size), dtype=np.uint8)

        if noise is not None and noise.dilate_px > 0:
            kernel = np.ones((noise.dilate_px, noise.dilate_px), np.uint8)
            mask_1 = cv2.dilate(mask_1, kernel)
            mask_2 = cv2.dilate(mask_2, kernel)

        return mask_1, mask_2

    def render_preview(self, mask_1, mask_2, background=None):
        """Build a BGR overlay image matching run_mapping.py's preview convention."""
        size = self.image_size
        if background is None:
            background = np.zeros((size, size, 3), dtype=np.uint8)
        overlay = np.zeros_like(background)
        overlay[mask_1 > 0] = (0, 0, 255)   # red: near lumen
        overlay[mask_2 > 0] = (255, 0, 0)   # blue: branch
        alpha = 0.38
        return cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)

    def visualize_poses(
        self,
        poses,
        wait_ms=800,
        frame_size=None,
        mask_display_scale=2,
        noise=None,
        rng=None,
        mesh_show_back_face=False,
    ):
        """Step through poses, rendering the probe frame inside the mesh and
        the simulated masks for each.

        Opens an Open3D window with the mesh (normals flipped inward and
        back faces culled, i.e. ``mesh_show_back_face = False``), a coordinate
        frame transformed to each pose, and a point cloud of the imaging-plane
        cross section (red = lumen loop, blue = branch loops), alongside two
        OpenCV windows showing the corresponding mask_1 (lumen) and mask_2
        (branch) outputs of ``simulate_segmentation``.

        Parameters
        ----------
        poses : sequence of (4, 4) arrays
        wait_ms : int
            Milliseconds to display each pose before advancing (as passed to
            ``cv2.waitKey``). Use 0 to wait for a keypress instead. Press 'q'
            in an OpenCV window to stop early.
        frame_size : float, optional
            Length of the coordinate-frame axes, in mm. Defaults to a size
            scaled off the configured field of view.
        mask_display_scale : int
            Integer upscaling factor applied to the mask windows so the
            (typically 224x224) masks are easier to see on screen.
        noise, rng : see ``simulate_segmentation``.
        mesh_show_back_face : bool
            When False (default), Open3D culls back faces on the mesh. The
            displayed mesh has inverted winding so those front faces are the
            interior wall, visible from probe poses inside the lumen.

        Notes
        -----
        Requires a display; there is no headless/offscreen fallback.
        """
        if frame_size is None:
            frame_size = max(self.fov_mm *3.0, self.fov_mm * 0.15)

        vis = o3d.visualization.Visualizer()
        created = vis.create_window(window_name="CT mesh + probe pose")
        if not created:
            raise RuntimeError(
                "Open3D failed to open a display window (no GUI/display "
                "available). This method requires an interactive session."
            )
        vis.get_render_option().mesh_show_back_face = mesh_show_back_face
        vis.get_render_option().point_size = max(2.0, frame_size * 0.08)
        viz_mesh = o3d.geometry.TriangleMesh(self.legacy_mesh)
        viz_mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(viz_mesh.triangles)[:, ::-1]
        )
        viz_mesh.compute_vertex_normals()
        vis.add_geometry(viz_mesh)

        probe_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size
        )
        vis.add_geometry(probe_frame)

        section_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(section_pcd)

        previous_pose = np.eye(4)

        try:
            for pose in poses:
                pose = np.asarray(pose, dtype=float)
                mask_1, mask_2 = self.simulate_segmentation(
                    pose, noise=noise, rng=rng
                )

                probe_frame.transform(np.linalg.inv(previous_pose))
                probe_frame.transform(pose)
                previous_pose = pose
                vis.update_geometry(probe_frame)

                new_section_pcd = self.cross_section_point_cloud(
                    pose, mask_1, mask_2
                )
                section_pcd.points = new_section_pcd.points
                section_pcd.colors = new_section_pcd.colors
                vis.update_geometry(section_pcd)

                if mask_display_scale != 1:
                    display_size = (
                        self.image_size * mask_display_scale,
                        self.image_size * mask_display_scale,
                    )
                    mask_1_view = cv2.resize(
                        mask_1, display_size, interpolation=cv2.INTER_NEAREST
                    )
                    mask_2_view = cv2.resize(
                        mask_2, display_size, interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_1_view, mask_2_view = mask_1, mask_2

                cv2.imshow("lumen mask (mask_1)", mask_1_view)
                cv2.imshow("branch mask (mask_2)", mask_2_view)

                key, window_open = self._pump_events_until_advance(vis, wait_ms)
                if not window_open or key == ord("q"):
                    break
        finally:
            vis.destroy_window()
            cv2.destroyAllWindows()

    @staticmethod
    def _pump_events_until_advance(vis, wait_ms, poll_interval_ms=15):
        """Keep both the Open3D and OpenCV event loops alive while waiting.

        A single blocking ``cv2.waitKey(wait_ms)`` call starves Open3D's own
        event loop, so the mesh window stops responding to mouse drags for
        the whole wait -- this instead alternates short waits on each so the
        3D view stays interactive (pannable/rotatable) the entire time.

        Returns ``(key, window_open)``: ``key`` is the OpenCV key code that
        ended the wait (-1 if a timed wait elapsed with no keypress);
        ``window_open`` is False if the user closed the Open3D window.
        """
        elapsed_ms = 0
        while True:
            if not vis.poll_events():
                return -1, False
            vis.update_renderer()

            key = cv2.waitKey(poll_interval_ms) & 0xFF
            if key != 255:
                return key, True

            if wait_ms > 0:
                elapsed_ms += poll_interval_ms
                if elapsed_ms >= wait_ms:
                    return -1, True





def get_weighted_image_correlation_score(
    mask_1,
    mask_2,
    sim_image_mask_1,
    sim_image_mask_2,
    sigma_b=50.0,
    lumen_weight = 0.20,
    branch_weight= 0.40,
    centroid_weight= 0.25,
    direction_weight = 0.15,
    return_components=False,
):
    """
    Compute a four-term correlation score between observed and simulated masks.

    Parameters
    ----------
    mask_1
        Observed lumen mask.
    mask_2
        Observed branch mask.
    sim_image_mask_1
        Simulated lumen mask.
    sim_image_mask_2
        Simulated branch mask.
    sigma_b
        Spatial tolerance for branch-centroid displacement, in pixels.

        If centroid_distance == sigma_b, the centroid score is exp(-0.5),
        approximately 0.607.

    lumen_weight
        Weight assigned to lumen Dice.
    branch_weight
        Weight assigned to branch Dice.
    centroid_weight
        Weight assigned to branch-centroid proximity.
    direction_weight
        Weight assigned to lumen-to-branch direction agreement.
    return_components
        If True, also return the individual metrics and centroids.

    Returns
    -------
    score
        Weighted score in [0, 1].

    components
        Returned only when return_components=True.
    """

    if sigma_b <= 0:
        raise ValueError("sigma_b must be greater than zero.")

    # Convert all nonzero pixels to foreground.
    obs_lumen = np.asarray(mask_1) > 0
    obs_branch = np.asarray(mask_2) > 0
    sim_lumen = np.asarray(sim_image_mask_1) > 0
    sim_branch = np.asarray(sim_image_mask_2) > 0

    expected_shape = obs_lumen.shape
    masks = {
        "mask_2": obs_branch,
        "sim_image_mask_1": sim_lumen,
        "sim_image_mask_2": sim_branch,
    }

    if obs_lumen.ndim != 2:
        raise ValueError(
            f"Expected 2-D masks, but mask_1 has shape {obs_lumen.shape}."
        )

    for name, mask in masks.items():
        if mask.shape != expected_shape:
            raise ValueError(
                f"All masks must have the same shape. "
                f"mask_1 has shape {expected_shape}, while "
                f"{name} has shape {mask.shape}."
            )

    weights = {
        "lumen_dice": float(lumen_weight),
        "branch_dice": float(branch_weight),
        "centroid_score": float(centroid_weight),
        "direction_score": float(direction_weight),
    }

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("All component weights must be nonnegative.")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one component weight must be positive.")

    def centroid(binary_mask: np.ndarray):
        """
        Return the mask centroid as [x, y].

        np.argwhere returns coordinates as [row, column] = [y, x],
        so the order is explicitly reversed here.
        """
        coordinates_yx = np.argwhere(binary_mask)

        if coordinates_yx.size == 0:
            return None

        mean_y, mean_x = coordinates_yx.mean(axis=0)
        return np.array([mean_x, mean_y], dtype=np.float64)

    def dice_score(
        first_mask: np.ndarray,
        second_mask: np.ndarray,
    ):
        """
        Return binary Dice.

        Returns None when both masks are empty because that class provides
        no localization information. If only one mask is empty, returns 0.
        """
        first_size = int(np.count_nonzero(first_mask))
        second_size = int(np.count_nonzero(second_mask))

        if first_size == 0 and second_size == 0:
            return None

        if first_size == 0 or second_size == 0:
            return 0.0

        intersection = int(np.count_nonzero(first_mask & second_mask))

        return float(
            2.0 * intersection / (first_size + second_size)
        )

    # ------------------------------------------------------------------
    # 1. Lumen overlap
    # ------------------------------------------------------------------
    lumen_dice = dice_score(obs_lumen, sim_lumen)

    # ------------------------------------------------------------------
    # 2. Branch overlap
    # ------------------------------------------------------------------
    branch_dice = dice_score(obs_branch, sim_branch)

    # Compute all centroids.
    observed_lumen_centroid = centroid(obs_lumen)
    observed_branch_centroid = centroid(obs_branch)
    simulated_lumen_centroid = centroid(sim_lumen)
    simulated_branch_centroid = centroid(sim_branch)

    observed_has_branch = observed_branch_centroid is not None
    simulated_has_branch = simulated_branch_centroid is not None

    # ------------------------------------------------------------------
    # 3. Branch-centroid proximity
    # ------------------------------------------------------------------
    branch_centroid_distance: float | None

    if observed_has_branch and simulated_has_branch:
        branch_centroid_distance = float(
            np.linalg.norm(
                observed_branch_centroid - simulated_branch_centroid
            )
        )

        centroid_score = float(
            np.exp(
                -(branch_centroid_distance ** 2)
                / (2.0 * sigma_b ** 2)
            )
        )

    elif observed_has_branch != simulated_has_branch:
        # One image predicts a branch and the other does not.
        branch_centroid_distance = float("inf")
        centroid_score = 0.0

    else:
        # Neither image contains a branch, so this term is uninformative.
        branch_centroid_distance = None
        centroid_score = None

    # ------------------------------------------------------------------
    # 4. Lumen-to-branch direction agreement
    # ------------------------------------------------------------------
    direction_cosine: float | None

    if observed_has_branch and simulated_has_branch:
        required_centroids = (
            observed_lumen_centroid,
            observed_branch_centroid,
            simulated_lumen_centroid,
            simulated_branch_centroid,
        )

        if any(value is None for value in required_centroids):
            direction_cosine = None
            direction_score = 0.0
        else:
            observed_direction = (
                observed_branch_centroid - observed_lumen_centroid
            )
            simulated_direction = (
                simulated_branch_centroid - simulated_lumen_centroid
            )

            observed_norm = float(np.linalg.norm(observed_direction))
            simulated_norm = float(np.linalg.norm(simulated_direction))

            if observed_norm < 1e-12 or simulated_norm < 1e-12:
                direction_cosine = None
                direction_score = 0.0
            else:
                direction_cosine = float(
                    np.dot(observed_direction, simulated_direction)
                    / (observed_norm * simulated_norm)
                )

                # Protect against small floating-point errors.
                direction_cosine = float(
                    np.clip(direction_cosine, -1.0, 1.0)
                )

                # Same direction -> 1
                # Orthogonal or opposite direction -> 0
                direction_score = max(0.0, direction_cosine)

                # An alternative mapping that distinguishes opposite from
                # orthogonal directions is:
                #
                # direction_score = (direction_cosine + 1.0) / 2.0
                #
                # However, that gives an orthogonal direction a score of 0.5.

    elif observed_has_branch != simulated_has_branch:
        direction_cosine = None
        direction_score = 0.0

    else:
        direction_cosine = None
        direction_score = None

    component_scores = {
        "lumen_dice": lumen_dice,
        "branch_dice": branch_dice,
        "centroid_score": centroid_score,
        "direction_score": direction_score,
    }

    # Omit terms that are uninformative because the relevant class is
    # absent from both observed and simulated images.
    active_weight_sum = sum(
        weights[name]
        for name, component_score in component_scores.items()
        if component_score is not None
    )

    if active_weight_sum <= 0:
        score = 0.0
    else:
        score = sum(
            weights[name] * component_score
            for name, component_score in component_scores.items()
            if component_score is not None
        ) / active_weight_sum

    score = float(np.clip(score, 0.0, 1.0))

    if not return_components:
        return score

    components = {
        "score": score,
        "lumen_dice": lumen_dice,
        "branch_dice": branch_dice,
        "branch_centroid_distance_pixels": branch_centroid_distance,
        "centroid_score": centroid_score,
        "direction_cosine": direction_cosine,
        "direction_score": direction_score,
        "observed_lumen_centroid_xy": observed_lumen_centroid,
        "observed_branch_centroid_xy": observed_branch_centroid,
        "simulated_lumen_centroid_xy": simulated_lumen_centroid,
        "simulated_branch_centroid_xy": simulated_branch_centroid,
        "weights": weights,
    }

    return score, components

def pose_from_position_forward(position, forward, roll=0.0):
    """Build a 4x4 pose from a position, a forward (probe/x) axis, and a roll angle."""
    forward = np.asarray(forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    x_axis = forward

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, x_axis)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(helper, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    c, s = np.cos(roll), np.sin(roll)
    y_rolled = c * y_axis + s * z_axis
    z_rolled = -s * y_axis + c * z_axis

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_rolled
    pose[:3, 2] = z_rolled
    pose[:3, 3] = position
    return pose





def remove_branch_from_mesh_poisson(
    mesh_smp,
    relevant_side_branch_centrelines_pc_points,
    node_pool_distance,
    poisson_depth=10,
    poisson_scale=1.1,
    visualize=True,
):
    """
    Remove vertices surrounding selected branch centerlines and reconstruct
    a closed surface using Poisson surface reconstruction.

    The number of sampled point-cloud points is set equal to the number of
    vertices in the original input mesh.

    Parameters
    ----------
    mesh_smp : o3d.geometry.TriangleMesh
        Original Open3D legacy triangle mesh.

    relevant_side_branch_centrelines_pc_points : array-like
        Centerline points associated with the branches to remove.

    node_pool_distance : float
        Distance around the branch centerlines within which mesh vertices
        are removed.

    poisson_depth : int, default=10
        Octree depth used for Poisson reconstruction. Higher values preserve
        more detail but increase memory and runtime.

    poisson_scale : float, default=1.1
        Ratio between the Poisson reconstruction cube and the point-cloud
        bounding cube.

    visualize : bool, default=True
        Display the original and reconstructed meshes.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Reconstructed Open3D legacy mesh with the selected branches removed.
    """

    if not isinstance(mesh_smp, o3d.geometry.TriangleMesh):
        raise TypeError(
            "mesh_smp must be an open3d.geometry.TriangleMesh"
        )

    original_vertex_count = len(mesh_smp.vertices)
    original_triangle_count = len(mesh_smp.triangles)

    if original_vertex_count == 0 or original_triangle_count == 0:
        raise ValueError("The input mesh is empty.")

    # Use the original mesh vertex count as the Poisson point count.
    number_of_points = original_vertex_count

    print(
        f"Input mesh: {original_vertex_count} vertices, "
        f"{original_triangle_count} triangles"
    )
    print(
        f"Poisson sample count: {number_of_points}"
    )

    # Preserve the original mesh for visualization.
    mesh_old = copy.deepcopy(mesh_smp)
    mesh_old.compute_vertex_normals()
    mesh_old.paint_uniform_color([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # 1. Identify vertices belonging to the branches being removed
    # ------------------------------------------------------------------

    relevant_nodes_sub = get_all_nodes_inside_radius(
        relevant_side_branch_centrelines_pc_points,
        node_pool_distance,
        mesh_smp,
    )

    relevant_nodes_sub = np.asarray(relevant_nodes_sub)

    # Support either:
    #   - a Boolean mask of length number_of_vertices
    #   - an array of vertex indices
    if relevant_nodes_sub.dtype == bool:
        vertex_remove_mask = relevant_nodes_sub.reshape(-1)

        if len(vertex_remove_mask) != original_vertex_count:
            raise ValueError(
                "Boolean removal mask length does not match the "
                "number of input mesh vertices."
            )

        vertex_remove_mask = vertex_remove_mask.copy()

    else:
        vertex_indices = np.asarray(
            relevant_nodes_sub,
            dtype=np.int64,
        ).reshape(-1)

        vertex_indices = np.unique(vertex_indices)

        valid_indices = (
            (vertex_indices >= 0)
            & (vertex_indices < original_vertex_count)
        )
        vertex_indices = vertex_indices[valid_indices]

        vertex_remove_mask = np.zeros(
            original_vertex_count,
            dtype=bool,
        )
        vertex_remove_mask[vertex_indices] = True

    number_removed = int(np.count_nonzero(vertex_remove_mask))

    if number_removed == 0:
        raise ValueError(
            "No vertices were selected for branch removal."
        )

    print(
        f"Removing {number_removed} of "
        f"{original_vertex_count} vertices"
    )

    # ------------------------------------------------------------------
    # 2. Remove the branch vertices
    # ------------------------------------------------------------------

    mesh_cut = copy.deepcopy(mesh_smp)

    # This also removes triangles incident to the deleted vertices.
    mesh_cut.remove_vertices_by_mask(
        vertex_remove_mask.tolist()
    )

    mesh_cut.remove_duplicated_vertices()
    mesh_cut.remove_duplicated_triangles()
    mesh_cut.remove_degenerate_triangles()
    mesh_cut.remove_unreferenced_vertices()

    if len(mesh_cut.vertices) == 0 or len(mesh_cut.triangles) == 0:
        raise ValueError(
            "Branch removal produced an empty mesh. "
            "Reduce node_pool_distance."
        )

    print(
        f"After branch removal: {len(mesh_cut.vertices)} vertices, "
        f"{len(mesh_cut.triangles)} triangles"
    )

    # ------------------------------------------------------------------
    # 3. Prepare consistently oriented normals
    # ------------------------------------------------------------------

    orientable = mesh_cut.orient_triangles()

    if not orientable:
        print(
            "Warning: Open3D could not make all triangle orientations "
            "consistent. Poisson reconstruction may be less reliable."
        )

    mesh_cut.compute_triangle_normals()
    mesh_cut.compute_vertex_normals()
    mesh_cut.normalize_normals()

    # ------------------------------------------------------------------
    # 4. Sample a point cloud
    # ------------------------------------------------------------------

    point_cloud = mesh_cut.sample_points_poisson_disk(
        number_of_points=number_of_points,
        init_factor=5,
        use_triangle_normal=False,
    )

    if len(point_cloud.points) == 0:
        raise ValueError(
            "Point-cloud sampling produced no points."
        )

    if not point_cloud.has_normals():
        # Normally the sampled mesh normals are transferred automatically,
        # but estimate them as a fallback.
        nearest_neighbor_distance = np.mean(
            point_cloud.compute_nearest_neighbor_distance()
        )

        normal_radius = max(
            3.0 * nearest_neighbor_distance,
            np.finfo(float).eps,
        )

        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=50,
            )
        )

        point_cloud.orient_normals_consistent_tangent_plane(
            k=min(50, max(3, len(point_cloud.points) - 1))
        )

    point_cloud.normalize_normals()

    print(
        f"Sampled point cloud: {len(point_cloud.points)} points"
    )

    # ------------------------------------------------------------------
    # 5. Poisson surface reconstruction
    # ------------------------------------------------------------------

    poisson_mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=int(poisson_depth),
            scale=float(poisson_scale),
            linear_fit=False,
            n_threads=-1,
        )
    )

    print(
        f"Raw Poisson mesh: {len(poisson_mesh.vertices)} vertices, "
        f"{len(poisson_mesh.triangles)} triangles"
    )

    # ------------------------------------------------------------------
    # 6. Crop Poisson extrapolation outside the cut mesh bounds
    # ------------------------------------------------------------------

    cut_min = mesh_cut.get_min_bound()
    cut_max = mesh_cut.get_max_bound()

    diagonal_length = np.linalg.norm(cut_max - cut_min)
    padding = 0.01 * diagonal_length

    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=cut_min - padding,
        max_bound=cut_max + padding,
    )

    poisson_mesh = poisson_mesh.crop(crop_box)

    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_unreferenced_vertices()

    if len(poisson_mesh.triangles) == 0:
        raise ValueError(
            "Poisson reconstruction produced no triangles after cropping."
        )

    # ------------------------------------------------------------------
    # 7. Keep the largest connected surface
    # ------------------------------------------------------------------

    (
        triangle_clusters,
        cluster_triangle_counts,
        cluster_areas,
    ) = poisson_mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(
        triangle_clusters,
        dtype=np.int64,
    )
    cluster_areas = np.asarray(
        cluster_areas,
        dtype=np.float64,
    )

    if len(cluster_areas) == 0:
        raise ValueError(
            "Poisson reconstruction produced no connected components."
        )

    print(
        f"Poisson reconstruction produced "
        f"{len(cluster_areas)} connected components"
    )

    largest_cluster_id = int(np.argmax(cluster_areas))

    triangles_to_remove = (
        triangle_clusters != largest_cluster_id
    )

    poisson_mesh.remove_triangles_by_mask(
        triangles_to_remove.tolist()
    )
    poisson_mesh.remove_unreferenced_vertices()

    # ------------------------------------------------------------------
    # 8. Final cleanup and validation
    # ------------------------------------------------------------------

    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_unreferenced_vertices()

    poisson_mesh.orient_triangles()
    poisson_mesh.compute_triangle_normals()
    poisson_mesh.compute_vertex_normals()
    poisson_mesh.normalize_normals()

    # Validate using Trimesh.
    validation_mesh = trimesh.Trimesh(
        vertices=np.asarray(
            poisson_mesh.vertices,
            dtype=np.float64,
        ).copy(),
        faces=np.asarray(
            poisson_mesh.triangles,
            dtype=np.int64,
        ).copy(),
        process=True,
    )

    validation_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(validation_mesh)
    trimesh.repair.fix_normals(validation_mesh)

    print(
        f"Final mesh: {len(validation_mesh.vertices)} vertices, "
        f"{len(validation_mesh.faces)} faces"
    )
    print(
        f"Watertight: {validation_mesh.is_watertight}"
    )
    print(
        f"Winding consistent: "
        f"{validation_mesh.is_winding_consistent}"
    )

    # Convert the validated mesh back to Open3D so the function returns
    # the expected legacy Open3D mesh type.
    mesh_new = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.asarray(
                validation_mesh.vertices,
                dtype=np.float64,
            ).copy()
        ),
        triangles=o3d.utility.Vector3iVector(
            np.asarray(
                validation_mesh.faces,
                dtype=np.int32,
            ).copy()
        ),
    )

    mesh_new.compute_triangle_normals()
    mesh_new.compute_vertex_normals()
    mesh_new.paint_uniform_color([1.0, 0.0, 0.0])

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh_new],
            window_name="Poisson branch removal",
        )

    return mesh_new

def get_all_nodes_inside_radius(centroids, radius, mesh):
    """
    Finds all nodes on a mesh that are within a specified radius of given centroids.

    Parameters:
        centroids (numpy.ndarray): Array of shape (n, 3) containing the centroid coordinates.
        radius (float): Radius within which to search for nodes.
        mesh (o3d.geometry.TriangleMesh): The registered mesh to search nodes in.

    Returns:
        dict: A dictionary where keys are centroid indices, and values are lists of mesh node indices within the radius.
    """
    vertices = np.asarray(mesh.vertices)

    # Build a KDTree for the mesh vertices
    kdtree = o3d.geometry.KDTreeFlann(mesh)

    # Dictionary to store the result
    result = {}

    for i, centroid in enumerate(centroids):
        # Query the KDTree for all points within the radius
        [_, idxs, _] = kdtree.search_radius_vector_3d(centroid, radius)

        # Store the indices in the result dictionary
        result[i] = idxs  # idxs is a list of indices of vertices within the radius

    combined_list = []
    for key, int_vector in result.items():
        combined_list.extend(list(int_vector))  # Convert IntVector to list and extend
    result = np.array(combined_list)

    return result

def fix_mesh_poisson(mesh_cut, poisson_depth=10,
    poisson_scale=1.1):

    original_vertex_count = len(mesh_cut.vertices)
    original_triangle_count = len(mesh_cut.triangles)

    if original_vertex_count == 0 or original_triangle_count == 0:
        raise ValueError("The input mesh is empty.")

    # Use the original mesh vertex count as the Poisson point count.
    number_of_points = original_vertex_count

    point_cloud = mesh_cut.sample_points_poisson_disk(
        number_of_points=number_of_points,
        init_factor=5,
        use_triangle_normal=False,
    )

  

    if not point_cloud.has_normals():
        # Normally the sampled mesh normals are transferred automatically,
        # but estimate them as a fallback.
        nearest_neighbor_distance = np.mean(
            point_cloud.compute_nearest_neighbor_distance()
        )

        normal_radius = max(
            3.0 * nearest_neighbor_distance,
            np.finfo(float).eps,
        )

        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=50,
            )
        )

        point_cloud.orient_normals_consistent_tangent_plane(
            k=min(50, max(3, len(point_cloud.points) - 1))
        )

    point_cloud.normalize_normals()

    print(
        f"Sampled point cloud: {len(point_cloud.points)} points"
    )

    # ------------------------------------------------------------------
    # 5. Poisson surface reconstruction
    # ------------------------------------------------------------------

    poisson_mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=int(poisson_depth),
            scale=float(poisson_scale),
            linear_fit=False,
            n_threads=-1,
        )
    )

    return poisson_mesh

def erode_connected_masks(
    mask_1,
    mask_2,
    erosion_pixels=3,
):
   

    mask_1_binary = np.asarray(mask_1) > 0
    mask_2_binary = np.asarray(mask_2) > 0

    if mask_1_binary.shape != mask_2_binary.shape:
        raise ValueError(
            "mask_1 and mask_2 must have the same shape. "
            f"Received {mask_1_binary.shape} and {mask_2_binary.shape}."
        )

    if mask_1_binary.ndim != 2:
        raise ValueError("mask_1 and mask_2 must be 2-D arrays.")

    if erosion_pixels < 0:
        raise ValueError("erosion_pixels must be nonnegative.")

    # Form one connected semantic object.
    combined_mask = mask_1_binary | mask_2_binary

    if erosion_pixels == 0:
        eroded_combined = combined_mask
    else:
        kernel_size = 2 * erosion_pixels + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        # Erosion is determined exclusively from the exterior boundary
        # of the combined lumen-plus-branch object.
        eroded_combined = cv2.erode(
            combined_mask.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0

    # Apply the same combined erosion result to each original class.
    # The internal lumen/branch boundary is not independently eroded.
    eroded_mask_1 = mask_1_binary & eroded_combined
    eroded_mask_2 = mask_2_binary & eroded_combined


    mask_1_eroded= cv2.erode(
            mask_1_binary.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0


    mask_1_ring = mask_1_binary & ~mask_1_eroded
    mask_1_bit = mask_1_ring & eroded_combined
    # cv2.imshow("mask_1_ring", mask_1_ring.astype(np.uint8) * 255)
    # cv2.imshow("mask_1_bit", mask_1_bit.astype(np.uint8) * 255)
    eroded_mask_2 = eroded_mask_2 | mask_1_bit
    eroded_mask_1 = eroded_mask_1 & ~mask_1_bit
    

    


    return (
        eroded_mask_1.astype(np.uint8),
        eroded_mask_2.astype(np.uint8),
        
    )
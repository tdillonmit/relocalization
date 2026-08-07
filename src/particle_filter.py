#!/usr/bin/env python3.9
"""Particle-filter localization of the IVUS catheter's extrinsic pose.

Implements the Unscented-Particle-Filter-style resampling/covariance scheme
from Koolwal, Barbagli, Carlson and Liang, "An Ultrasound-Based Localization
Algorithm for Catheter Ablation Guidance in the Left Atrium" (IJRR, 2010),
Section 4.4 ("RBSE Implementation"), adapted to this project:

  * There is no robot and no inverse kinematics. The paper's control input
    (robotic guide kinematics) is replaced entirely by the *relative* EM
    sensor displacement between consecutive frames (Section 4.4, "control
    input u"): ``delta = inv(T_prev_measured) @ T_curr_measured``, composed
    onto each particle's own pose estimate and corrupted by EM sensor noise.

  * The paper's per-particle Unscented Kalman Filter (each particle carries
    a full Gaussian propagated through sigma points) is simplified to a
    standard point-particle SIR filter: each particle is scored once
    (not 2n+1 times) against the simulated IVUS image, using
    ``ultrasound_mesh_simulator.get_weighted_image_correlation_score`` in
    place of the paper's NMI correlation metric (Section 4.4.1). This is
    ~13x cheaper per particle per frame, which matters because scoring
    requires a full ``VesselUltrasoundSimulator`` forward pass per particle
    (by default ``simulate_segmentation_fast``, a ~2.7x-faster
    trimesh-plane-section-based equivalent of ``simulate_segmentation``
    validated to ~0.99/~0.98 mean Dice agreement with it -- see
    ``score_particles``'s ``use_fast_simulation``). Each particle still
    carries a diagonal covariance
    ``Sigma_i`` (grown by process noise every ``predict()``, reset on
    injection) so that Section 4.4.2's resampling/weighting and Section
    4.4.3's weighted mean/covariance formulas apply exactly as written.

  * Particle positions are sampled within ``tol = vessel_diameter / 2`` of a
    supplied vessel centerline point cloud (rather than a raw bounding box
    the way the paper's original x_range, Eq. 29, samples over the whole
    left atrium), so no particle is proposed somewhere physically outside
    the vessel. Orientations are biased toward the local centerline
    tangent (the probe's local x axis is its forward/roll axis -- see
    ``ultrasound_mesh_simulator``'s module docstring), since a catheter's
    axis roughly follows the vessel it is sitting in.

State convention
-----------------
A particle's "value" is a 4x4 SE(3) pose ``T`` in the same coordinate frame
as the registered CT mesh (``T[:3, 3]`` = probe tip position, ``T[:3, 0]`` =
probe forward/roll axis). This *is* the extrinsic matrix
(``TW_EM @ TEM_C`` in run_relocalization.py's terms) -- callers pass whatever
consecutive "measured" poses they already compute that way as
``predict()``'s ``T_prev_measured`` / ``T_curr_measured``, and the filter's
weighted-mean estimate (Eq. 42) is the localized extrinsic.

Equation references below (Eq. NN) are to the 2010 IJRR paper.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from reconstruction_helpers_runtime import get_transform_inverse
from ultrasound_mesh_simulator import (
    erode_connected_masks,
    get_weighted_image_correlation_score,
    pose_from_position_forward,
)

__all__ = [
    "ParticleFilterConfig",
    "ParticleFilterEstimate",
    "Centerline",
    "ParticleFilter",
    "ParticleCloudVisualizer",
]

# Cylinder-template alignment: o3d.geometry.TriangleMesh.create_cylinder
# builds a cylinder whose axis is the local +Z axis. Particle poses use the
# local +X axis as the probe forward axis, so every template vertex is
# rotated by this fixed 90 degree rotation (Z -> X) before being placed at a
# particle's pose.
_CYLINDER_Z_TO_X = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
)


# ---------------------------------------------------------------------------
# Small SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------
def _random_unit_vectors(rng, n):
    vectors = rng.normal(size=(n, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _sample_tangent_biased_forward(rng, tangent, spread_rad, prefer_sign_ref=None):
    """Sample a unit forward vector within a cone of half-angle ``spread_rad``
    around ``tangent`` (or its negation).

    ``tangent`` is sign-ambiguous (it comes from PCA over nearby centerline
    points). When ``prefer_sign_ref`` is given, the sign that best agrees
    with it is kept (keeps a re-seeded particle's forward axis continuous
    with the previous best estimate); otherwise the sign is chosen randomly.
    """
    tangent = np.asarray(tangent, dtype=float)
    norm = np.linalg.norm(tangent)
    if norm < 1e-9:
        return _random_unit_vectors(rng, 1)[0]
    tangent = tangent / norm

    if prefer_sign_ref is not None:
        if np.dot(tangent, prefer_sign_ref) < 0:
            tangent = -tangent
    elif rng.random() < 0.5:
        tangent = -tangent

    if spread_rad >= np.pi:
        return _random_unit_vectors(rng, 1)[0]

    cos_theta = rng.uniform(np.cos(spread_rad), 1.0)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))
    phi = rng.uniform(0.0, 2.0 * np.pi)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, tangent)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, tangent)
    u /= np.linalg.norm(u)
    v = np.cross(tangent, u)

    forward = cos_theta * tangent + sin_theta * (np.cos(phi) * u + np.sin(phi) * v)
    return forward / np.linalg.norm(forward)


def _weighted_chordal_rotation_mean(rotations, weights):
    """Weighted chordal-L2 mean rotation (projects the weighted average of
    rotation matrices back onto SO(3) via SVD). Standard, well-conditioned
    substitute for a weighted quaternion/Euler-angle average."""
    weights = weights / weights.sum()
    m = np.einsum("i,ijk->jk", weights, rotations)
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    return u @ correction @ vt


# ---------------------------------------------------------------------------
# Centerline-constrained sampling
# ---------------------------------------------------------------------------
class Centerline:
    """Vessel centerline point cloud used to keep sampled particle positions
    physically inside the vessel, per the project's own
    ``centerline_pc.ply`` convention (loaded from a dataset directory, the
    same way ``VesselUltrasoundSimulator`` loads ``no_branch_mesh.ply`` /
    ``side_branch_centrelines.ply`` from its ``mesh_path``)."""

    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            raise ValueError("Centerline point cloud is empty.")
        self.points = points
        self.tree = cKDTree(points)

    @classmethod
    def from_ply(cls, path):
        pc = o3d.io.read_point_cloud(str(path))
        if pc.is_empty():
            raise FileNotFoundError(
                f"No points loaded from centerline point cloud: {path}"
            )
        return cls(np.asarray(pc.points))

    def distance_to(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dist, _ = self.tree.query(points, k=1)
        return dist

    def tangent_at(self, points, k=8):
        """Local tangent direction(s) at ``points`` via PCA over the ``k``
        nearest centerline points. Sign is arbitrary (resolved by callers)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        k = min(k, len(self.points))
        _, idx = self.tree.query(points, k=k)
        idx = np.atleast_2d(idx)
        tangents = np.empty((len(points), 3))
        for row in range(len(points)):
            neighbors = self.points[idx[row]]
            centered = neighbors - neighbors.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            tangents[row] = vt[0]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return tangents / norms

    def sample_uniform(self, rng, n, tol):
        """``n`` positions drawn uniformly within radius ``tol`` of a
        uniformly-random centerline point (Section 4.4's Eq. 29 analog,
        constrained to the vessel instead of a bounding box)."""
        idx = rng.integers(0, len(self.points), size=n)
        base = self.points[idx]
        directions = _random_unit_vectors(rng, n)
        radii = tol * np.cbrt(rng.uniform(0.0, 1.0, size=n))
        return base + directions * radii[:, None]

    def sample_near_seed(
        self, rng, seed_position, pos_std, n, tol, is_inside_fn=None,
        pool_factor=20, min_pool=256,
    ):
        """``n`` positions Gaussian-scattered around ``seed_position``
        (Eq. 40/41's re-seeding), rejected against the centerline tolerance
        and, optionally, an ``is_inside_fn`` SDF/occupancy check.

        Always returns exactly ``n`` positions: if not enough candidates
        pass rejection (e.g. the seed sits far outside the vessel, as when
        the tank has been physically moved), the remainder falls back to
        jittering the centerline point nearest the seed -- i.e. still
        "close to the current best estimate" in the sense of being the
        nearest feasible vessel location to it.
        """
        seed_position = np.asarray(seed_position, dtype=float)
        pos_std = np.asarray(pos_std, dtype=float)
        pool_size = max(n * pool_factor, min_pool)

        candidates = seed_position + rng.normal(scale=pos_std, size=(pool_size, 3))
        dist = self.distance_to(candidates)
        valid = dist <= tol

        if is_inside_fn is not None and valid.any():
            valid_idx = np.where(valid)[0]
            inside = np.asarray(is_inside_fn(candidates[valid_idx]))
            valid[valid_idx] = inside

        accepted = candidates[valid]

        if len(accepted) >= n:
            return accepted[:n]

        deficit = n - len(accepted)
        nearest_dist, nearest_idx = self.tree.query(seed_position[None, :], k=1)
        anchor = self.points[nearest_idx[0]]
        offsets = rng.normal(scale=tol * 0.3, size=(deficit, 3))
        offset_norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        too_far = offset_norms[:, 0] > tol
        if too_far.any():
            offsets[too_far] = offsets[too_far] / offset_norms[too_far] * tol
        fallback = anchor + offsets

        return np.vstack([accepted, fallback]) if len(accepted) else fallback


# ---------------------------------------------------------------------------
# Configuration / result types
# ---------------------------------------------------------------------------
@dataclass
class ParticleFilterConfig:
    num_particles: int = 300

    # D in "sample within D/2 of the centerline"; also sets the initial
    # per-particle position variance (Sigma_init's translation block).
    vessel_diameter_m: float = 0.02

    # EM sensor accuracy (process/control noise), applied per predict() step.
    position_noise_std_m: float = 0.0005
    rotation_noise_std_rad: float = np.deg2rad(0.1)

    # Reject sampled positions that fall outside the lumen SDF as well as
    # outside the centerline tolerance. Cheap (batched raycasting), so left
    # on by default; set False if it becomes a bottleneck.
    check_sdf_inside: bool = True

    # "tangent": bias sampled orientations toward the local centerline
    # direction (the probe's local x axis). "uniform": fully random
    # orientation, matching the paper's literal global-localization prior.
    orientation_prior: str = "tangent"
    tangent_spread_rad: float = np.deg2rad(30.0)

    # Sharpens the [0, 1] correlation score into a resampling fitness
    # (fitness = max(score, floor) ** temperature) so that near-ties in
    # score don't produce a near-uniform (uninformative) resampling
    # distribution.
    score_temperature: float = 6.0
    min_fitness_floor: float = 1e-6

    # Gutmann & Fox (2002) augmented-MCL confidence tracking (Eq. 34/35).
    kappa_slow_alpha: float = 0.02
    kappa_fast_alpha: float = 0.3
    min_injection_spread_scale: float = 0.05


@dataclass
class ParticleFilterEstimate:
    pose: np.ndarray  # (4, 4) weighted-mean extrinsic (Eq. 42)
    covariance_diag: np.ndarray  # (6,) weighted sum of particle Sigma_i (Eq. 43)
    covariance_diag_with_spread: np.ndarray  # + inter-particle translation spread
    confidence: Optional[float]  # f_confidence(S_k), Eq. 34
    num_particles: int


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------
class ParticleFilter:
    """SIR particle filter over SE(3) catheter poses, resampled/re-seeded
    per Koolwal Section 4.4.2 and estimated per Section 4.4.3."""

    def __init__(self, centerline, config: ParticleFilterConfig = None, seed=None):
        self.centerline = (
            centerline if isinstance(centerline, Centerline) else Centerline.from_ply(centerline)
        )
        self.config = config or ParticleFilterConfig()
        self.rng = np.random.default_rng(seed)

        self.tol = self.config.vessel_diameter_m / 2.0
        self.init_cov_diag = np.array(
            [self.tol ** 2] * 3 + [self.config.tangent_spread_rad ** 2] * 3
        )

        self.poses: Optional[np.ndarray] = None       # (N, 4, 4)
        self.cov_diag: Optional[np.ndarray] = None     # (N, 6)
        self.weights: Optional[np.ndarray] = None      # (N,)
        self.scores: Optional[np.ndarray] = None        # (N,) or None if stale

        self._kappa_slow = None
        self._kappa_fast = None
        self.last_confidence = None
        self.last_p_new = None
        self.last_n_old = None
        self.last_n_new = None

    # ------------------------------------------------------------------
    # Particle generation
    # ------------------------------------------------------------------
    def _sample_orientations(self, positions, prefer_sign_ref=None):
        n = len(positions)
        if self.config.orientation_prior == "uniform":
            forwards = _random_unit_vectors(self.rng, n)
        else:
            tangents = self.centerline.tangent_at(positions)
            forwards = np.empty((n, 3))
            for i in range(n):
                forwards[i] = _sample_tangent_biased_forward(
                    self.rng, tangents[i], self.config.tangent_spread_rad,
                    prefer_sign_ref=prefer_sign_ref,
                )
        rolls = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        return np.stack(
            [pose_from_position_forward(positions[i], forwards[i], rolls[i]) for i in range(n)]
        )

    def initialize(self, seed_pose=None, sim=None, num_particles=None):
        """(Re-)populate the particle set.

        ``seed_pose=None``: global initialization, Eq. 29's analog --
        particles scattered uniformly within ``tol`` of the whole
        centerline.

        ``seed_pose``: a full warm re-seed around a known/hinted pose
        (e.g. after externally recognizing the estimate has drifted, such
        as the phantom/tank having been physically moved) -- particles are
        scattered around ``seed_pose`` but still constrained to be within
        ``tol`` of the centerline (and, if ``sim`` is given, inside the
        lumen SDF), so the new particles favor the region that is both
        "close to the current best estimate" and physically plausible.
        """
        n = num_particles or self.config.num_particles
        is_inside_fn = sim.is_inside if (sim is not None and self.config.check_sdf_inside) else None

        if seed_pose is None:
            positions = self.centerline.sample_uniform(self.rng, n, self.tol)
            poses = self._sample_orientations(positions, prefer_sign_ref=None)
        else:
            seed_pose = np.asarray(seed_pose, dtype=float)
            positions = self.centerline.sample_near_seed(
                self.rng, seed_pose[:3, 3], np.sqrt(self.init_cov_diag[:3]),
                n, self.tol, is_inside_fn=is_inside_fn,
            )
            poses = self._sample_orientations(positions, prefer_sign_ref=seed_pose[:3, 0])

        self.poses = poses
        self.cov_diag = np.tile(self.init_cov_diag, (n, 1))
        self.weights = np.full(n, 1.0 / n)
        self.scores = None
        self._kappa_slow = None
        self._kappa_fast = None
        self.last_confidence = None
        self.last_p_new = None
        self.last_n_old = None
        self.last_n_new = None

    # ------------------------------------------------------------------
    # Prediction: control input u = relative measured EM displacement
    # ------------------------------------------------------------------
    def predict(self, T_prev_measured, T_curr_measured):
        """Compose each particle with the relative EM displacement between
        two consecutive measured poses (already including any fixed
        EM-sensor-to-catheter calibration the caller applies -- this method
        only needs the two poses to be expressed in the same convention as
        the particle poses), corrupted per-particle by independent EM
        sensor noise. This is the "control input u" the paper would
        otherwise obtain from robotic kinematics (Section 4, replaced here
        as no robot is used)."""
        if self.poses is None:
            raise RuntimeError("initialize() must be called before predict().")

        T_prev_measured = np.asarray(T_prev_measured, dtype=float)
        T_curr_measured = np.asarray(T_curr_measured, dtype=float)
        delta = get_transform_inverse(T_prev_measured) @ T_curr_measured

        n = len(self.poses)
        rotvecs = self.rng.normal(scale=self.config.rotation_noise_std_rad, size=(n, 3))
        r_noise = Rotation.from_rotvec(rotvecs).as_matrix()
        t_noise = self.rng.normal(scale=self.config.position_noise_std_m, size=(n, 3))

        noise_transform = np.tile(np.eye(4), (n, 1, 1))
        noise_transform[:, :3, :3] = r_noise
        noise_transform[:, :3, 3] = t_noise

        noisy_delta = np.matmul(delta[None, :, :], noise_transform)
        self.poses = np.matmul(self.poses, noisy_delta)

        process_noise = np.array(
            [self.config.position_noise_std_m ** 2] * 3
            + [self.config.rotation_noise_std_rad ** 2] * 3
        )
        self.cov_diag = self.cov_diag + process_noise
        self.scores = None

    # ------------------------------------------------------------------
    # Measurement: correlation score against the observed IVUS masks
    # ------------------------------------------------------------------
    def score_particles(
        self, sim, observed_mask_1, observed_mask_2, score_kwargs=None,
        align_to_real_image_convention=True, erosion_pixels=6, use_fast_simulation=True,
    ):
        """Score every particle's simulated (mask_1, mask_2) against the
        observed masks with ``get_weighted_image_correlation_score``
        (Section 4.4.1's role, replacing the paper's NMI metric).

        ``use_fast_simulation``: use ``VesselUltrasoundSimulator.
        simulate_segmentation_fast`` (trimesh-plane-section based, ~2.7x
        faster, validated to ~0.99/~0.98 mean Dice agreement with the
        original) rather than ``simulate_segmentation``. This only affects
        per-particle scoring here -- run_relocalization.py and everything
        else that calls ``simulate_segmentation`` directly is untouched.
        Set False to fall back to the original method (e.g. to re-validate
        agreement, or if a future mesh triggers the fast path's rare
        sub-min-component-area sliver disagreement in a way that matters).

        ``align_to_real_image_convention``: when the observed masks come
        from the real DeepLumen segmentation network (rather than from the
        simulator itself, as in a synthetic self-test), run_relocalization.py
        vertically flips the simulator's output and applies
        ``erode_connected_masks`` before comparing it against DeepLumen's
        output -- the two pipelines don't share a row-order/boundary
        convention otherwise. Leave this on (the default) whenever
        ``observed_mask_1/2`` are real segmentations; set it False when both
        sides come from ``VesselUltrasoundSimulator`` (they already share a
        convention, and flipping/eroding would only discard information).
        """
        if self.poses is None:
            raise RuntimeError("initialize() must be called before score_particles().")

        score_kwargs = score_kwargs or {}
        simulate_fn = sim.simulate_segmentation_fast if use_fast_simulation else sim.simulate_segmentation
        scores = np.empty(len(self.poses))
        for i, pose in enumerate(self.poses):
            sim_mask_1, sim_mask_2 = simulate_fn(pose)
            if align_to_real_image_convention:
                sim_mask_1 = cv2.flip(sim_mask_1, 0)
                sim_mask_2 = cv2.flip(sim_mask_2, 0)
                sim_mask_1, sim_mask_2 = erode_connected_masks(
                    sim_mask_1, sim_mask_2, erosion_pixels
                )
            scores[i] = get_weighted_image_correlation_score(
                observed_mask_1, observed_mask_2, sim_mask_1, sim_mask_2, **score_kwargs
            )
        self.scores = scores
        return scores

    # ------------------------------------------------------------------
    # Resampling + augmented (Gutmann & Fox) re-seeding -- Section 4.4.2
    # ------------------------------------------------------------------
    @staticmethod
    def _sus_resample(fitness, n_draws, rng):
        """Stochastic universal sampling (Baker 1987), as used by the
        paper's resampling step."""
        fitness = np.asarray(fitness, dtype=float)
        total = fitness.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.full(len(fitness), 1.0 / len(fitness))
        else:
            probs = fitness / total
        cumulative = np.cumsum(probs)
        start = rng.uniform(0.0, 1.0 / n_draws)
        pointers = start + np.arange(n_draws) / n_draws
        indices = np.clip(np.searchsorted(cumulative, pointers), 0, len(fitness) - 1)
        return np.bincount(indices, minlength=len(fitness))

    def _weighted_estimate(self, poses, weights, cov_diag):
        weights = weights / weights.sum()
        mean_t = np.sum(weights[:, None] * poses[:, :3, 3], axis=0)
        mean_r = _weighted_chordal_rotation_mean(poses[:, :3, :3], weights)
        mean_pose = np.eye(4)
        mean_pose[:3, :3] = mean_r
        mean_pose[:3, 3] = mean_t
        mean_cov = np.sum(weights[:, None] * cov_diag, axis=0)
        return mean_pose, mean_cov

    def resample_and_augment(self, sim=None):
        """Resample by correlation score, discard/collapse duplicates into
        weighted unique particles (Eq. 33), then track the fast/slow
        confidence averages and inject fresh, centerline-constrained
        particles around the current estimate when confidence has dropped
        (Eq. 34-41). Returns the post-augmentation state estimate."""
        if self.scores is None:
            raise RuntimeError("score_particles() must be called before resample_and_augment().")

        cfg = self.config
        fitness = np.clip(self.scores, cfg.min_fitness_floor, 1.0) ** cfg.score_temperature

        counts = self._sus_resample(fitness, cfg.num_particles, self.rng)
        unique_idx = np.nonzero(counts)[0]

        old_weights = counts[unique_idx] / float(cfg.num_particles)
        old_poses = self.poses[unique_idx]
        old_cov = self.cov_diag[unique_idx]
        old_scores = self.scores[unique_idx]
        n_old = len(unique_idx)

        # Eq. 34
        confidence = float(np.sum(old_weights * old_scores))
        if self._kappa_slow is None:
            self._kappa_slow = confidence
            self._kappa_fast = confidence
        else:
            self._kappa_slow += cfg.kappa_slow_alpha * (confidence - self._kappa_slow)
            self._kappa_fast += cfg.kappa_fast_alpha * (confidence - self._kappa_fast)

        # Eq. 35
        p_new = float(np.clip(1.0 - self._kappa_fast / max(self._kappa_slow, 1e-9), 0.0, 1.0))

        # Eq. 36 (solved for N_new given the target ratio)
        if p_new <= 0.0 or n_old == 0:
            n_new = 0
        elif p_new >= 1.0:
            n_new = cfg.num_particles
        else:
            n_new = int(round(p_new / (1.0 - p_new) * n_old))
        n_new = min(n_new, cfg.num_particles)  # keep the per-step budget bounded

        # Eq. 40/41's seed: the (pre-augmentation) resampled set's own
        # weighted estimate.
        seed_pose, _ = self._weighted_estimate(old_poses, old_weights, old_cov)

        if n_new > 0:
            spread_scale = float(
                np.clip(1.0 - self._kappa_fast, cfg.min_injection_spread_scale, 1.0)
            )
            new_pos_std = np.sqrt(self.init_cov_diag[:3]) * spread_scale
            is_inside_fn = sim.is_inside if (sim is not None and cfg.check_sdf_inside) else None
            new_positions = self.centerline.sample_near_seed(
                self.rng, seed_pose[:3, 3], new_pos_std, n_new, self.tol,
                is_inside_fn=is_inside_fn,
            )
            new_poses = self._sample_orientations(new_positions, prefer_sign_ref=seed_pose[:3, 0])
            new_cov = np.tile(self.init_cov_diag * spread_scale, (n_new, 1))
            new_scores = np.full(n_new, np.nan)
        else:
            new_poses = np.empty((0, 4, 4))
            new_cov = np.empty((0, 6))
            new_scores = np.empty(0)

        denom = float(n_old + n_new)
        if denom == 0:
            raise RuntimeError(
                "Particle filter lost all particles (every score was rejected); "
                "increase num_particles, raise score_temperature's floor, or "
                "loosen the centerline tolerance."
            )

        # Eq. 37-39
        merged_old_weights = old_weights * (n_old / denom)
        merged_new_weights = (
            np.full(n_new, 1.0 / denom) if n_new > 0 else np.empty(0)
        )

        self.poses = np.vstack([old_poses, new_poses]) if n_new > 0 else old_poses
        self.cov_diag = np.vstack([old_cov, new_cov]) if n_new > 0 else old_cov
        self.weights = np.concatenate([merged_old_weights, merged_new_weights])
        self.scores = np.concatenate([old_scores, new_scores])

        self.last_confidence = confidence
        self.last_p_new = p_new
        self.last_n_old = n_old
        self.last_n_new = n_new

        return self.estimate()

    # ------------------------------------------------------------------
    # State estimate expected value + covariance -- Section 4.4.3
    # ------------------------------------------------------------------
    def estimate(self) -> ParticleFilterEstimate:
        if self.poses is None or len(self.poses) == 0:
            raise RuntimeError("No particles to estimate from; call initialize() first.")

        mean_pose, cov_diag = self._weighted_estimate(self.poses, self.weights, self.cov_diag)

        # Eq. 43 sums only each particle's *own* covariance. In this
        # simplified SIR filter Sigma_i is just accumulated process noise,
        # so on its own it can understate the true spread whenever most of
        # the uncertainty lives in *disagreement between particles* rather
        # than within any one particle. covariance_diag is the paper-exact
        # Eq. 43 value; covariance_diag_with_spread additionally folds in
        # the weighted spread of particle positions around the mean, for
        # practical use (e.g. sizing a visualization or a gating region).
        diffs = self.poses[:, :3, 3] - mean_pose[:3, 3]
        spread_pos = np.sum(self.weights[:, None] * diffs ** 2, axis=0)
        cov_with_spread = cov_diag.copy()
        cov_with_spread[:3] += spread_pos

        return ParticleFilterEstimate(
            pose=mean_pose,
            covariance_diag=cov_diag,
            covariance_diag_with_spread=cov_with_spread,
            confidence=self.last_confidence,
            num_particles=len(self.poses),
        )

    def step(
        self, T_prev_measured, T_curr_measured, sim, observed_mask_1, observed_mask_2,
        score_kwargs=None, align_to_real_image_convention=True, use_fast_simulation=True,
    ):
        """Convenience one-shot predict -> score -> resample/augment."""
        self.predict(T_prev_measured, T_curr_measured)
        self.score_particles(
            sim, observed_mask_1, observed_mask_2, score_kwargs=score_kwargs,
            align_to_real_image_convention=align_to_real_image_convention,
            use_fast_simulation=use_fast_simulation,
        )
        return self.resample_and_augment(sim=sim)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def build_particle_cylinders_mesh(
        self, radius=0.0004, height=0.0025, resolution=3,
        weight_color_low=(0.15, 0.35, 0.9), weight_color_high=(1.0, 0.15, 0.1),
    ):
        """A single merged ``o3d.geometry.TriangleMesh`` of small cylinders,
        one per particle, aligned to each particle's probe (local x) axis
        and colored from ``weight_color_low`` (least likely) to
        ``weight_color_high`` (most likely)."""
        if self.poses is None or len(self.poses) == 0:
            return o3d.geometry.TriangleMesh()

        template = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height, resolution=resolution
        )
        template_verts = np.asarray(template.vertices)
        template_tris = np.asarray(template.triangles)
        n_verts = len(template_verts)
        n_tris = len(template_tris)
        n = len(self.poses)

        weights = self.weights
        span = weights.max() - weights.min()
        w_norm = (weights - weights.min()) / span if span > 0 else np.zeros_like(weights)
        low = np.asarray(weight_color_low)
        high = np.asarray(weight_color_high)

        all_verts = np.empty((n * n_verts, 3))
        all_tris = np.empty((n * n_tris, 3), dtype=np.int64)
        all_colors = np.empty((n * n_verts, 3))

        for i in range(n):
            rot = self.poses[i, :3, :3] @ _CYLINDER_Z_TO_X
            translation = self.poses[i, :3, 3]
            all_verts[i * n_verts:(i + 1) * n_verts] = template_verts @ rot.T + translation
            all_tris[i * n_tris:(i + 1) * n_tris] = template_tris + i * n_verts
            all_colors[i * n_verts:(i + 1) * n_verts] = low + (high - low) * w_norm[i]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
        mesh.compute_vertex_normals()
        return mesh


class ParticleCloudVisualizer:
    """Manages an Open3D merged-cylinder particle cloud plus a distinct
    estimate cylinder inside a caller-owned ``o3d.visualization.Visualizer``,
    following the same transform-delta update pattern
    ``PointCloudUpdater.tracker`` uses in run_relocalization.py."""

    def __init__(
        self, vis, cylinder_radius=0.0006, cylinder_height=0.00375,
        cylinder_resolution=8, estimate_color=(0.05, 0.9, 0.05),
    ):
        self.vis = vis
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.cylinder_resolution = cylinder_resolution

        self.particle_mesh = o3d.geometry.TriangleMesh()
        self.estimate_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cylinder_radius * 2.2, height=cylinder_height * 1.6,
            resolution=cylinder_resolution,
        )
        self.estimate_mesh.transform(
            np.block([[_CYLINDER_Z_TO_X, np.zeros((3, 1))], [0, 0, 0, 1]])
        )
        self.estimate_mesh.compute_vertex_normals()
        self.estimate_mesh.paint_uniform_color(estimate_color)
        self._estimate_prev_pose = np.eye(4)

        # vis.add_geometry(self.particle_mesh)
        vis.add_geometry(self.estimate_mesh)

    def update(self, pf: ParticleFilter, estimate_pose=None):
        new_mesh = pf.build_particle_cylinders_mesh(
            radius=self.cylinder_radius, height=self.cylinder_height,
            resolution=self.cylinder_resolution,
        )
        self.particle_mesh.vertices = new_mesh.vertices
        self.particle_mesh.triangles = new_mesh.triangles
        self.particle_mesh.vertex_colors = new_mesh.vertex_colors
        self.particle_mesh.compute_vertex_normals()
        self.vis.update_geometry(self.particle_mesh)

        if estimate_pose is not None:
            estimate_pose = np.asarray(estimate_pose, dtype=float)
            self.estimate_mesh.transform(get_transform_inverse(self._estimate_prev_pose))
            self.estimate_mesh.transform(estimate_pose)
            self._estimate_prev_pose = estimate_pose.copy()
            self.vis.update_geometry(self.estimate_mesh)

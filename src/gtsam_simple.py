import numpy as np
import open3d as o3d
import open3d.core as o3c
import gtsam
from gtsam import Pose3, Rot3, Point3, noiseModel

# ============================================================
# Ellipse curve on cylinder at an angle
# ============================================================
def make_tilted_ellipse_on_cylinder(radius=1.0, slope=0.4, n=200):
    ts = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radius * np.cos(ts)
    y = radius * np.sin(ts)
    z = slope * radius * np.sin(ts)
    return np.stack([x, y, z], axis=1)


# ============================================================
# Open3D SDF helper (point -> float)
# ============================================================
def sdf_query(scene, pw):
    p = np.asarray(pw, dtype=np.float32)
    dist = scene.compute_signed_distance(o3c.Tensor(p[None], dtype=o3c.float32))
    return float(dist.numpy()[0])



# ============================================================
# Numerical gradient of SDF wrt world-point
# ============================================================
def sdf_grad(scene, p, eps=1e-4):
    g = np.zeros(3)
    for i in range(3):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        g[i] = (sdf_query(scene, pp) - sdf_query(scene, pm)) / (2*eps)
    return g


# ============================================================
# CustomFactor — SDF boundary alignment
# residual = signed distance (not hinge)
# ============================================================
def make_sdf_boundary_factor(scan_point, scene):
    scan_point = np.asarray(scan_point, float)

    def f(this, values, jacobians):
        pose = values.atPose3(this.keys()[0])
        T = pose.matrix()
        R = T[:3,:3]
        t = T[:3,3]

        pw = R @ scan_point + t
        d = sdf_query(scene, pw)

        res = np.array([d], float)

        if jacobians is not None:
            # numerical grad wrt point
            g = sdf_grad(scene, pw)

            J = np.zeros((1,6))
            # d/dt = grad
            J[:,0:3] = g

            # d/dθ = grad^T * (-R * skew(p))
            # skew:
            x,y,z = scan_point
            S = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            J[:,3:6] = g @ (-R @ S)
            jacobians[0] = J

        return res

    return f


# ============================================================
# Build cylinder & SDF scene
# ============================================================
cyl_height = 2.5
radius = 1.0

mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, cyl_height)
mesh.compute_vertex_normals()

tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangle_mesh(tmesh)
scene.add_triangles(tmesh)


# ============================================================
# Generate target ellipse (in true pose = identity)
# ============================================================
p_scan = make_tilted_ellipse_on_cylinder(radius=radius, slope=0.4, n=150)


# ============================================================
# Build factor graph
# ============================================================
graph = gtsam.NonlinearFactorGraph()
key = 1

prior_noise = noiseModel.Diagonal.Sigmas(np.array([0.4]*3 + [0.4]*3))
graph.add(gtsam.PriorFactorPose3(key, Pose3(), prior_noise))

obs_noise = noiseModel.Isotropic.Sigma(1, 0.01)

for pt in p_scan:
    ef = make_sdf_boundary_factor(pt, scene)
    graph.add(gtsam.CustomFactor(obs_noise, [key], ef))


# ============================================================
# Initial guess (wrong rotation + translation)
# ============================================================
# initial = gtsam.Values()
# initial_pose = Pose3(
#     Rot3.RzRyRx(0.4,-0.3,0.15),
#     Point3(0.4, -0.6, 0.35)
# )
# initial.insert(key, initial_pose)

initial = gtsam.Values()
initial.insert(key, Pose3())


# ============================================================
# Solve
# ============================================================
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(20)
result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

pose_opt = result.atPose3(key)

# print("\nInitial Pose\n", initial_pose)
print("\nOptimized Pose\n", pose_opt)

T_init = np.eye(4)
T_opt  = pose_opt.matrix()


# ============================================================
# Visualize
# ============================================================
pc_true = o3d.geometry.PointCloud()
pc_true.points = o3d.utility.Vector3dVector(p_scan)
pc_true.paint_uniform_color([0,1,0])

pc_init = o3d.geometry.PointCloud()
pi = (T_init[:3,:3] @ p_scan.T + T_init[:3,3:4]).T
pc_init.points = o3d.utility.Vector3dVector(pi)
pc_init.paint_uniform_color([1,0,0])

pc_opt = o3d.geometry.PointCloud()
po = (T_opt[:3,:3] @ p_scan.T + T_opt[:3,3:4]).T
pc_opt.points = o3d.utility.Vector3dVector(po)
pc_opt.paint_uniform_color([0,0,1])

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

print("\nVISUAL:")
print("Green  - true ellipse (identity)")
print("Red    - initial pose")
print("Blue   - optimized boundary aligned ellipse")

o3d.visualization.draw_geometries(
    [mesh, frame, pc_true, pc_init, pc_opt],
    window_name="RaycastingScene SDF Boundary Alignment"
)

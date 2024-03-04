import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt

# Load the point cloud
DATANAME = "appartment_cloud.ply"

data_folder="cloud_points/"

pcd = o3d.io.read_point_cloud(data_folder+DATANAME)

# Data preprocessing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# Statical outlier filter
nn = 16
std_multiplier = 10

filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1,0,0])

#o3d.visualization.draw_geometries([filtered_pcd, outliers])

# Voxel downsampling
voxel_size = 0.01

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
#o3d.visualization.draw_geometries([pcd_downsampled])

# Estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
radius_normals = nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)
#id(radius=radius_normals, max_nn=16, fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6,0.6,0.6])
#o3d.visualization.draw_geometries([pcd_downsampled,outliers])


# Extracting and Setting parameters

front = [0.99492346647821917, 0.033437973212523599, 0.094916794079881614]
lookat = [-0.008619773416219445, 0.02734387141551986, -0.032977855769833142]
up = [-0.097672258596389144, 0.093696959268069729, 0.99079816800627862]
zoom = 0.23999999999999996

pcd = pcd_downsampled
#o3d.visualization.draw_geometries([pcd], front=front, lookat=lookat, up=up, zoom=zoom)


# RANSAC plane segmentation

pt_to_plane_dist = 0.02

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)

[a,b,c,d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

#Paint the clouds
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom = zoom, front=front, lookat=lookat, up=up)

# Multi-order RANSEC
max_plane_idx = 6
pt_to_plane_dist = 0.02

segment_models = {}
segments = {}
rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab10")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i, "/", max_plane_idx, "done")

#o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+ [rest], zoom = zoom, front=front, lookat=lookat, up=up)

# DBSCAN clustering
labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+ [rest], zoom = zoom, front=front, lookat=lookat, up=up)
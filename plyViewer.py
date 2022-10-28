import numpy as np
import open3d as o3d

# Read .ply file
input_file = "out.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
#o3d.visualization.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format.
point_cloud_in_numpy = np.asarray(pcd.points)
#print("cloud points")
#print(point_cloud_in_numpy)
colors = np.asarray(pcd.colors)
#print("cloud colors")
#print(colors[:,2])
colors[:,2] = 0
#colors[3,:] = 0
#print(colors)

pcdWithoutGreen = o3d.geometry.PointCloud()
pcdWithoutGreen.points = o3d.utility.Vector3dVector(point_cloud_in_numpy)
pcdWithoutGreen.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcdWithoutGreen])

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
#colorsOrange = colors[:,0] ==1# and colors[:,1] > 127 and colors[:,2] < 80  
#colorsOrange = [colors if colors[:,2] < 80 else 0 for colorsGreen in colors[:,2]]
#colorsOrange = np.array([colors[:,3] for ])
pointOrange = np.empty((0,3), float)
pointPosOrange = np.empty((0,3), float)
#print(colors[0,0:3])
for i, pointColor in enumerate(colors[:,0:3]):
	#print(pointColor)
	if pointColor[0] > 0.6 and pointColor[1] > 0.3 and pointColor[2] < 0.4:
		#pointOrange.append(pointColor)
		pointOrange = np.append(pointOrange, np.array([pointColor]), axis=0)
		#np.append(pointOrange, pointColor, axis=0)
		#np.vstack([pointOrange, pointColor])
		#print(pointColor)
		#pointPosOrange.append(point_cloud_in_numpy[i,0:3])
		pointPosOrange = np.append(pointPosOrange, np.array([point_cloud_in_numpy[i,0:3]]), axis=0)
	#print("point")
	#print(point)
#colors[:,2] = 0
#colors[3,:] = 0
#print(colors)

#print("pointOrange")
#print(pointOrange)
#print("pointPosOrange")
#print(pointPosOrange)

pcdOrange = o3d.geometry.PointCloud()
pcdOrange.points = o3d.utility.Vector3dVector(pointPosOrange)
pcdOrange.colors = o3d.utility.Vector3dVector(pointOrange)
o3d.visualization.draw_geometries([pcdOrange])

#pcdWithoutGreen = o3d.geometry.PointCloud()
#pcdWithoutGreen.points = o3d.utility.Vector3dVector(point_cloud_in_numpy)
#pcdWithoutGreen.colors = o3d.utility.Vector3dVector(colorsOrange)
#o3d.visualization.draw_geometries([pcdWithoutGreen])

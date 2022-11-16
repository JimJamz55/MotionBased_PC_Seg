# examples/Python/Basic/icp_registration.py

import open3d as o3d
import numpy as np
import copy

from matplotlib import pyplot as plt


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    target = o3d.io.read_point_cloud("ObjMoveEx/bottle2.ply")
    # threshold = 0.02
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # # draw_registration_result(source, target, trans_init)
    # # print("Initial alignment")
    # # evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    # #                                                     threshold, trans_init)
    # # print(evaluation)
    #
    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # # draw_registration_result(source, target, reg_p2p.transformation)
    #
    transformation = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(source, target, transformation)

    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    p1_load = np.asarray(source_temp.points)
    p2_load = np.asarray(target_temp.points)
    p3_load = np.concatenate((p1_load, p2_load), axis=0)
    p1_color = np.asarray(source_temp.colors)
    p2_color = np.asarray(target_temp.colors)
    p3_color = np.concatenate((p1_color, p2_color), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p3_load)
    pcd.colors = o3d.utility.Vector3dVector(p3_color)
    # o3d.io.write_point_cloud("ObjMoveEx/combined.ply",p3_load)

    # pcd = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # inlier_cloud = pcd.select_by_index(inliers)
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label
                                             if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
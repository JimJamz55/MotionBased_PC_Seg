# examples/Python/Basic/icp_registration.py
from math import dist

import open3d as o3d
import numpy as np
import copy
from rtree import index

from matplotlib import pyplot as plt


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def combinePoints(source, target):
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
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(p3_load)
    # pcd.colors = o3d.utility.Vector3dVector(p3_color)
    # o3d.io.write_point_cloud("ObjMoveEx/combined.ply",p3_load)

    # pcd = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # inlier_cloud = pcd.select_by_index(inliers)
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
    # max_label = labels.max()
    # colors = plt.get_cmap("tab20")(labels / (max_label
    #                                          if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    return p3_load, p3_color

def sparse_subset3(points,colors,r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.
    https://codereview.stackexchange.com/questions/196104/removing-neighbors-in-a-point-cloud
    """
    result = []
    resultC = []
    pro = index.Property()
    pro.dimension = 3
    # pro.dat_extension = 'data'
    # pro.idx_extension = 'index'
    idx3d = index.Index(properties=pro)
    # idx3d.insert(1, (0, 60, 23.0, 0, 60, 42.0))
    # idx3d.intersection((-1, 62, 22, -1, 62, 43))
    for i, p in enumerate(points):
        px, py, pz = p
        nearby = idx3d.intersection((px - r, py - r, pz - r, px + r, py + r, pz + r))
        if all(dist(p, points[j]) >= r for j in nearby):
            result.append(p)
            resultC.append(colors[i])
            idx3d.insert(i, (px, py, pz, px, py, pz))

    if len(result) > 0:
        result = np.vstack(result)

    if len(resultC) > 0:
        resultC = np.vstack(resultC)

    return result, resultC

def differencePoints(source, target,r):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    points1 = np.asarray(source_temp.points)
    points2 = np.asarray(target_temp.points)
    colors1 = np.asarray(source_temp.colors)
    colors2 = np.asarray(target_temp.colors)

    result1 = []
    resultC1 = []
    # result2 = []
    # resultC2 = []
    pro = index.Property()
    pro.dimension = 3
    idx3d = index.Index(properties=pro)
    for i, p in enumerate(points1):
        px, py, pz = p
        nearby = idx3d.intersection((px - r, py - r, pz - r, px + r, py + r, pz + r))
        if all(dist(p, points1[j]) >= r for j in nearby):
            result1.append(p)
            resultC1.append(colors1[i])
            idx3d.insert(i, (px, py, pz, px, py, pz))

    if len(result1) > 0:
        result = np.vstack(result1)

    if len(resultC1) > 0:
        resultC = np.vstack(resultC1)

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(result1)
    # pcd1.colors = o3d.utility.Vector3dVector(resultC1)

    return result1, resultC1

if __name__ == "__main__":
    source = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    target = o3d.io.read_point_cloud("ObjMoveEx/bottle2.ply")
    p3_load, p3_color = combinePoints(source,target)
    # listOfRedPoints, listOfRedColors = sparse_subset3(p3_load, p3_color, 0.01)
    listOfRedPoints, listOfRedColors = differencePoints(source, target, 0.1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(listOfRedPoints)
    pcd.colors = o3d.utility.Vector3dVector(listOfRedColors)
    o3d.visualization.draw_geometries([pcd])
    print("DONE")
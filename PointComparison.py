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

def numpyToPC(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

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

def differencePoints(source, target, padding):
    source_temp = copy.deepcopy(source)
    points = np.asarray(source_temp.points)
    colors = np.asarray(source_temp.colors)
    distances = source.compute_point_cloud_distance(target)
    listOfRedPoints = []
    listOfRedColors = []
    twentyfifth, seventyfifth = np.quantile(distances, [0.25,0.75])
    iqr = seventyfifth-twentyfifth
    upperFence = seventyfifth+3*iqr
    for i, val in enumerate(distances):
        if val > upperFence + padding:
            listOfRedPoints.append(points[i])
            listOfRedColors.append(colors[i])

    return listOfRedPoints, listOfRedColors

def ransacDB(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # inlier_cloud = pcd.select_by_index(inliers)
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=0))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")((labels / (max_label if max_label > 0 else 1)))

    unique, counts = np.unique(labels, return_counts=True)
    order_labels = np.argsort(counts)
    # colors[labels < 0] = 0
    # colors[labels != 909090909090909090] = 0
    # colors[labels == order_labels[-1]] = [0.0356, 0.096, 0.043, 0]
    # colors[labels == order_labels[-2]] = [0.01234, 0.1234, 0.09382, 0]
    colors[labels != order_labels[-1]] = 0

    # colors1 = colors[labels != order_labels[-1]] = 0
    # colors2 = colors[labels != order_labels[-2]] = 0
    # colors = colors1+colors2
    # for num, i in enumerate(labels):
    #     if i != order_labels[-1] or i != order_labels[-2]:
    #         colors[num] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # segment_models = {}
    # segments = {}
    # max_plane_idx = 20
    # rest = pcd
    # for i in range(max_plane_idx):
    #     colors = plt.get_cmap("tab20")(i)
    #     segment_models[i], inliers = rest.segment_plane(
    #         distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    #     segments[i] = rest.select_by_index(inliers)
    #     labels = np.array(segments[i].cluster_dbscan(eps=d_threshold * 10, min_points=10))
    #     segments[i].paint_uniform_color(list(colors[:3]))
    #     rest = rest.select_by_index(inliers, invert=True)
    #     print("pass", i, "/", max_plane_idx, "done.")

    return pcd

if __name__ == "__main__":
    source = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    target = o3d.io.read_point_cloud("ObjMoveEx/bottle2.ply")
    # p3_load, p3_color = combinePoints(source,target)

    # listOfRedPoints, listOfRedColors = sparse_subset3(p3_load, p3_color, 0.01)
    listOfRedPoints, listOfRedColors = differencePoints(source, target, 0)

    pcd = numpyToPC(listOfRedPoints,listOfRedColors)
    pcd = ransacDB(pcd)
    o3d.visualization.draw_geometries([pcd])
    print("DONE")
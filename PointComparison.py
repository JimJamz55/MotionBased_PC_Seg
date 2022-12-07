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


def sparse_subset3(points, colors, r):
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
    twentyfifth, seventyfifth = np.quantile(distances, [0.25, 0.75])
    iqr = seventyfifth - twentyfifth
    upperFence = seventyfifth + 3 * iqr
    for i, val in enumerate(distances):
        if val > upperFence + padding:
            listOfRedPoints.append(points[i])
            listOfRedColors.append(colors[i])

    return listOfRedPoints, listOfRedColors


def ransacDB(pcd, highlight):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # inlier_cloud = pcd.select_by_index(inliers)
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=0))

    unique, counts = np.unique(labels, return_counts=True)
    order_labels = np.argsort(counts)

    if highlight:
        max_label = labels.max()
        colors = plt.get_cmap("tab20")((labels / (max_label if max_label > 0 else 1)))
        colors_temp = copy.deepcopy(colors)
        colors[labels != order_labels[-1]] = 0
        colors_temp[labels != order_labels[-2]] = 0
        colors = colors + colors_temp

        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels, order_labels


def pcOnlyLabel(pcd, target, labels):
    source_temp = copy.deepcopy(pcd)
    points = np.asarray(source_temp.points)
    colors = np.asarray(source_temp.colors)
    listOfRedPoints = []
    listOfRedColors = []
    for i, val in enumerate(labels):
        if val == target:
            listOfRedPoints.append(points[i])
            listOfRedColors.append(colors[i])

    pcd = numpyToPC(listOfRedPoints, listOfRedColors)

    # colors = plt.get_cmap("tab20")((labels / 10)) #arbitrary value
    # colors[labels != target] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def getCOMMainClusters(pcd, labels, order_labels):
    source_temp = copy.deepcopy(pcd)
    points = np.asarray(source_temp.points)
    coords = [[], []]
    COMs = []
    for i, label in enumerate(labels):
        if label == order_labels[-1]:
            coords[0].append(points[i])
        elif label == order_labels[-2]:
            coords[1].append(points[i])
    for list in coords:
        sumX = 0
        sumY = 0
        sumZ = 0
        count = len(list)
        for coord in list:
            sumX += coord[0]
            sumY += coord[1]
            sumZ += coord[2]
        COMs.append((sumX / count, sumY / count, sumZ / count))
    return COMs


def likelyObject(pcd, labels, order_labels):
    """
    Assume most likely object is the one closer to camera
    """
    COMs = getCOMMainClusters(pcd, labels, order_labels)
    # print(COMs)
    if COMs[0][2] < COMs[1][2]:
        return order_labels[-2]
    else:
        return order_labels[-1]


def getOrienBoundBox(pcd):
    bb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points, robust=False)
    return bb


def drawBB(bb, vis):
    # https://stackoverflow.com/questions/62938546/how-to-draw-bounding-boxes-and-update-them-real-time-in-python
    corners = np.asarray(bb.get_box_points())
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
    #          [4, 5], [5, 6], [6, 7], [4, 7],
    #          [0, 4], [1, 5], [2, 6], [3, 7]]

    lines = [[0, 3], [3, 5], [5, 2], [2, 0],
             [1, 6], [6, 4], [4, 7], [7, 1],
             [1, 0], [3, 6], [4, 5], [7, 2]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Display the bounding boxes:
    vis.add_geometry(line_set)


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("ObjMoveEx/bottle1.ply")
    target = o3d.io.read_point_cloud("ObjMoveEx/bottle2.ply")
    # p3_load, p3_color = combinePoints(source,target)

    # listOfRedPoints, listOfRedColors = sparse_subset3(p3_load, p3_color, 0.01)
    listOfRedPoints, listOfRedColors = differencePoints(source, target, 0)
    pcd = numpyToPC(listOfRedPoints, listOfRedColors)

    pcd, labels, order_labels = ransacDB(pcd, highlight=False)
    objectLabel = likelyObject(pcd, labels, order_labels)
    # print(order_labels,objectLabel)
    pcd = pcOnlyLabel(pcd, objectLabel, labels)

    bb = getOrienBoundBox(pcd)
    print(bb.volume())

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    drawBB(bb, vis)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("DONE")

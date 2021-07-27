import argparse
import numpy as np
from numpy import linalg as LA
from traditional.skylineND import skyline
from scipy.spatial import ConvexHull


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


def evaluate(true_set, pred_set, points):
    numApprox = 0

    approx_true_set = true_set - true_set.intersection(pred_set)
    approx_pred_set = pred_set - true_set.intersection(pred_set)
    for id in approx_true_set:
        for jd in approx_pred_set:
            tempDist = LA.norm(points[id - 1] - points[jd - 1], 2)
            if tempDist < 0.1:
                numApprox = numApprox + 1
                break

    numCorrect = len(true_set.intersection(pred_set)) + numApprox
    if numCorrect == 0:
        recall = 0.0
        precision = 0.0
    else:
        recall = numCorrect / len(true_set)
        precision = numCorrect / len(pred_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (precision + recall)
    return numCorrect, recall, precision, f1


def CHarea(index, all_points, shift=True):
    index = np.array(list(index))
    if shift:
        index = index - 1
    pts = all_points[index]
    if len(pts) < 3:
        return 0
    hull = ConvexHull(pts)
    return hull.volume


def enclosed(index, all_points, shift=True):
    index = np.array(list(index))
    if shift:
        index = index - 1

    if len(index) == 0:
        return 0
    else:
        sky_points = all_points[index].tolist()
    return enclosed_helper(sky_points)


def enclosed_helper(points):
    _, sky_points = skyline(points)
    sky_points.sort()
    dim = len(points[0])
    V = 0
    if dim == 2:
        for i in range(len(sky_points)):
            if i == 0:
                V += (1 - sky_points[i][0]) * (1 - sky_points[i][1])
            else:
                V += (1 - sky_points[i][0]) * (sky_points[i - 1][1] - sky_points[i][1])

    else:
        temp = []
        for i in range(len(sky_points)):
            temp.append(sky_points[i][1:dim])
            if i == len(sky_points) - 1:
                V += (1 - sky_points[i][0]) * enclosed_helper(temp)
            else:
                V += (sky_points[i + 1][0] - sky_points[i][0]) * enclosed_helper(temp)
    return V

# coding=utf-8
'''K-means算法的实现
K-means 算法思想：
1.初始化聚类个数及中心点。人为给定。
2.划分数据到每个类。计算样本数据到各聚类中心的距离（欧式距离或其他距离等），把每个样本划分到最近的类中。
3.重新计算类中心点。一般是求坐标平均值。
4.重复2、3步骤。直到聚类中心不再移动位置。'''


import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import math

d = sio.loadmat('dataset16.mat')
data = (d['kmeans_test'])


# x, y = data[:, 0], data[:, 1]
# 计算欧式距离(Euclidean distance)
def eucl_distance(vec_1, vec_2):
    return math.sqrt(np.sum(np.power(vec_1 - vec_2, 2)))


# s随机初始化聚类中心(inicialize centroids)
def init_centroids(data_set, k):
    data_num, dimension = data_set.shape
    centroids = np.zeros((k, dimension))
    for i in range(k):
        index = int(np.random.uniform(0, data_num))
        centroids[i, :] = data_set[index, :]
    return centroids


def k_means(data_set, k):
    data_num = data_set.shape[0]
    clus_assiment = np.zeros((data_num, 1))
    clus_adjusted = True
    # 初始化聚类
    centroids = init_centroids(data_set, k)
    while clus_adjusted:
        clus_adjusted = False
        # 对于每个样本
        for i in range(data_num):
            min_dist = 1000000
            min_index = 0
            # 对于每个聚类中心
            for j in range(k):
                # 找到最近的聚类
                distance = eucl_distance(centroids[j, :], data_set[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            # 更新点的聚类
            if clus_assiment[i, 0] != min_index:
                clus_adjusted = True
                clus_assiment[i] = min_index
        # 更新聚类
        for j in range(k):
            points = data_set[np.nonzero(clus_assiment[:, 0] == j)[0]]
            centroids[j, :] = np.mean(points, axis=0)
    print('K-means聚类完成~')
    # 绘图
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    for i in range(data_num):
        mark_index = int(clus_assiment[i, 0])
        plt.scatter(data_set[i, 0], data[i, 1], color=colors[mark_index])
    plt.show()

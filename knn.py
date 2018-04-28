# coding=utf-8
'''KNN算法的实现，时间复杂度O(n)
计算测试数据与各个训练数据之间的距离； 
按照距离的递增关系进行排序； 
选取距离最小的K个点； 
确定前K个点所在类别的出现频率； 
返回前K个点中出现频率最高的类别作为测试数据的预测分类'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import operator

d = sio.loadmat('dataset16.mat')
data = (d['fourclass'])
x, y = data[:, 1], data[:, 2]  # 样本的两个特征x
xy_set = np.array([np.reshape(x, len(x)), np.reshape(y, len(y))]).transpose()
label = data[:, 0]  # 对应的标签y


def Knn_Algorithm(input=([0, 0.5]), k=3):
    # 计算输入点据样本中各个点的距离
    data_num = data.shape[0]
    diff_xy = np.tile(input, (data_num, 1)) - xy_set
    diff_xy_2 = diff_xy ** 2
    dist_2 = diff_xy_2[:, 0] + diff_xy_2[:, 1]
    # 另一种操作：dist_2 = np.sum(cha_xy_2,axis=1)
    distance = dist_2 ** 0.5
    # 对距离进行排序
    sorted_index = np.argsort(distance)
    # 对所属类别进行统
    cluster_cnt = {}
    for i in range(k):
        to_label = label[sorted_index[i]]
        if 'to_label' in cluster_cnt.keys():
            cluster_cnt[to_label] += 1
        else:
            cluster_cnt[to_label] = 1
            # 更好的操作：cluster_cnt[to_label] = cluster_cnt.get(to_label,0)+1
    # 遍历一次得出结果
    max = 0
    for key, value in cluster_cnt.items():
        if (value) > max:
            result = key
    return result

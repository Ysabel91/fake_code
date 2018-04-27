# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time

'''符号函数'''
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

'''
逻辑回归训练
'''
def train_logRegres(train_x, train_y, opts):
    startTime = time.time()
    numSamples, numFeatures = np.shape(train_x)
    alpha = opts['alpha'] #步长
    maxIter = opts['maxIter']#迭代次数
    #权重
    weights = np.ones((numFeatures, 1)) #初始化参数为1

    for k in range(maxIter):
        output = sigmoid(np.dot(train_x, weights))
        error = train_y - output
        weights = weights + alpha * np.dot(train_x.transpose(), error)

    print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    print(weights)
    return weights


'''逻辑回归测试'''
def LogRegres_ttest(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


if __name__ == "__main__":
    # "step 1: loading data..."

    train_data = np.array([
        [-0.017612, 14.053064, 0],
        [-1.395634, 4.662541, 1],
        [-0.752157, 6.538620, 0],
        [-1.322371, 7.152853, 0],
        [0.423363, 11.054677, 0],
        [0.406704, 7.067335, 1],
        [0.667394, 12.741452, 0],
        [-2.460150, 6.866805, 1],
        [0.569411, 9.548755, 0],
        [-0.026632, 10.427743, 0],
        [0.850433, 6.920334, 1],
        [1.347183, 13.175500, 0],
        [1.176813, 3.167020, 1],
        [-1.781871, 9.097953, 0],
        [-0.566606, 5.749003, 1],
        [0.931635, 1.589505, 1],
        [-0.024205, 6.151823, 1],
        [-0.036453, 2.690988, 1],
        [-0.196949, 0.444165, 1],
        [1.014459, 5.754399, 1],
        [1.985298, 3.230619, 1],
        [-1.693453, -0.557540, 1]
    ])

    train_x = train_data[:, [0, 1]]
    train_y = train_data[:, [2]]

    test_x = train_x
    test_y = train_y

    # "step 2: training..."
    alpha = 0.01
    maxIter = 200
    # gradDescent ,stocGradDescent ,smoothStocGradDescent
    optimizeType = 'gradDescent'  # 调用的方法

    opts = {'alpha': alpha, 'maxIter': maxIter, 'optimizeType': optimizeType}
    optimalWeights = train_logRegres(train_x, train_y, opts)

    ## step 3: testing
    accuracy = LogRegres_ttest(optimalWeights, test_x, test_y)

    ## step 4: show the result
    print('The classify accuracy is: %.3f%%' % (accuracy * 100))
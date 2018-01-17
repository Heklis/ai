# -*- coding: UTF-8 -*-

import knn
import preprocess
import numpy as np

DATASET_PATH = 'iris.data'

# 读取数据集
dataset = []
with open(DATASET_PATH, 'r') as file:
    for line in file:
        dataset.append(line.strip().split(','))
# 特征值规范化
dataset = preprocess.minMaxScale(dataset)

# bestK, error = knn.train(dataset, True)
# print '最优k:%d' % bestK
# print '平均误差:%f' % error


# 使用knn训练(找出最优k)
# nominalFeats = range(6)
bestK, accuracy = knn.train(dataset, False)
print '最优k:%d' % bestK
print '平均精度:%f%%' % (accuracy * 100)
# 使用knn预测(测试集自己编的)
testset = [[5.0, 2.2, 4.7, 3.4]]
classList = knn.classify(dataset, testset, bestK)
print '预测类别：%s' % classList[0]

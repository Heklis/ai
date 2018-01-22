# -*- coding: UTF-8 -*-

import knn
import preprocess
import numpy
DATASET_PATH = 'datasets/iris.data'

# 读取数据集
dataset = []
with open(DATASET_PATH, 'r') as file:
	for line in file:
		line = line.strip().split(',')
		sample = [float(line[i]) for i in range(len(line)-1)]
		sample.append(line[-1])
		dataset.append(sample)
# dataset = numpy.loadtxt(DATASET_PATH)
# 特征值规范化
scaler = preprocess.MinMaxScaler(dataset)
dataset = scaler.scale(dataset)

# bestK, error = knn.train(dataset, True)
# print '最优k:%d' % bestK
# print '平均误差:%f' % error
bestK, accuracy = knn.train(dataset, False)
print '最优k:%d' % bestK
print '平均精度:%f%%' % (accuracy * 100)


# 使用knn训练(找出最优k)
# 使用knn预测(测试集自己编的)
# testset = [[800, 0, 0.3048, 71.3, 0.00266337]]
# testset = scaler.scale(testset)
# classList = knn.classify(dataset, testset, bestK, True)
# print '预测类别：%s' % classList[0]

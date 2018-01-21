# -*- coding: UTF-8 -*-

import numpy as np
import k_means
import preprocess

# 读取数据集
dataset = np.loadtxt("datasets/movement_libras.data", delimiter=',', dtype=float)
# with open("datasets/user_modeling.data", 'r') as file:
# 	for line in file:
# 		dataset.append(line.strip().split(','))
# for sample in dataset:
# 	for i in range(len(sample)-1):
# 		sample[i] = float(sample[i])
# 特征标准化
scaler = preprocess.StandardScaler(dataset)
dataset = scaler.scale(dataset)

# 用Kmeans聚类
clusters, meanVectors = k_means.clustering(dataset, 15)
# 计算平方误差和
errors = k_means.evaluate(clusters, meanVectors)

numSample = len(dataset)
for i in range(len(clusters)):
	prop = float(len(clusters[i]))/numSample*100
	print 'cluster %d : %d	%f%%' %(i+1, len(clusters[i]), prop)
print '平方误差和：', errors

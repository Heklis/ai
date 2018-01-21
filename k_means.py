# -*- coding: UTF-8 -*-

from random import randint

def clustering(dataset, k, count=-1):
	"""把dataset划分成k个簇

	Args:
	    dataset (array): 数据集
	    k (int): 聚类簇数
	    count(int): 最大迭代次数

	Returns:
	    clusters (list): 分好的簇
	    means (list): 均值向量
	"""
	numSample = len(dataset)
	# 均值向量(随机选取数据集中K个向量)
	means = [dataset[randint(0,numSample-1)] for i in range(k)]
	clusters = []
	for i in range(k):
		cluster = []
		clusters.append(cluster)

	numFeat = len(dataset[0]) - 1
	while True:
		# 清空簇
		for i in range(k):
			clusters[i] = []
		# 根据当前均值向量划分簇
		for sample in dataset:
			dists = []
			for mean in means:
				dist = 0.0
				# 计算样本与均值向量之间的距离
				for i in range(numFeat):
					dist += (sample[i] - mean[i]) ** 2
				dists.append(dist)
			# 找出与样本距离最近的均值向量索引
			index = dists.index(min(dists))
			# 将样本划入相应的簇
			clusters[index].append(sample)

		isChange = False
		# 计算新均值向量
		for i in range(k):
			# 簇中所有样本的和向量
			sumVector = [0.0] * numFeat

			for sample in clusters[i]:
				for j in range(numFeat):
					sumVector[j] += sample[j]

			numSample = len(clusters[i])
			new_mean = [value / numSample for value in sumVector]
			# 更新均值向量
			if new_mean != means[i]:
				means[i] = new_mean
				isChange = True

		# 如果当前均值向量均未更新或者达到最大迭代次数
		if not isChange or count == 0:
			break
		# 控制迭代次数
		if count > 0:
			count -= 1

	return clusters, means


def evaluate(clusters, meanVectors):
	"""评估聚类结果：通过平方误差

	Args:
	    clusters (list): 分好的簇
	    meanVectors (list): 均值向量

	Returns:
	    float：平方误差和
	"""
	# 平方误差
	errors = 0
	k = len(clusters)
	numFeat = len(clusters[0][0]) - 1
	for i in range(k):
		for sample in clusters[i]:
			for j in range(numFeat):
				errors += (sample[j] - meanVectors[i][j]) ** 2
	return errors
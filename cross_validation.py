# -*- coding: UTF-8 -*-

def split(dataset, isRegress, k=10):
	"""把数据集划分为k组训练集和验证集

	Args:
	    dataset (list): 数据集
	    k (int, optional): 交叉验证折数

	Returns:
	    train_validate_sets:
	    训练集、验证集对的集合
	"""
	if isRegress:
		subsets = []
		# 划分出k个子集
		step = len(dataset) / k
		for i in range(k):
			subsets.append(dataset[i*step : (i+1)*step])
		return generate_train_validate_sets(subsets, k)


	classsets = {}
	# 把数据集按照类别划分
	for sample in dataset:
		label = sample[-1]
		if label not in classsets.keys():
			classsets[label] = []
		classsets[label].append(sample)

	subsets = []
	for i in range(k):
		subset = []
		subsets.append(subset)
	# 划分出k个子集
	for classset in classsets.values():
		step = len(classset) / k
		for i in range(k):
			subsets[i].extend(classset[i*step : (i+1)*step])
		# subsets[k-1].extend(classset[k*step:])


	return generate_train_validate_sets(subsets, k)


def generate_train_validate_sets(subsets, k):
	# 合并成k组训练集、验证集
	train_validate_sets = []
	for i in range(k):
		train_validate_set = []
		train_validate_set.append(merge(subsets[:i]+subsets[i+1:]))
		train_validate_set.append(subsets[i])
		train_validate_sets.append(train_validate_set)

	return train_validate_sets


def merge(subsets):
	"""将多个子集合并

	Args:
	    subsets (list): 想要合并的子集集合

	Returns:
	    list: 子集合并后的集合
	"""
	s = []
	for subset in subsets:
		s.extend(subset)
	return s


def evaluate(classList, testset, isRegress=False):
	"""评估对验证集分类结果，计算精度

	Args:
	    classList (list): 分类结果
	    testset (list): 验证集

	Returns:
	    float: 精度(分类)，平均绝对误差（回归）
	"""
	# 如果是回归任务
	if isRegress:
		errors = 0.0
		for i, sample in enumerate(testset):
			# 求绝对值误差
			errors += abs(classList[i] - sample[-1])
		mean_error = errors / len(testset)
		return mean_error

	correctCount = 0
	for i, sample in enumerate(testset):
		if classList[i] == sample[-1]:
			correctCount += 1
	numSample = len(testset)
	accuracy = float(correctCount) / numSample
	return accuracy
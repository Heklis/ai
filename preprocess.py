# -*- coding: UTF-8 -*-

class StandardScaler:

    """z-score标准化

    """

    def __init__(self, dataset):
        # 特征数
        self.numFeat = len(dataset[0]) - 1
        # 样本均值
        self.mean = [0] * self.numFeat
        # 样本标准差
        self.std = [0] * self.numFeat

        self.fit(dataset)

    def fit(self, dataset):
        """匹配数据集，保存其均值和标准差

        Args:
            dataset (array): 数据集
        """
        for i in range(self.numFeat):
            sum = 0
            for example in dataset:
                sum += example[i]
            # 样本均值
            mean = sum / len(dataset)
            self.mean[i] = mean

            sum = 0
            for example in dataset:
                sum += (example[i] - mean) ** 2
            # 样本标准差
            self.std[i] = (sum / (len(dataset) - 1)) ** 0.5

    def scale(self, dataset):
        """使用存储的均值和标准差对数据集标准化

        Args:
            dataset (array): 数据集

        Returns:
            array: 标准化的数据集
        """
        for i in range(self.numFeat):
            for example in dataset:
                example[i] = (example[i] - self.mean[i]) / self.std[i]

        return dataset



class MinMaxScaler:

    """利用最值标准化

    """

    def __init__(self, dataset):
        # 特征数
        self.numFeat = len(dataset[0]) - 1
        # 样本特征最大值
        self.max = [0] * self.numFeat
        # 样本特征最小值
        self.min = [0] * self.numFeat

        self.fit(dataset)

    def fit(self, dataset):
        """匹配数据集，保存其特征最值

        Args:
            dataset (array): 数据集
        """
        for i in range(self.numFeat):
            maxVal = dataset[0][i]
            minVal = dataset[0][i]
            for example in dataset:
                if maxVal < example[i]:
                    maxVal = example[i]
                if minVal > example[i]:
                    minVal = example[i]
            self.max[i] = maxVal
            self.min[i] = minVal

    def scale(self, dataset):
        """使用存储的最值对数据集标准化

        Args:
            dataset (array): 数据集

        Returns:
            array: 标准化的数据集
        """
        for i in range(self.numFeat):
            # 规范化
            for example in dataset:
                example[i] = (example[i] - self.min[i]) \
                / (self.max[i] - self.min[i])

        return dataset


def minMaxScale(dataset, nominalFeats=[]):
    """
    利用两个最值缩放特征值区间
    arg:
        dataset：数据集
        nominalFeats(list): 分类型特征索引
    return:
        规范化的数据集
    """
    # 特征数
    numFeat = len(dataset[0]) - 1

    for i in range(numFeat):
        # 如果是分类型特征，不作处理
        if i in nominalFeats:
            continue

        featVals = []
        for example in dataset:
            # 缺失值处理
            if example[i] == '?':
                continue
            example[i] = float(example[i])
            featVals.append(example[i])
        maxVal = max(featVals)
        minVal = min(featVals)
        # 规范化
        for example in dataset:
            # 缺失值处理
            if example[i] == '?':
                continue
            example[i] = (example[i] - minVal)/(maxVal - minVal)

    return dataset

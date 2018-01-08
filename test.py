# -*- coding: UTF-8 -*-

from math import log

DATASET_PATH = "watermelon_2.data"

# 计算数据集的信息熵
def ent(dataSet):
    # 样本数
    numEntries = len(dataSet)
    # 标记计数
    labelCounts = {}
    # 统计各类别出现次数
    for sample in dataSet:
        currentLabel = sample[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 信息熵
    ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        ent -= prob * log(prob, 2)
    return ent

# 计算某特征信息增益
def gain(dataSet, feature_index):
    # 按照特征feature划分数据集
    sets = {}
    for sample in dataSet:
        value = sample[feature_index]
        if value not in sets.keys():
            sets[value] = []
        sets[value].append(sample)
    # for s in sets:
    #     print s
    #     print sets[s]
    #     print '------------'
    gain = ent(dataSet)
    for subSet in sets.values():
        gain -= float(len(subSet)) / len(dataSet) * ent(subSet)
    return gain
     
# 选择最优划分属性
def selectOptimumFeature(dataSet, numFeature):
    gains = []
    for i in range(numFeature):
        gains.append(gain(dataSet, i))
    return gains.index(max(gains))


def main():
    # 读取数据集
    dataSet = []
    with open(DATASET_PATH, 'r') as file:
        for line in file:
            dataSet.append(line.strip().split(','))
    # 特征个数
    numFeature = len(dataSet[0]) - 3

    print selectOptimumFeature(dataSet, numFeature)
    
def generateTree():    

if __name__ == '__main__':
    main()
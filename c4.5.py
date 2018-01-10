# -*- coding: UTF-8 -*-

from math import log
import treePlotter

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

# 计算某特征信息增益率
def gainRatio(dataSet, feature_index):
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
    # 信息增益
    gain = ent(dataSet)
    # 固有值
    iv = 0
    for subSet in sets.values():
        prop = float(len(subSet)) / len(dataSet)
        gain -= prop * ent(subSet)
        iv -= prop * log(prop, 2)

    return gain, gain / iv

# 选择最优划分属性
def selectOptimumFeature(dataSet):
    gains = []
    gainRatios = []
    gainSum = 0
    # 特征个数
    numFeature = len(dataSet[0]) - 3
    for i in range(numFeature):
        gain, gainRat = gainRatio(dataSet, i)
        gainSum += gain
        gains.append(gain)
        gainRatios.append(gainRat)
    # 平均信息增益
    averGain = gainSum / numFeature

    for i, gain in enumerate(gains):
        if gain <= averGain:
            gainRatios[i] = 0

    return gainRatios.index(max(gainRatios))

# 获得数据集中数量最多的标记类型
def majorityLabel(labelList):
    labelSet = set(labelList)
    labelCount = {}
    for labelClass in labelSet:
    	labelCount[labelClass] = labelList.count(labelClass)
    counts = labelCount.values()
    return labelCount.keys()[counts.index(max(counts))]

# 生成决策树
def generateTree(dataSet, featureNames):
    labelList = [example[-1] for example in dataSet]
    # 所有样本都是同一类
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    # 特征被划分完
    if len(dataSet[0]) == 1:
        return majorityLabel(labelList)
    bestFeat = selectOptimumFeature(dataSet)
    bestFeatName = featureNames[bestFeat]

    # if(bestFeat == -1):        #特征一样，但类别不一样，即类别与特征不相关，随机选第一个类别做分类结果
    # return labelList[0]
    myTree = {bestFeatName:{}}
    del(featureNames[bestFeat])
    # 划分子集
    subsets = {}
    for sample in dataSet:
        value = sample[bestFeat]
        if value not in subsets.keys():
            subsets[value] = []
        subSample = sample[:bestFeat] + sample[bestFeat+1:]
        subsets[value].append(subSample)
    # 生成决策树分支
    for value in subsets:
        subnames = featureNames[:]
        myTree[bestFeatName][value] = generateTree(subsets[value], subnames)

    return myTree

def main():
    # 读取数据集
    dataSet = []
    with open(DATASET_PATH, 'r') as file:
        for line in file:
            dataSet.append(line.strip().split(','))

    # 特征名称
    featureNames = ['seze', 'gendi', 'qiaosheng', 'wenli', 'qibu', 'chugan']
    desicionTree = generateTree(dataSet, featureNames)
    print desicionTree
    treePlotter.createPlot(desicionTree)

if __name__ == '__main__':
    main()
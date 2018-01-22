# -*- coding: UTF-8 -*-
import cross_validation


def train(dataset, isRegress):
    """根据交叉验证选取最优的k
    k范围：1-20

    Args:
        dataset (list): 数据集
        isRegress(boolean): 是否是回归任务

    Returns:
        bestK(int):最优K
        maxAccuracy(float):最大精度
    """
    ratios = []
    for k in range(1, 20):
        # 10折交叉验证
        # 把数据集划分为10组训练集和验证集
        train_validate_sets = \
        cross_validation.split(dataset, isRegress)
        # 用KNN分类，然后计算分类精度
        accuracys = 0.0
        for pair in train_validate_sets:
            trainset = pair[0]
            validateset = pair[1]
            # 获取预测分类列表
            classList = classify(
                trainset, validateset, k, isRegress)
            # 对分类结果进行评估
            accuracy = cross_validation.evaluate(
                classList, validateset, isRegress)
            accuracys += accuracy
        averageAccuracy = accuracys / 10
        ratios.append(averageAccuracy)
    # 如果是回归，取最小误差
    if isRegress:
        maxAccuracy = min(ratios)
    else:
        maxAccuracy = max(ratios)
    # 得到最优k
    bestK = ratios.index(maxAccuracy) + 1
    return bestK, maxAccuracy


def classify(trainset, testset, k, isRegress=False):
    """采用k,使用trainset，对testset预测类别

    Args:
        trainset (list): 训练集
        testset (list): 测试集
        k (int): 最邻近参数
        isRegress(boolean): 是否是回归任务

    Returns:
        list: 预测分类
    """
    numFeat = len(trainset[0]) - 1
    # 分类结果
    classList = []

    for sample in testset:
        dists = []
        for example in trainset:
            label = example[-1]
            # 计算欧氏距离
            dist = calDist(sample, example, numFeat)

            dists.append((dist, label))
        # 对欧氏距离排序
        dists = sorted(dists, key=lambda d: d[0])
        # 如果是回归任务,计算前k个样例的加权平均标记
        if isRegress:
            # 标记和
            sumLabel = 0.0
            for i in range(k):
                sumLabel += float(dists[i][1])
            cla = sumLabel / k
            classList.append(cla)
            continue

        # 对前k个样例找到数量最多的标记
        count = {}
        for i in range(k):
            label = dists[i][1]
            if label not in count.keys():
                count[label] = 0
            count[label] += 1
        countValues = count.values()
        cla = count.keys()[countValues.index(max(countValues))]
        classList.append(cla)

    return classList


def calDist(sample, example, numFeat):
    """计算欧氏距离

    Args:
        sample (list): 样本
        example (list): 样例
        numFeat (int): 特征数

    Returns:
        float: 欧氏距离
    """
    dist = 0.0
    for i in range(numFeat):
        dist += (float(sample[i]) - float(example[i]))**2
    return dist
        # # 如果特征为分类型
        # if i in nominalFeats:
        #     # 缺失值处理
        #     # if sample[i] == '?' or example[i] == '?':
        #     #     dist += 1
        #     #     continue
        #     if sample[i] != example[i]:
        #         dist += 1
        #     continue

        # 缺失值处理
        # if sample[i] == '?' and example[i] == '?':
        #     dist += 1
        #     continue
        # if sample[i] == '?':
        #     dist += max(abs(1-example[i]), abs(0-example[i]))
        #     continue
        # if example[i] == '?':
        #     dist += max(abs(1-sample[i]), abs(0-sample[i]))
        #     continue
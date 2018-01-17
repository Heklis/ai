# -*- coding: UTF-8 -*-


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

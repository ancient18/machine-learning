
from math import log
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from decisionTree.treePlotter import createPlot


def createDataSet():

    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    shannonEnt = 0.0  # 经验熵(香农熵)
    # 计算香农熵（需要自己实现）
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt  # 返回经验熵(香农熵)


# if __name__ == '__main__':
#     dataSet, features = createDataSet()
#     print(calcShannonEnt(dataSet))


def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        # if bestInfoGain == 0:  # 查看元素
        #     print(uniqueVals)
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            # print(subDataSet)
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        # 找到最大的信息增益，记录信息增益最大的特征的索引值
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最优特征的索引值


def buildDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同，停止划分
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityClass(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # if(bestFeatLabel == "1.0"):
    #     bestFeatLabel = "right"
    # else:
    #     bestFeatLabel = "left"
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制
        myTree[bestFeatLabel][value] = buildDecisionTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def majorityClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print("最优特征索引值:" + str(buildDecisionTree(dataSet, features)))
    createPlot(buildDecisionTree(dataSet, features))


# data_init = pd.read_csv('TUANDROMD.csv')
# shape = data_init.shape


# # 合并data_init非标签属性
# data_init_1 = data_init.iloc[:, :-1]
# data_init_2 = data_init.iloc[:, -1]


# x_train, x_test, y_train, y_test = train_test_split(
#     data_init_1, data_init_2, random_state=1)


# # 合并x_train.values和y_train.values

# train_data = np.c_[x_train.values, y_train.values]
# test_data = np.c_[x_test.values, y_test.values]

# # 训练集的属性名
# train_features = x_train.columns.tolist()

# # 将train_data和test_data转换为list
# train_data_list = train_data.tolist()
# test_data_list = test_data.tolist()


# # print(train_data_list)
# result = buildDecisionTree(train_data_list, train_features)
# print("最优特征索引值:" + str(result))

# createPlot(result)

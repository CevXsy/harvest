import csv
import math
import random
import operator
"""
    KNN方法用于分类
    已经有一些已知类型的数据，暂称其为训练集。
    当一个新数据（暂称其为测试集）进入的时候，开始跟训练集数据中的每个数据点求距离，
    挑选与这个训练数据集中最近的K个点看这些点属于什么类型，用少数服从多数的方法将测试数据归类
"""
def openCsv(fileName, split, trainingSet=[], testSet=[]):
    """
        提取数据 D:\sublime\lingyu\ML\iris.csv
        分成训练集和测试集
    """
    with open(fileName, "r") as csvFile:
        lines = csv.reader(csvFile)
        dataSets = list(lines)
        for x in range(len(dataSets)):
            for y in range(4):
                dataSets[x][y] = float(dataSets[x][y]) # str转换float类型
            if random.random() < split:
                trainingSet.append(dataSets[x])
            else:
                testSet.append(dataSets[x])

def euclideanDistance(instance1,instance2,length):
    """
        计算测试集与每个训练集的距离
    """
    distance = 0
    for x in range(length):
        distance += math.pow((float(instance1[x]) - float(instance2[x])),2)
    return math.sqrt(distance)

def getNeighbors(testInstance,trainingSet,k):
    """
        获取最邻近测试集k个点
    """
    distances = []
    length = len(testInstance)-1  # 判断测试集是几维
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    neighbors = []
    distances.sort(key=operator.itemgetter(1)) # 按小到大排序
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    """
        获取最邻近测试集k个点,比较k个点中那个结果多，少数服从多数
    """
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes =sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getAccuracy(predicted,testSet):
    """
        判断测试集的正确率
    """
    correct = 0
    for x in range(len(testSet)):
        if predicted[x] == testSet[x][-1]:
            correct +=1
    return (correct/float(len(testSet)))*100

def main():
    trainingSet = []
    testSet = []
    fileName = "./iris.txt"
    sep = 0.67
    openCsv(fileName,sep, trainingSet, testSet)
    print("Train set:", repr(len(trainingSet)))
    print("Test set:", repr(len(testSet)))
    predicted = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(testSet[x],trainingSet,k)
        result = getResponse(neighbors)
        predicted.append(result)
        print('>predicted:'+result+', testRusult:'+testSet[x][-1])
    current = getAccuracy(predicted,testSet)
    print("length predicted:"+repr(len(predicted)))
    print("Accuracy:"+repr(current)+'%')

if __name__ == '__main__':
    main()

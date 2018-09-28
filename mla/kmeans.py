import numpy as np

"""
    kmeans  选择中心点
    X  数据集
    k  分多少类
    maxNum  循环的最大次数
"""
def kmeans(X,k,maxNum):
    numPoints,numDim = X.shape
    # print(numPoints,numDim)
    dataSet = np.zeros((numPoints,numDim+1))
    """
        dataSet输出数据
        [[ 1.  1.  0.]
        [ 2.  1.  0.]
        [ 4.  3.  0.]
        [ 5.  4.  0.]]
    """
    dataSet[:,:-1] = X
    # print(dataSet)
    # 随机选择中心点
    centroids = dataSet[np.random.randint(numPoints,size=k),:]
    # 给选出的中心点赋值分类
    """
       centroids输出的类型
         [[ 5.  4.  1.]
         [ 4.  3.  2.]]
    """
    centroids[:,-1]=range(1,k+1)

    oldCentroids = None
    #  迭代次数
    iteractions = 0

    while not shouldStop(centroids,oldCentroids,maxNum,iteractions):
        print("iteractions:\n",iteractions)
        print("dataSet:\n",dataSet)
        print("centroids:\n",centroids)
        print("oldCentroids:\n",oldCentroids)
        iteractions += 1
        oldCentroids = np.copy(centroids)
        #更新数据集
        updateLabels(dataSet,centroids)
        #重新获取中心点
        centroids = getCentroids(dataSet,k)

    return dataSet
def shouldStop(centroids,oldCentroids,maxNum,iteractions):

    if maxNum < iteractions:
        return True
    return np.array_equal(centroids,oldCentroids)


def updateLabels(dataSet,centroids):
    numPoints,numDim = dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1] = getLabelFromClosestCentroids(dataSet[i,:-1],centroids)

def getCentroids(dataSet,k):
    result = np.zeros((k,dataSet.shape[1]))
    print("getCentroids",dataSet)
    for i in range(1,k+1):
        print("i:",i)
        oneCluster = dataSet[dataSet[:,-1] == i,:-1]
        result[i-1,:-1] = np.mean(oneCluster,axis=0)
        result[i-1,-1] = i
    print("result:",result)
    return result

def getLabelFromClosestCentroids(dataSetRow,centroids):
    label = centroids[0,-1]
    # 计算他们之间的距离
    minDist = np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist< minDist:
            minDist = dist
            label = centroids[i,-1]
    print("minDist:",minDist)
    print("label",label)
    return label

x1 = np.array([1,1])
x2 = np.array([2,1])
x3 = np.array([4,3])
x4 = np.array([5,4])
testX = np.vstack((x1,x2,x3,x4))

result = kmeans(testX,2,10)
print("final result")
print(result)

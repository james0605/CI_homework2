import numpy as np
import matplotlib.pyplot as plt

# 取得4維資料
def getTrain4d():
    with open('train4dAll.txt', 'r') as file:
        data = []
        for f in file:
            data.append(list(map(float, f.split())))
    return data

# 取得6維資料
def getTrain6d():
    with open('train6dAll.txt', 'r') as file:
        data = []
        for f in file:
            data.append(list(map(float, f.split())))
    return data

# 高斯函數
def gaussian(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.sum((x-c)**2))

# 兩點距離
def distance(a, b):
    dist = np.sum([c**2 for c in np.subtract(a, b)])
    return dist
    
def kmeans(X, k):

    # 隨機選擇cluster
    clusters_index = np.random.choice(len(X), size=k)
    clusters = [X[c] for c in clusters_index]
    
    prevClusters = clusters.copy()
    stds = np.zeros((k, len(X[0])))
    converged = False

    # 執行kmeans直到所有點距離 < 0.000001
    while not converged:
        distances = np.array([distance(x, c) for x in X  for c in clusters])
        distances = distances.reshape(-1, k)

        closestCluster = np.argmin(distances, axis=1)


        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
                stds[i] = np.std(X[closestCluster == i])
        converged = np.max([distance(c, pre_c) for c, pre_c in zip(clusters, prevClusters)]) < 0.000001
        prevClusters = clusters.copy()

    
    distances = np.array([distance(x, c) for x in X  for c in clusters])
    distances = distances.reshape(-1, k)
    closestCluster = np.argmin(distances, axis=1)
    clustersWithNoPoints = []
    # 找出沒有其他點依附的cluster
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            clustersWithNoPoints.append(i)
        else:
            stds[i] = np.std(X[closestCluster == i])
    
    
    # 如果有沒有點或只有1點的cluster, 取std為平均std
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
    
    # 所有維度共用最大stds
    stds = [np.max(std) for std in stds]
    
    return clusters, stds

   
# RBFN實作，使用虛擬反矩陣一次更新
class RBFNet(object):
    def __init__(self, k=2):
        self.k = k
        self.w = np.random.randn(k + 1)
        self.traindata = np.array(getTrain4d())
        self.X = self.traindata[:, :-1]
        self.Y = self.traindata[:, -1]
        self.centers, self.stds = kmeans(self.X, self.k)
    # def fit(self, X, y):
    #     self.centers, self.stds = kmeans(X, self.k)

    #     qp = []    
    #     for i in range(X.shape[0]):
    #         a = np.array([1] + [gaussian(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
    #         qp.append(a)
    #     qp = np.array(qp)
    #     y = np.array(y)
    #     self.w = ((np.linalg.inv(qp.T.dot(qp))).dot(qp.T)).dot(y)
            
    def fit(self, X, y):
        qp = []    
        for i in range(X.shape[0]):
            a = np.array([1] + [gaussian(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            qp.append(a)
        qp = np.array(qp)
        y = np.array(y)
        print(y)
        self.w = ((np.linalg.inv(qp.T.dot(qp))).dot(qp.T)).dot(y)

    def predict(self, X):
        a = np.array([1] + [gaussian(X, c, s) for c, s, in zip(self.centers, self.stds)])
        # a = np.array([1] + [X])
        y_pred = a.dot(self.w)
        return np.array(y_pred)
    
if __name__ == "__main__":
    rbfn = RBFNet(k=10)
    traindata = np.array(getTrain4d())
    X = traindata[:, :-1]
    Y = traindata[:, -1]
    rbfn.fit(X,Y)
    print(rbfn.predict([9.7355, 10.9379, 18.5740]))

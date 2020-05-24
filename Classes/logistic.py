import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# ------------------------------------ CALCULATE WEIGHT ------------------------------------
# SO TRUNG BINH
def AVG(a):
    return sum(a) / len(a)
# PHUONG SAI
def Variance(a):
    return sum(((a1 - AVG(a))**2) / (len(a)-1) for a1 in a)
# DO LECH CHUAN
def StaDev(a):
    return np.sqrt(Variance(a))
# HIEP PHUONG SAI
def Covariance(x, y):
    return sum(((x[i] - AVG(x)) * (y[i] - AVG(y))) / (len(x)-1) for i in range(0, len(x)))
# HE SO TUONG QUAN 
def CorrCoe(x, y):
    return Covariance(x,y) / np.sqrt(Variance(x) * Variance(y))
# DO DOC QUY HOI (HE SO MOI QUAN HE GIUA X VA Y)
def calWeight(x, y):
    return CorrCoe(x, y) * (StaDev(y) / StaDev(x))

# --------------------------------------- PLOT GRAPH ---------------------------------------
def plot(X, w):
    # IRIS SETOSA
    sepal_length_setosa = []
    sepal_width_setosa  = []
    petal_length_setosa = []
    petal_width_setosa  = []
    # IRIS VERSICOLOR
    sepal_length_versicolor = []
    sepal_width_versicolor  = []
    petal_length_versicolor = []
    petal_width_versicolor  = []
    # IRIS VIRGINICA
    sepal_length_virginica = []
    sepal_width_virginica  = []
    petal_length_virginica = []
    petal_width_virginica  = []

    data = pd.read_csv('Iris/iris.csv', header=None)

    for a in data.values:
        if a[4] == 'Iris-setosa':
            sepal_length_setosa.append(a[0])
            sepal_width_setosa.append(a[1])
            petal_length_setosa.append(a[2])
            petal_width_setosa.append(a[3])   
        elif a[4] == 'Iris-versicolor':
            sepal_length_versicolor.append(a[0])
            sepal_width_versicolor.append(a[1])
            petal_length_versicolor.append(a[2])
            petal_width_versicolor.append(a[3])
        else:
            sepal_length_virginica.append(a[0])
            sepal_width_virginica.append(a[1])
            petal_length_virginica.append(a[2])
            petal_width_virginica.append(a[3])
    '''
    ACREAGE 
    '''
    # x_setosa     = []
    # y_setosa     = []
    # x_versicolor = []
    # y_versicolor = []
    # x_virginica  = []
    # y_virginica  = []
    # for i in range(0, len(sepal_length_setosa)):
    #     x_setosa.append(sepal_length_setosa[i]*sepal_width_setosa[i])
    #     y_setosa.append(petal_length_setosa[i]*petal_width_setosa[i])

    #     x_versicolor.append(sepal_length_versicolor[i]*sepal_width_versicolor[i])
    #     y_versicolor.append(petal_length_versicolor[i]*petal_width_versicolor[i])

    #     x_virginica.append(sepal_length_virginica[i]*sepal_width_virginica[i])
    #     y_virginica.append(petal_length_virginica[i]*petal_width_virginica[i])

    # plt.scatter(x_setosa    , y_setosa    , marker='o', c='r')
    # plt.scatter(x_versicolor, y_versicolor, marker='s', c='b')
    # plt.scatter(x_virginica , y_virginica , marker='^', c='y')
    # plt.title('Acreage')
    # plt.xlabel('Sepal acreage')
    # plt.ylabel('Petal acreage')

    # # PLOT DECISION BOUNDARY
    # x1 = np.linspace(start=10, stop=30, num=50)
    # x2 = -(w[4] + w[0]*x1)/w[1]
    # plt.plot(x1,x2)
    # plt.show()

    '''
    SEPAL
    '''
    plt.scatter(sepal_length_setosa    , sepal_width_setosa    , marker='o', c='r')
    plt.scatter(sepal_length_versicolor, sepal_width_versicolor, marker='s', c='b')
    plt.scatter(sepal_length_virginica , sepal_width_virginica , marker='^', c='y')
    plt.title('Sepal')
    plt.xlabel('Length')
    plt.ylabel('Width')

    # PLOT DECISION BOUNDARY
    x1 = np.linspace(start=4.5, stop=8, num=50)
    x3 = np.linspace(start=1, stop=7, num=50)
    x4 = np.linspace(start=0, stop=2.5, num=50)
    x2 = -(w[4] + w[0]*x1 + w[2]*x3 + w[3]*x4)/w[1]
    plt.plot(x1,x2)
    plt.show()

    '''
    PETAL
    '''
    # plt.scatter(petal_length_setosa    , petal_width_setosa    , marker='o', c='r')
    # plt.scatter(petal_length_versicolor, petal_width_versicolor, marker='s', c='b')
    # plt.scatter(petal_length_virginica , petal_width_virginica , marker='^', c='y')
    # plt.title('Petal')
    # plt.xlabel('Length')
    # plt.ylabel('Width')

    # # PLOT DECISION BOUNDARY
    # x1 = np.linspace(start=4, stop=8, num=50)
    # x2 = np.linspace(start=2, stop=5, num=50)
    # x3 = np.linspace(start=1, stop=7, num=50)
    # x4 = -(w[4] + w[0]*x1 + w[1]*x2 + w[2]*x3)/w[3]
    # plt.plot(x3,x4)
    # plt.show()
# ------------------------------------------------------------------------------------------

# ASSIGN ATTRIBUTES AND LABELS (X & Y)
def assgin(flower):
    data = pd.read_csv('Iris/iris.csv', header=None)
    y = []
    x = []
    
    # FLOWER INPUT WILL BE ASSIGNED AS 1, THE REST WILL BE ASSIGNED AS 0
    for i in range(0, len(data)):
        arr = [data[0][i], data[1][i], data[2][i], data[3][i]]
        x.append(arr)
        if data[4][i] == flower:
            y.append([1])
        else:
            y.append([0])

    attributes = np.array(x)
    labels = np.array(y)

    return attributes, labels

class LogisticRegression:
    def __init__(self, flower):
        self.X, self.y = assgin(flower)

        # APPEND BIAS
        bias = np.ones((len(self.y), 1))
        self.X = np.append(self.X, bias, axis=1)

        self.w = np.ones([1, self.X.shape[1]]).T
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    # CALCULATE PROBABILITYS
    def h(self):
        z = np.dot(self.X, self.w)
        return self.sigmoid(z)

    def costFunction(self):
        # IF Y = 1
        cost1 = self.y * np.log(self.h())
        # IF Y = 0
        cost2 = (1 - self.y) * np.log(1 - self.h())

        cost = -(cost1 + cost2)
        return cost.sum()/len(self.y)

    def gradientDescent(self):
        gd = np.dot(self.X.T, self.h() - self.y)
        return gd/len(self.y)

    def updateWeight(self, learningRate):
        gd = self.gradientDescent()
        self.w -= gd*learningRate

    def train(self, epoch, learningRate):
        for _ in range(epoch):
            oldLoss = self.costFunction()
            self.updateWeight(learningRate)
            newLoss = self.costFunction()
            
            if round(newLoss, 4) == round(oldLoss, 4):
                # print('Epoch:',i+1)
                # print('Loss:',newLoss)
                break
                
    def accuracy(self):
        res = 0
        h = self.h()
        for i in range(0, len(h)):
            if int(h[i] + 0.5) == self.y[i]:
                res += 1
        return round(res/len(self.y), 4) * 100

    def plotGraph(self):
        plot(self.X, self.w)

import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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

class NeuralNetwork:
    def __init__(self, layers,flower):
        self.x, self.y = assgin(flower)
        self.layers = layers 
        self.W = []
        self.b = []

        # GENERATE WEIGHT AND BIAS
        for i in range(0, len(self.layers)-1):
            w_ = np.random.randn(self.layers[i], self.layers[i+1])
            b_ = np.ones((self.layers[i+1], 1))
            self.W.append(w_/self.layers[i])
            self.b.append(b_)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def feedFoward(self, preLayer, i):
        return self.sigmoid(np.dot(preLayer, self.W[i]) + (self.b[i].T))

    def derivative(self,x):
        return x*(1-x)

    # LOSS FUNCTION
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict))) 

    def train(self, X, y, lr):
        data = [X]

        # FEEDFORWARD
        oldLayer = data[0]
        for i in range(0, len(self.layers) - 1):
            newLayer = self.feedFoward(oldLayer, i)
            data.append(newLayer)
            oldLayer = newLayer

        # BACKPROPAGATION 
        dL = [-(y/data[-1] - (1-y)/(1-data[-1]))] #Đạo Hàm Loss
        dW = []
        db = []

        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot((data[i]).T, dL[-1] * self.derivative(data[i+1]))
            db_ = (np.sum(dL[-1] * self.derivative(data[i+1]), 0)).reshape(-1,1)
            dL_ = np.dot(dL[-1] * self.derivative(data[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dL.append(dL_)
        
        # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]
        
		# Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - lr * dW[i]
            self.b[i] = self.b[i] - lr * db[i]
    
    def start(self, epochs, lr):
        for _ in range(epochs):
            self.train(self.x, self.y, lr)          
     

    # PREDICT OUTPUT
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = self.feedFoward(X, i)
        return X

    def accuracy(self):
        res = []
        for i in range(len(self.predict(self.x))):
            if self.predict(self.x)[i] >= 0.5:
                res.append([1])
            else:
                res.append([0])
        res = np.array(res)
        return round(accuracy_score(res,self.y)*100, 3)

    def accuracy2(self):
        res = 0
        for i in range(len(self.predict(self.x))):
            if int(self.predict(self.x)[i] + 0.5) == self.y[i]:
                res += 1
        return round(res/len(self.y), 4) * 100





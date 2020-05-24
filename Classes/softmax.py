import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ASSIGN ATTRIBUTES AND LABELS (X & Y)
def assgin():
    data = pd.read_csv('Iris/iris.csv', header=None)
    y = []
    x = []
    for i in range(0, len(data)):
        arr = [data[0][i], data[1][i], data[2][i], data[3][i]]
        x.append(arr)
        if data[4][i] == 'Iris-setosa':
            y.append([0])
        elif data[4][i] == 'Iris-versicolor':
            y.append([1])
        else:
            y.append([2])

    attributes = np.array(x)
    labels = np.array(y)

    return attributes, labels

class Softmax:
    def __init__(self):
        self.X, self.y = assgin()

        # APPEND BIAS
        bias = np.ones((len(self.y), 1))
        self.X = np.append(self.X, bias, axis=1)

        # W SIZE: NUMBER OF ATTRIBUTES x NUMBER OF UNIQUE LABELS
        self.w = np.ones([self.X.shape[1], len(np.unique(self.y))])

    # USE SOFTMAX TO CALCULATE PROBABILITIES
    def softmax(self):
        z = np.dot(self.X, self.w)
        probs = np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)
        return probs

    def predict(self):
        predict = np.argmax(self.softmax(),axis=1)
        return predict

    # CONVERT UNIDIMENSIONAL ARRAY OF LABELS INTO ON-HOT VARIENT (SIZE: LEN(Y) x NUMBER OF LABELS)
    def oneHot(self):
        uniqueY = list(np.unique(self.y))
        oneHot_y = np.zeros((len(self.y), len(uniqueY)))
        for cnt, ele in enumerate(self.y):
            oneHot_y[cnt][uniqueY.index(ele)] = 1
        return oneHot_y

    def costFunction(self, oneHot_y):
        probs = self.softmax()
        loss = -np.sum(oneHot_y * np.log(probs))
        return loss / len(self.y)

    def gradientDescent(self, oneHot_y):
        probs = self.softmax()
        return np.dot(self.X.T, probs - oneHot_y)
        
    def updateWeight(self, oneHot_y, learningRate):
        gd = self.gradientDescent(oneHot_y)
        self.w -= gd*learningRate

    def train(self, epoch, learningRate):
        for _ in range(epoch):
            oldLoss = self.costFunction(self.oneHot())
            self.updateWeight(self.oneHot(), learningRate)
            newLoss = self.costFunction(self.oneHot())   

            # print(i+1)
            # print('new loss:',newLoss)
            # print('----------------------------')

            if round(newLoss, 4) == round(oldLoss, 4):
                break

    def accuracy(self):
        res = 0
        for i in range(0, len(self.predict())):
            if self.predict()[i] == self.y[i]:
                res += 1
        return round(res/len(self.y), 5) * 100
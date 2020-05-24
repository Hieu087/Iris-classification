import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classes.logistic import LogisticRegression
from Classes.softmax  import Softmax
from Classes.neuralnetwork import NeuralNetwork
from Classes.naivesbayes import NaivesBayes

'''
Iris : Họ Diên vĩ
Sepal: Lá đài
Petal: Cánh hoa
'''

def main():
    # ------------------------ LOGISTIC REGRESSION ------------------------
    '''
    Iris-setosa
    Iris-versicolor
    Iris-virginica
    '''
    test = LogisticRegression('Iris-setosa')

    print('\n#######################')
    print('| Logistic Regression |')
    print('#######################\n')    

    print('----- Before train -----')
    print('  Accuracy:',test.accuracy(), '%')

    test.train(1000, 0.01)

    print('------ After train -----')
    print('  Accuracy:',test.accuracy(), '%\n')

    # test.plotGraph()

    # ------------------------ SOFTMAX REGRESSION -------------------------
    test = Softmax()
    
    print('\n######################')
    print('| Softmax Regression |')
    print('######################\n')

    print('----- Before train -----')
    print('Accuracy:', test.accuracy(), '%')

    test.train(1000, 0.01)

    print('------ After train -----')
    print('Accuracy:', test.accuracy(), '%\n')

    # -------------------------- NEUTRAL NETWORK ---------------------------
    test = NeuralNetwork([4, 5, 5, 1],'Iris-virginica')

    print('\n###################')
    print('| Neutral Network |')
    print('###################\n')

    print('----- Before train -----')
    print('Accuracy:', test.accuracy(), '%')

    test.start(1000, 0.01)
    
    print('------ After train -----')
    print('Accuracy:', test.accuracy(), '%\n')

    # ---------------------------- NAIVES BAYES -----------------------------
    test = NaivesBayes('Iris-virginica')

    print('\n##########################')
    print('| Bernoulli Naives Bayes |')
    print('##########################\n')

    test.start()
    
    print('------ After train -----')
    print('Accuracy:', test.accuracy(), '%\n')
    # ------------------------------------------------------------------------

main()


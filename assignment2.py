from random import choice
import numpy as np
from numpy import array, dot, random


class Perecptron():
    def __init__(self, x_train, lr):
        self.W = np.zeros(len(x_train[0])+1)
        self.lr = lr
    
    def fit(self, x_train,y_train):
        for _ in range(300):
            for x, y in zip(x_train, y_train):
                update = self.lr * (y - self.predict(x))
                self.W[1:] += update * x
                self.W[0] += update
            return self.W
    
    def dot_product(self, X):
        return (np.dot(X, self.W[1:]) + self.W[0])

    def predict(self, X):
        return np.where(self.dot_product(X) >= 0.0, 1, 0)



def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def cleanFile(fileLines):
    temp1 = []
    for line in fileLines:
        temp2 = []
        for element in line:
            if element == ',':
                continue
            if element == '\n' :
                continue
            if element == ' ' :
                continue
            temp2.append(int(element))
        temp1.append(temp2)
    
    temp3 = []
    temp4 = []

    for i in temp1:
        temp3.append(np.array(i[:len(i)-1]))
        temp4.append(i[len(i)-1])
    return temp3, temp4

#main
file = open('train.txt', 'r') 
fileLinesTrain = file.readlines()
x_train, y_train = cleanFile(fileLinesTrain)

x_train = np.array(x_train)
y_train = np.array(y_train)

file2 = open('SPECT.test.txt', 'r') 
fileLinesTest = file2.readlines()
x_test, y_test= cleanFile(fileLinesTest)

x_test = np.array(x_test)
y_test = np.array(y_test)


test = Perecptron(x_train,0.1)
test.fit(x_train, y_train)
predicted = []
for i in range(len(x_test)):
    p = test.predict(x_test[i])
    predicted.append(p.tolist())

print("Expected Outcome",'->',"Prediction by model")
for i,j in zip(y_test,predicted):
    print (i,'->',j)
    
print('\n')
print('Accuaracy of the prediction:')
accuracy_metric( y_test, predicted)

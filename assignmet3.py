import random
from torch.autograd import Variable
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1234)
import numpy as np
from sklearn.metrics import accuracy_score


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    
#get data
data = pd.read_csv('iris.csv')
data.loc[data['class']=='Iris-setosa','class']=0
data.loc[data['class']=='Iris-versicolor','class']=1
data.loc[data['class']=='Iris-virginica','class']=2
data = data.apply(pd.to_numeric)

#80% train data 20% test data
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]

train = train.sample(frac=1)
test = test.sample(frac=1)

#change dataframe to array
train_array = train.as_matrix()

#split x and y (feature and target)
X_train = train_array[:,:4]
Y_train = train_array[:,4]



X_train = Variable(torch.Tensor(X_train).float())
Y_train = Variable(torch.Tensor(Y_train).long(),requires_grad=False)


# Construct our model by instantiating the class defined above
model = TwoLayerNet(4, 100, 3)


#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
for t in range(500):
    y_pred = model(X_train)

    # Compute and print loss
    loss = criterion(y_pred, Y_train)
    #print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#test part
#change dataframe to array
test_array = test.as_matrix()

#split x and y (feature and target)
X_test = test_array[:,:4]
Y_test = test_array[:,4]


#get prediction
X = Variable(torch.Tensor(X_test).float())
Y = torch.Tensor(Y_test).long()
y_pred = model(X)
_, predicted = torch.max(y_pred.data, 1)

#get accuration
print('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / len(predicted)))

print("Expected Outcome",'->',"Prediction by model")
for i,j in zip(Y,predicted):
    print (i,'->',j)

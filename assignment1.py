# Load libraries
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

url = "iris.txt"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.30
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

logistic = LogisticRegression()
logistic.fit(X_train,Y_train)
prediction = logistic.predict(X_test)

print("Expected Outcome",'->',"Prediction by model")
for i,j in zip(Y_test,prediction):
    print (i,'->',j)

#accuaracy of the prediction 
print('\n')
print('Accuaracy of the prediction:')
print(accuracy_score(prediction, Y_test))

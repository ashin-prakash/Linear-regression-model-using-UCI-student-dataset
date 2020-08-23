#you need to install the packages 
#pip install pandas
#pip install numpy
#pip install sklearn

#import the libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


#adding the data
data = pd.read_csv("student-mat.csv", sep=";")

#print head
#print(data.head(10))

#setting the attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#print to see changes
#print(data.head(10))

predict = "G3"

#distinguishing the teat and train data
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

""""
#training initiation
best = 0
for _ in range(1000):
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

  #training the model using linear regression
  linear = sklearn.linear_model.LinearRegression()

  linear.fit(x_train, y_train)
  acc = linear.score(x_test, y_test)
  print(acc*100, "%")

  if acc > best:
    best = acc
    with open("studentmodel.pickle", "wb") as f:
      pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#printing the coef  and intercept
print("Co: \n", linear.coef_)
print("intercept: \n", linear.intercept_)

#predicting the model wit test data
predictions = linear.predict(x_test)

#using the model for real time prediction
for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


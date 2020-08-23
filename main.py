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

#training initiation
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#training the model using linear regression
linear = sklearn.linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc*100, "%")

#printing the coef  and intercept
print("Co: \n", linear.coef_)
print("intercept: \n", linear.intercept_)

#predicting the model wit test data
predictions = linear.predict(x_test)

#using the model for real time prediction
for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

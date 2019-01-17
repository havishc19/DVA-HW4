## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=100, shuffle=True)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

linRegr = LinearRegression()
linRegr.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX

train_predict = linRegr.predict(x_train)
test_predict = linRegr.predict(x_test)
print("Linear Regression")
print(accuracy_score(y_train, train_predict.round()))
print(accuracy_score(y_test, test_predict.round()))
print("\n")

# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX

MLPC = MLPClassifier()
MLPC.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

train_predict = MLPC.predict(x_train)
test_predict = MLPC.predict(x_test)
print("MLPClassifier")
print(accuracy_score(y_train, train_predict.round()))
print(accuracy_score(y_test, test_predict.round()))
print("\n")


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

train_predict = rf.predict(x_train)
test_predict = rf.predict(x_test)
print("RandomForestClassifier")
print(accuracy_score(y_train, train_predict.round()))
print(accuracy_score(y_test, test_predict.round()))
print("\n")

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators': [10,15,20], 'max_depth': [10,15,20]}
clf = GridSearchCV(rf, parameters, cv=10)
clf.fit(x_train, y_train)
print("Best Param & Score for Random Forest")
print(clf.best_params_)
print(clf.best_score_)
print("\n")
results = clf.cv_results_

for i in results:
    print(i, results[i])

# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX



print("SVC---------------------------")
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

train_predict = svm.predict(x_train)
test_predict = svm.predict(x_test)
print(accuracy_score(y_train, train_predict.round()))
print(accuracy_score(y_test, test_predict.round()))
print("\n ")

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.1, 1]}
clf = GridSearchCV(svm, parameters, cv=10)
clf.fit(x_train, y_train)
print("Best Param & Score for SVC")
print(clf.best_params_)
print(clf.best_score_)
print("\n")
results = clf.cv_results_

for i in results:
    print(i, results[i])



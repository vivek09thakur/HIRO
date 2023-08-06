import csv
import re
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.svm import SVC

training = pandas.read_csv('dataset/data/Training.csv')
testing = pandas.read_csv('dataset/data/Testing.csv')

cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print (scores.mean())
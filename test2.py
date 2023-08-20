import re
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

class hiro:
    
    def __init__(self):
        self.training = pd.read_csv("dataset/data/Training.csv")
        self.testing = pd.read_csv("dataset/data/Testing.csv")
        self.cols = self.training.columns
        self.cols = self.cols[:-1]
        self.x = self.training[self.cols]
        self.y = self.training["prognosis"]
        self.y1 = self.y
        self.reduced_data = self.training.groupby(self.training["prognosis"]).max()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.y)
        self.y = self.le.transform(self.y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.33, random_state=42
        )
        self.testx = self.testing[self.cols]
        self.testy = self.testing["prognosis"]
        self.testy = self.le.transform(self.testy)
        self.clf1 = DecisionTreeClassifier()
        self.clf = self.clf1.fit(self.x_train, self.y_train)
        self.scores = cross_val_score(self.clf, self.x_test, self.y_test, cv=3)
        self.model = SVC()
        self.model.fit(self.x_train, self.y_train)
        self.importances = self.clf.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]
        self.features = self.cols
        self.severityDictionary = dict()
        self.description_list = dict()
        self.precautionDictionary = dict()
        self.symptoms_dict = {}
        for index, symptom in enumerate(self.x):
            self.symptoms_dict[symptom] = index
        pass
    
    def calcCondition(self,exp,days):
        sum = 0
        for item in exp:
            sum = sum + self.symptoms_dict[item]
        sum = sum/len(exp)
        sum = sum + days
        return sum
    
    def getDescription():
        global description_list
        with open("dataset/data/symptom_Description.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                _name, _desc = row
                description_list[_name] = _desc
                
    def getServersity():
        global severityDictionary
        with open("dataset/data/symptom_severity.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                _diction  = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
            
    def getPrecaution():
        global precautionDictionary
        with open("dataset/data/symptom_precaution.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                _name, *precaution = row
                precautionDictionary[_name] = precaution
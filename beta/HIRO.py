# import csv
# import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, _tree


class HIRO:
    
    def __init__(self,
                 training_data_file,
                 testing_data_file,
                 serverity_dataset,
                 precaution_dataset,
                 description_dataset
                ):
        
        # training and test datasets
        self.training_data_file = training_data_file
        self.testing_data_file = testing_data_file
        
        # precautions , description and serverity dataset
        self.serverity_dataset = serverity_dataset
        self.precaution_dataset = precaution_dataset
        self.description_dataset = description_dataset
        
        # read training and test datasets
        self.training = pd.read_csv(self.training_data_file)
        self.testing = pd.read_csv(self.testing_data_file)
        
        # preprocessing training and test datasets
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
        # 
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
            
        self.description_list = {}
        self.severityDictionary = {}
        self.precautionDictionary = {}
        
    
    def read_csv(self,csv_file):
        with open(csv_file) as f:
            readed_date = pd.read_csv(f)
        return readed_date
    
    def get_choice(self,inp,error_message):
        try:
            return int(inp)
        except Exception as e:
            print(error_message)
        
    def calcCondition(self,exp,days):
        sum = 0
        for item in exp:
            sum = sum + self.symptoms_dict[item]
        sum = sum/len(exp)
        sum = sum + days
        return sum
    
    def getDescription(self):
        csv_reader = self.read_csv(self.description_dataset)
        for row in csv_reader:
            description = {row[0]: row[1]}
            self.description_list.update(description)
            break
        return self.description_list
                
    def getServersity(self):
        csv_reader = self.read_csv(self.serverity_dataset)
        for row in csv_reader:
            try:
                diction  = {row[0]: int(row[1])}
                self.severityDictionary.update(diction)
            except Exception:
                pass
            break
        return self.severityDictionary
            
    def getPrecaution(self):
        csv_reader = self.read_csv(self.precaution_dataset)
        for row in csv_reader:
            prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            self.precautionDictionary.update(prec)
            break
        return self.precautionDictionary
    
    def introduce(self,patient_name):
        print('Hello , I am HIRO , your own healthcare companion.I am here to make you fit and fine ^_^')
        small_talks = [
            f'So hello {patient_name} , let\'s start with your problem'
        ]
        print(small_talks[0])
    
    def prepare(self):
        warnings.filterwarnings("ignore", category=UserWarning) 
        self.getServersity()
        self.getDescription()
        self.getPrecaution()
        
    def match_patterns(self,dis_list,inp):
        prediction_list = []
        inp = str(inp).replace(" ","_")
        pattern = f'{inp}'
        regular_expression = re.compile(pattern)
        prediction_list = [item for item in dis_list if regular_expression.search(item)]
        if len(prediction_list)>0:
            return 1 , prediction_list
        else:
            return 0,[]
        
    def get_user_problem(self,user_problem):
        tree = self.clf
        feature_name = self.cols
        tree_ = tree.tree_
        feature_name = [
            feature_name[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
            for i in tree_.feature
        ]
        chk_dis = ','.join(feature_name).split(',')
        symtoms_present = []
        confidence , cnf_dis = self.match_patterns(chk_dis,user_problem)
        
        return confidence,cnf_dis,symtoms_present
    
    
    def second_prediction(self,symptoms_exp):
        dataframe = self.training
        X = dataframe.iloc[:,:-1]
        y = dataframe['prognosis']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=20)
        rf_clf = self.clf1
        rf_clf.fit(X_train,y_train)
        
        symptoms_dictionary = {symptom:index for index,symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dictionary))
        for item in symptoms_exp:
            input_vector[[symptoms_dictionary[item]]] =1
        
        return rf_clf.predict([input_vector])
    
    def daignose_diseases(self,node):
        node = node[0]
        val = node.nonzero()
        disease = self.le.inverse_transform(val[0])
        return list(map(lambda x:x.strip(),list(disease)))
    
    def recurse(self, node, depth, disease_input, symptoms_present):
        tree_ = self.clf.tree_
        present_diseases = []
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_index = tree_.feature[node]
            threshold = tree_.threshold[node]
            if feature_index == self.symptoms_dict[disease_input]:
                val = 1
            else:
                val = 0
            if val <= threshold:
                self.recurse(tree_.children_left[node], depth + 1, disease_input, symptoms_present)
            else:
                symptoms_present.append(self.cols[feature_index])
                self.recurse(tree_.children_right[node], depth + 1, disease_input, symptoms_present)
        else:
            present_diseases = self.daignose_diseases(tree_.value[node])
            # print(present_diseases)
            symptoms_given = self.reduced_data.columns[self.reduced_data.loc[present_diseases].values[0].nonzero()]
            symptoms_present.extend(symptoms_given)
            
        return symptoms_present,present_diseases
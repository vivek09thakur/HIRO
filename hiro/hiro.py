import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC
import csv


class HIRO:
    
    def __init__(self,training_data_file,testing_data_file):
        
        # training and test datasets
        self.training_data_file = training_data_file
        self.testing_data_file = testing_data_file
        
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
        self.present_diseases = None
    
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
        if sum/len(exp) >= 13:
            return sum , 'You should take the consultation from doctor'
        return sum , 'It might not be that bad but you should take precautions'
    
    def getDescription(self,description_dataset):
        with open(description_dataset) as f:
            csv_reader = csv.reader(f,delimiter=',')
            for row in csv_reader:
                description = {row[0]: row[1]}
                self.description_list.update(description)
                break
            return self.description_list
                
    def getServersity(self,serverity_dataset):
        with open(serverity_dataset) as f:
            csv_reader = csv.reader(f,delimiter=',')
            for row in csv_reader:
                try:
                    diction  = {row[0]: int(row[1])}
                    self.severityDictionary.update(diction)
                except Exception as e:
                    print(e)
                    pass
                break
            return self.severityDictionary
            
    def getPrecaution(self,precaution_dataset):
        with open(precaution_dataset) as f:
            csv_reader = csv.reader(f,delimiter=',')
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
    
    def predict_disease(self, node, depth, disease_input, symptoms_present):
        tree_ = self.clf.tree_
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
            symptoms_present.extend(present_diseases)
        return symptoms_present
    
    def recurse(self, node, depth, disease_input, symptoms_present):
        tree_ = self.clf.tree_
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
            self.present_diseases = self.daignose_diseases(tree_.value[node])
            symptoms_given = self.reduced_data.columns[self.reduced_data.loc[self.present_diseases].values[0].nonzero()]
            list(symptoms_present).append(symptoms_given)
            
        return symptoms_present,self.present_diseases
    
    def show_first_prediction(self,disease,discrption_list):
        print(f'You may have {disease}')
        print(f'DISEASE DESCRIPTION -- {discrption_list[disease]}')
        
    def show_second_prediction(self,disease1,disease2,discrption_list):
        print(f'You may have {disease1} or {disease2}')
        print(f'''
            DISEASE DESCRIPTION [1] -- {discrption_list[disease1]}
            DISEASE DESCRIPTION [2] -- {discrption_list[disease2]}
        ''')
    
    def show_patient_condition(self,condition):
        print(f'PATIENT CONDITION -- {condition}')
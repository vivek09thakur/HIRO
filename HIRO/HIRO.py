import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from HIRO.support import chat_support, support
import csv

class HEALTHCARE_COMPANION:
    
    def __init__(self,training_data_path,test_data_path,description_data,precaution_data,chat_log_path):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.description_data = description_data
        self.precaution_data = precaution_data
        self.chat_log_path = chat_log_path
        self.data = pd.read_csv(self.training_data_path)
        self.test_data = pd.read_csv(self.training_data_path)
        self.disease_list = self.data['prognosis'].unique()
        self.supportive_module = support(self.test_data_path)
        self.chat_support = chat_support(self.chat_log_path)
        
    
    def preprocess(self):
        self.encoder = LabelEncoder()
        self.data = self.data.dropna(axis=1)
        self.data['prognosis'] = self.encoder.fit_transform(self.data['prognosis'])
        self.X = self.data.iloc[:,:-1]
        self.Y = self.data.iloc[:,-1]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state=24)
        
    def train(self,show_accuracy=False):
        def vc_scoring(estimator,X,Y):
            return accuracy_score(Y,estimator.predict(X))
        models = {
            'SVC': SVC(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier()
        }
        
        for model_name in models:
            model = models[model_name]
            scores = cross_val_score(model,self.X,self.Y,cv=10,n_jobs=-1,scoring=vc_scoring)
            
            if show_accuracy==True:
                self.type_text(f'(model loaded) MODEL NAME => {model_name} \\ Accuracy => {scores.mean()}\n')
                
    
    def build_model(self):
        # TRAINING AND TESTING SVC MODEL
        self.svc_model = SVC()
        self.svc_model.fit(self.X_train,self.Y_train)
        self.svc_pred = self.svc_model.predict(self.X_test)
        
        # TRAINING AND TESTING NAIVE BAYES MODEL
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.X_train,self.Y_train)
        self.svc_pred = self.nb_model.predict(self.X_test)
        
        # TRAINING AND TESTING RANDOM FOREST MODEL
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X_train,self.Y_train)
        self.svc_pred = self.rf_model.predict(self.X_test)
        
        # return self.svc_model,self.nb_model,self.rf_model
        
    def combine_model(self):
        # TRAINING THE MODELS ON THE ENTIRE DATASET
        self.svc_model.fit(self.X,self.Y)
        self.nb_model.fit(self.X,self.Y)
        self.rf_model.fit(self.X,self.Y)
        # Read the test data
        self.test_data = pd.read_csv(self.test_data_path).dropna(axis=1)
        self.test_X = self.test_data.iloc[:,:-1]
        self.test_Y = self.encoder.transform(self.test_data.iloc[:,-1])
        # predict the test data
        self.svc_pred = self.svc_model.predict(self.test_X)
        self.nb_pred = self.nb_model.predict(self.test_X)
        self.rf_pred = self.rf_model.predict(self.test_X)
        
        self.final_pred = [mode([self.svc_pred[i],self.nb_pred[i],self.rf_pred[i]])[0][0] for i in range(len(self.svc_pred))]
        
        return self.final_pred
    
    def collect_symptoms_data(self):
        try:
            self.symptoms = self.X.columns.values
            self.symptoms_index = {}
            for index,symtom in enumerate(self.symptoms):
                self.symptoms = ' '.join([i.capitalize() for i in symtom.split('_')])
                self.symptoms_index[self.symptoms] = index

            self.data_dict = {
                'symptoms_index': self.symptoms_index,
                'prediction_class': self.encoder.classes_
            }
        except Exception as e:
            print('ERROR OCCURED WHILE COLLECTING SYMPTOMS DATA')
            
    # SUPPORTIVE FUNCTIONS
    def extract_symptoms(self,sentence):
        return self.supportive_module.extract_symptoms(sentence)
        
    def talk_to_user(self,user_input):
        return self.chat_support.get_response(user_input)
    
    def predict_disease_from_symptoms(self,user_input):
        try:
            symptoms = user_input.split(',')
            input_data = [0]*len(self.data_dict['symptoms_index'])
            for symtom in symptoms:
                index = self.data_dict['symptoms_index'][symtom]
                input_data[index] = 1

            input_data = np.array(input_data).reshape(1,-1)

            # Generate the prediction individually
            rf_pred = self.data_dict['prediction_class'][self.rf_model.predict(input_data)[0]]
            svc_pred = self.data_dict['prediction_class'][self.svc_model.predict(input_data)[0]]
            nb_pred = self.data_dict['prediction_class'][self.nb_model.predict(input_data)[0]]

            final_pred = mode([rf_pred,svc_pred,nb_pred])[0][0]

            all_predictions = {
                'Random Forest': rf_pred,
                'SVC': svc_pred,
                'Naive Bayes': nb_pred,
                'Final Prediction': final_pred
            }

            return all_predictions
        except Exception as e:
            print('ERROR OCCURED WHILE PREDICTING DISEASE FROM SYMPTOMS\n ERROR =>{}'.format(e))
            
    def get_description(self,disease):
        disease_description_dict = {}
        with open(self.description_data, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                disease_description_dict[row[0]] = row[1]
        try:
            return disease_description_dict[disease]
        except Exception as e:
            return 'Sorry I could not find the description of the disease'
        
    def get_precautions(self,disease):
        precautions_dict = {}
        with open(self.precaution_data, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                precautions_dict[row[0]] = row[1]
        try:
            return precautions_dict[disease]
        except Exception as e:
            return 'Sorry I could not find the precautions of the disease'
        
    
    def process_training_data(self,show_accuracy=False):
        self.preprocess()
        self.train(show_accuracy=show_accuracy)
        
        
    def build_robust_model(self):
        self.build_model()
        self.combine_model()
        self.collect_symptoms_data()
        
    def introduce(self,patient_name):
        self.type_text(f'\nHello {patient_name}, I am HIRO, your healthcare chatbot. I can help you diagnose your disease based on your symptoms.')
    
    
    def show_diseases(self,disease_dictionary,show_description=False,show_precautions=False):
        self.type_text('\nOkay just wait for a second!, Let me analyze your symptoms :)')
        self.type_text(f'\nTEST 1 => {disease_dictionary["Random Forest"]}')
        self.type_text(f'\nTEST 2 => {disease_dictionary["SVC"]}')
        self.type_text(f'\nTEST 3 => {disease_dictionary["Naive Bayes"]}')
        self.type_text(f'\nAfter examining everything I found that you might have : {disease_dictionary["Final Prediction"]}')
        
        if show_description == True:
            disease_description = self.get_description(disease_dictionary['Final Prediction'])
            if disease_description != None:
                self.type_text(f'\n\nDisease Description : {disease_description}')
            else:
                self.type_text('\n\nSorry I could not find the description of the disease')
                
        if show_precautions == True:
            disease_precautions = self.get_precautions(disease_dictionary['Final Prediction'])
            if disease_precautions != None:
                self.type_text(f'\n\nDisease Precautions : {disease_precautions}')
            else:
                self.type_text('\n\nSorry I could not find the precautions of the disease')
        
            
    def type_text(self,text):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.004)
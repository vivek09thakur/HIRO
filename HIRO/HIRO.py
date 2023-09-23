import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class HEALTHCARE_CHATBOT:
    
    def __init__(self,training_data_path,test_data_path):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.data = pd.read_csv(self.training_data_path)
        self.test_data = pd.read_csv(self.test_data_path)
        self.disease_list = self.data['prognosis'].unique()
        
    
    def preprocess(self):
        self.encoder = LabelEncoder()
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
                print(f'{model_name} \nAccuracy: {scores.mean()}')
                
    
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
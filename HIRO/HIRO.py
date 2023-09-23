import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class HEALTHCARE_CHATBOT:
    
    def __init__(self,training_data_path,test_data_path):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.data = pd.read_csv(self.training_data_path)
        self.test_data = pd.read_csv(self.test_data_path)
        self.disease_list = self.data['prognosis'].unique()
        
    
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
                print(f'MODEL => {model_name} \nAccuracy => {scores.mean()}')
                
    
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
        self.symptoms = self.X.columns.values
        self.symptoms_index = {}
        for index,symtom in enumerate(self.symptoms):
            self.symptoms = ' '.join([i.capitalize() for i in symtom.split('_')])
            self.symptoms_index[self.symptoms] = index
            
        self.data_dict = {
            'symptoms_index': self.symptoms_index,
            'prediction_class': self.encoder.classes_
        }
        
    def predict_disease_from_symptoms(self,user_input):
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
    
    
    def process_training_data(self,show_accuracy=False):
        self.preprocess()
        self.train(show_accuracy=show_accuracy)
        
    def build_robust_model(self):
        self.build_model()
        self.combine_model()
        self.collect_symptoms_data()
        
    def introduce(self,patient_name):
        print(f'\nHello {patient_name}, I am HIRO, your healthcare chatbot. I can help you diagnose your disease based on your symptoms.')
    
    def show_diseases(self,disease_dictionary):
        print('\nOkay just wait for a second!, Let me analyze your symptoms :)')
        
        print(f'TEST 1 => {disease_dictionary["Random Forest"]}')
        print(f'TEST 2 => {disease_dictionary["SVC"]}')
        print(f'TEST 3 => {disease_dictionary["Naive Bayes"]}')
        print(f'\nAfter examining everything I found that you might have : {disease_dictionary["Final Prediction"]}')
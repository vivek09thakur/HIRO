import json
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class support:
    
    def __init__(self,csv_file_1):
        self.csv_file_1 = csv_file_1
        self.symptoms_list = []
        with open(self.csv_file_1, 'r') as f:
            self.reader = pd.read_csv(f)
            for row in self.reader:
                self.symptoms_list.append(row)
        
    def normalize_symptoms_list(self):
        self.normalized_symptoms_list = []
        # Normalize the symptoms list : skin_rash -> skin rash
        for i in self.symptoms_list:
            self.normalized_symptoms_list.append(i.replace('_',' '))
        return self.normalized_symptoms_list
    
    def extract_symptoms(self,sentence):
        try:
            self.normalize_symptoms_list()
            self.sentence = sentence.lower()
            self.extracted_symptoms = []
            for i in self.normalized_symptoms_list:
                if i in self.sentence:
                    self.extracted_symptoms.append(i)
            extracted_symptoms = ','.join(self.extracted_symptoms)
            self.extracted_symptoms = extracted_symptoms.title()
            
            return self.extracted_symptoms
        except KeyError:
            print('\nERROR OCCURED WHILE EXTRACTING SYMPTOMS => {}'.format(KeyError))
            return self.extracted_symptoms.pop(KeyError)
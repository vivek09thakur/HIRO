import csv
import json
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
        self.normalize_symptoms_list()
        self.sentence = sentence
        self.extracted_symptoms = []
        for i in self.normalized_symptoms_list:
            if i in self.sentence:
                self.extracted_symptoms.append(i)
        return ','.join(self.extracted_symptoms)
  
class chat_support:
    
    def __init__(self,pairs_file):
        self.pairs_file = pairs_file
        with open(self.pairs_file,'r') as f:
            self.pairs = json.load(f)
        self.questions = self.pairs['questions']
        self.answers = self.pairs['answers']
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.questions)
        self.question_vectors = self.vectorizer.transform(self.questions)
        
    def get_response(self,user_input):
        self.user_input = user_input
        self.user_input_vector = self.vectorizer.transform([self.user_input])
        self.similarity_scores = cosine_similarity(self.user_input_vector,self.question_vectors)
        self.most_similar_index = self.similarity_scores.argmax()
        self.response = self.answers[self.most_similar_index]
        return self.response

# Test case 
# if __name__ == '__main__':
#     s = support('Notebook/dataset/Testing.csv')
#     print(f"TEST CASE 1 : {s.extract_symptoms('I have a skin rash and headache')}")
#     print(f"TEST CASE 2 : {s.extract_symptoms('I am suffering from fever and headache')}")
#     print(f"TEST CASE 3 : {s.extract_symptoms('I have a headache')}")
#     print(f"TEST CASE 4 : {s.extract_symptoms('I have a skin rash')}")
#     print(f"TEST CASE 5 : {s.extract_symptoms('Bro I have a skin rash and headache')}")
#     print(f"TEST CASE 6 : {s.extract_symptoms('I am actually suffering from fever and headache')}")
#     print(f"TEST CASE 7 : {s.extract_symptoms('I have a headache')}")
#     print(f"TEST CASE 8 : {s.extract_symptoms('I have a stomach pain')}")
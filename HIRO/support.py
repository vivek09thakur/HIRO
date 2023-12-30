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
            # remove the element which caused the error and return the symptoms
            return self.extracted_symptoms.pop(KeyError)
            

    
# class chat_support:
    
#     def __init__(self,pairs_file):
#         self.pairs_file = pairs_file
#         with open(self.pairs_file,'r') as f:
#             self.pairs = json.load(f)
#         self.questions = self.pairs['questions']
#         self.answers = self.pairs['responses']
#         self.vectorizer = CountVectorizer()
#         self.vectorizer.fit(self.questions)
#         self.question_vectors = self.vectorizer.transform(self.questions)
#         self.fail_case_responses = [
#             'Sorry But the symptoms you have provided are not enough to predict the disease please provide more symptoms',
#             'Sorry I could not understand your symptoms please provide more symptoms clearly can you please repeat it',
#             'I am not trained to answer this question please provide more symptoms',
#             'Sorry but I did not understand your symptoms please provide more symptoms clearly can you please repeat it'
#         ]
        
#     def get_response(self,user_input):
#         try:
#             self.user_input = user_input
#             self.user_input_vector = self.vectorizer.transform([self.user_input])
#             self.similarity_scores = cosine_similarity(self.user_input_vector,self.question_vectors)
#             self.most_similar_index = self.similarity_scores.argmax()
#             self.response = self.answers[self.most_similar_index]
            
#             self.confidence = self.similarity_scores[0][self.most_similar_index]
#             if self.confidence < 0.5:
#                 return random.choice(self.fail_case_responses)
            
#             return self.response
        
#         except Exception:
#             return random.choice(self.fail_case_responses)
import numpy as np
import tensorflow as tf
import json
import os.path

class CHAT_SUPPORT:
    
    def __init__(self,intents_file,model_path):
        self.intents_file = intents_file
        self.model_path = model_path
        
        self.words = []
        self.classes = []
        self.documents = []
        self.training_data = []
        self.model = None
        
        with open(self.intents_file) as f:
            self.intents = json.load(f)
            
    def process_training_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                words_in_pattern = pattern.lower().split()
                self.words.extend(words_in_pattern)
                self.documents.append((words_in_pattern,intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
                    
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        for doc in self.documents:
            bag = [1 if word in doc[0] else 0 for word in self.words]
            output_row = [0]*len(self.classes)  
            output_row[self.classes.index(doc[1])] = 1
            self.training_data.append([bag,output_row])
            
        np.random.shuffle(self.training_data)
        
        self.train_x = np.array([data[0] for data in self.training_data])
        self.train_y = np.array([data[1] for data in self.training_data])
        
        
    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128,input_shape=(len(self.train_x[0]),),activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.train_y[0]),activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
    def train_model(self):
        self.model.fit(self.train_x,self.train_y,epochs=500,batch_size=8)
        self.model.save(self.model_path)
        
    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        
        
    def generate_response(self,prompt):
        input_words = prompt.lower().split()
        input_bag = [0]*len(self.words)
        
        for word in input_words:
            for i,w in enumerate(self.words):
                if w == word:
                    input_bag[i] = 1
                    
        results = self.model.predict(np.array([input_bag]))[0]
        results_index = np.argmax(results)
        tag = self.classes[results_index]
        
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
    
    def build_robust_model(self):
        self.process_training_data()
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.build_model()
            self.train_model()
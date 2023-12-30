import numpy as np
import tensorflow as tf
import json
import os.path


class CHAT_SUPPORT:
    
    def __init__(self,intent_data,modeL_path) -> None:
        self.intent_data = intent_data
        self.model_path = modeL_path
        
        with open(self.intent_data) as f:
            self.intents = json.load(f)
            
        self.intent = self.intents['intents']
        self.words = []
        self.classes = []
        self.documents = []
        self.training_data = []
        self.model = None
        self.failure_responses = [
            'Sorry I could not understand your question please provide more details',
            'Sorry I could you please repeat that',
            'Sorry I could not understand your question please provide more details',
            'I did not understand your question please provide more details',
            'Sorry But I didn\'t find any symptoms in your question please provide more details',
            'Sorry But I didn\'t see any symptoms in your message please provide more details',
            'Sorry I could not understand your question please provide more details',
        ]
        
    def process_training_data(self):
        for intent in self.intent:
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
            self.training_data.append(bag)
        self.training_data = np.array(self.training_data)
        self.output_empty = [0] * len(self.classes)
        self.training_labels = []
        for doc in self.documents:
            output_row = list(self.output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.training_labels.append(output_row)
        self.training_labels = np.array(self.training_labels)
        
        
    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128,input_shape=(len(self.training_data[0]),),activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.training_labels[0]),activation='softmax')
        ])
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        
    def train_model(self):
        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)
        else:
            self.process_training_data()
            self.create_model()
            self.model.fit(self.training_data,self.training_labels,epochs=1000,batch_size=8)
            self.model.save(self.model_path)
            
            
    def train_or_load_model(self):
        if os.path.isfile(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.train_model()
            
            
    def generate_response(self,user_input):
        self.user_input = user_input.lower().split()
        self.bag = [1 if word in self.user_input else 0 for word in self.words]
        self.bag = np.array(self.bag)
        print("self.words shape:", len(self.words))
        print("self.bag shape:", self.bag.shape)

        self.predictions = self.model.predict(np.array([self.bag]))[0]
        self.predictions_index = np.argmax(self.predictions)
        self.tag = self.classes[self.predictions_index]
        self.confidence = self.predictions[self.predictions_index]
        if self.confidence > 0.5:
            for intent in self.intent:
                if intent['tag'] == self.tag:
                    self.response = intent['responses']
            return np.random.choice(self.response)
        else:
            return np.random.choice(self.failure_responses)
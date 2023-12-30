import numpy as np
import tensorflow as tf
import json
import os.path

class CHAT_SUPPORT:

    def __init__(self, intent_data, model_path) -> None:
        self.intent_data = intent_data
        self.model_path = model_path

        # Load intent data
        with open(self.intent_data) as f:
            self.intents = json.load(f)

        # Extract patterns and responses from intents
        self.patterns = []
        self.responses = []
        for intent in self.intents['intents']:
            self.patterns.extend(intent['patterns'])
            self.responses.extend(intent['responses'])

        self.words = sorted(list(set(self.patterns)))
        self.classes = sorted(list(set(intent['tag'] for intent in self.intents['intents'])))

        self.documents = [(pattern.lower().split(), intent['tag']) for intent in self.intents['intents'] for pattern in intent['patterns']]

        self.training_data = []
        self.process_training_data()

        self.model = None
        self.failure_responses = [
            'Sorry, I could not understand your question. Please provide more details.',
            'Sorry, could you please repeat that?',
            'Sorry, I did not understand your question. Please provide more details.'
        ]

    def process_training_data(self):
        for doc in self.documents:
            bag = [1 if word in doc[0] else 0 for word in self.words]
            self.training_data.append(bag)

        self.training_data = np.array(self.training_data)
        self.output_empty = [0] * len(self.classes)
        self.training_labels = [self.output_empty[:] for _ in range(len(self.documents))]
        for i, doc in enumerate(self.documents):
            self.training_labels[i][self.classes.index(doc[1])] = 1

        self.training_labels = np.array(self.training_labels)

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(len(self.training_data[0]),), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.training_labels[0]), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self):
        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)
        else:
            self.process_training_data()
            self.create_model()
            self.model.fit(self.training_data, self.training_labels, epochs=1000, batch_size=8)
            self.model.save_weights(self.model_path)

    def train_or_load_model(self):
        if os.path.isfile(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.train_model()

    def get_response(self, user_input):
        bag_of_words = [1 if word in user_input.lower().split() else 0 for word in self.words]
        bag_of_words = np.array(bag_of_words).reshape(1, -1)

        results = self.model.predict(bag_of_words)[0]
        results_index = np.argmax(results)
        tag = self.classes[results_index]

        if results[results_index] > 0.7:
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    return np.random.choice(intent['responses'])
        else:
            return np.random.choice(self.failure_responses)

# Example usage
# intent_data_path = "path/to/your/intent_data.json"
# model_path = "path/to/your/model_weights.h5"

# chatbot = CHAT_SUPPORT(intent_data_path, model_path)
# chatbot.train_or_load_model()

# user_input = input("User: ")
# response = chatbot.get_response(user_input)
# print("Chatbot:", response)

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

from .support import support
from .chat_support import CHAT_SUPPORT

import csv
import pyttsx3
from warnings import filterwarnings


class HEALTHCARE_COMPANION:
    def __init__(
        self,
        training_data_path,
        test_data_path,
        description_data,
        precaution_data,
        chat_log_path,
    ):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.description_data = description_data
        self.precaution_data = precaution_data
        self.chat_log_path = chat_log_path
        self.data = pd.read_csv(self.training_data_path)
        self.test_data = pd.read_csv(self.training_data_path)
        self.disease_list = self.data["prognosis"].unique()
        self.supportive_module = support(self.test_data_path)
        
        self.chat_support = CHAT_SUPPORT(
            intent_data=self.chat_log_path, model_path="./HIRO/chat_model.h5"
        )
        self.chat_support.train_or_load_model()

    def preprocess(self):
        self.encoder = LabelEncoder()
        self.data = self.data.dropna(axis=1)
        self.data["prognosis"] = self.encoder.fit_transform(self.data["prognosis"])
        self.X = self.data.iloc[:, :-1]
        self.Y = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=24
        )

    def train(self, show_accuracy=False):
        def vc_scoring(estimator, X, Y):
            return accuracy_score(Y, estimator.predict(X))

        models = {
            "SVC": SVC(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
        }

        for model_name in models:
            model = models[model_name]
            scores = cross_val_score(
                model, self.X, self.Y, cv=10, n_jobs=-1, scoring=vc_scoring
            )

            if show_accuracy == True:
                self.type_text(
                    f"(model loaded) MODEL NAME => {model_name} \\ Accuracy => {scores.mean()}\n"
                )

    def build_model(self):
        # TRAINING AND TESTING SVC MODEL
        self.svc_model = SVC()
        self.svc_model.fit(self.X_train, self.Y_train)
        self.svc_pred = self.svc_model.predict(self.X_test)

        # TRAINING AND TESTING NAIVE BAYES MODEL
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.X_train, self.Y_train)
        self.svc_pred = self.nb_model.predict(self.X_test)

        # TRAINING AND TESTING RANDOM FOREST MODEL
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X_train, self.Y_train)
        self.svc_pred = self.rf_model.predict(self.X_test)

    def combine_model(self):
        # TRAINING THE MODELS ON THE ENTIRE DATASET
        self.svc_model.fit(self.X, self.Y)
        self.nb_model.fit(self.X, self.Y)
        self.rf_model.fit(self.X, self.Y)
        # Read the test data
        self.test_data = pd.read_csv(self.test_data_path).dropna(axis=1)
        self.test_X = self.test_data.iloc[:, :-1]
        self.test_Y = self.encoder.transform(self.test_data.iloc[:, -1])
        # predict the test data
        self.svc_pred = self.svc_model.predict(self.test_X)
        self.nb_pred = self.nb_model.predict(self.test_X)
        self.rf_pred = self.rf_model.predict(self.test_X)

        self.final_pred = [
            mode([self.svc_pred[i], self.nb_pred[i], self.rf_pred[i]])[0][0]
            for i in range(len(self.svc_pred))
        ]

        return self.final_pred

    def collect_symptoms_data(self):
        try:
            self.symptoms = self.X.columns.values
            self.symptoms_index = {}
            for index, symtom in enumerate(self.symptoms):
                self.symptoms = " ".join([i.capitalize() for i in symtom.split("_")])
                self.symptoms_index[self.symptoms] = index

            self.data_dict = {
                "symptoms_index": self.symptoms_index,
                "prediction_class": self.encoder.classes_,
            }
        except Exception as e:
            print(f"ERROR OCCURED WHILE COLLECTING SYMPTOMS DATA => {e}")

    # SUPPORTIVE FUNCTIONS
    def extract_symptoms(self, sentence, show_extracted_symptoms=False):
        extracted_symptoms = self.supportive_module.extract_symptoms(sentence)
        if show_extracted_symptoms == True:
            if extracted_symptoms:
                self.say_to_user(
                    f"\nOkay I have founded these following symptoms : {extracted_symptoms}"
                )
            else:
                print("\nSYMPTOMS FOUNDED => None")
        return extracted_symptoms

    def talk_to_user(self,user_input):
       return self.chat_support.get_response(str(user_input))

    def predict_disease_from_symptoms(self, user_input):
        try:
            symptoms = user_input.split(",")
            input_data = [0] * len(self.data_dict["symptoms_index"])
            for symtom in symptoms:
                index = self.data_dict["symptoms_index"][symtom]
                input_data[index] = 1

            input_data = np.array(input_data).reshape(1, -1)

            # Generate the prediction individually
            rf_pred = self.data_dict["prediction_class"][
                self.rf_model.predict(input_data)[0]
            ]
            svc_pred = self.data_dict["prediction_class"][
                self.svc_model.predict(input_data)[0]
            ]
            nb_pred = self.data_dict["prediction_class"][
                self.nb_model.predict(input_data)[0]
            ]

            final_pred = mode([rf_pred, svc_pred, nb_pred])[0][0]

            all_predictions = {
                "Random Forest": rf_pred,
                "SVC": svc_pred,
                "Naive Bayes": nb_pred,
                "Final Prediction": final_pred,
            }

            return all_predictions
        except Exception as e:
            print(
                "ERROR OCCURED WHILE PREDICTING DISEASE FROM SYMPTOMS\n ERROR =>{}".format(
                    e
                )
            )

    def get_description(self, disease):
        disease_description_dict = {}
        with open(self.description_data, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                disease_description_dict[row[0]] = row[1]
        try:
            return disease_description_dict[disease]
        except Exception as e:
            print("ERROR OCCURED WHILE GETTING DESRCIPTION => {}".format(e))
            return "Sorry I could not find the description of the disease"

    def get_precautions(self, disease):
        precautions_dict = {}
        with open(self.precaution_data, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                precautions_dict[row[0]] = row[1]
        try:
            return precautions_dict[disease]
        except Exception as e:
            print(f"ERROR OCCURED WHILE GETTING PRECUATIONS => {e}")
            return "Sorry I could not find the precautions of the disease"

    def process_training_data(self, show_models_stats=False):
        filterwarnings("ignore")
        self.preprocess()
        self.train(show_accuracy=show_models_stats)

    def build_robust_model(self):
        self.build_model()
        self.combine_model()
        self.collect_symptoms_data()

    def introduce(self, ask_for_paitent_name=False):
        try:
            if ask_for_paitent_name == True:
                paitent_name = None
                self.say_to_user("\nHey there!,Can I get your name first?")
                while paitent_name == None:
                    paitent_name = input("\nEnter your name here : ")
                    if paitent_name != None:
                        self.say_to_user(
                            f"\nHello {paitent_name}, I am HIRO, your healthcare chatbot.I can help you diagnose your disease based on your symptoms. Let's start with your problem"
                        )
                        break
                    else:
                        self.say_to_user(
                            "Please enter your name,So that we can continue"
                        )
            else:
                self.say_to_user(
                    f"\nHello there!, I am HIRO, your healthcare chatbot.I can help you diagnose your disease based on your symptoms. Let's start with your problem"
                )
        except Exception as introduction_error:
            self.say_to_user("Sorry I think that you have done something wrong")
            print("ERROR OCCRURED => {}".format(introduction_error))

    def show_diseases(
        self, disease_dictionary, show_description=False, show_precautions=False
    ):
        self.say_to_user("\nOkay just wait for a second!, Let me analyze your symptoms")
        self.type_text(f'\nTEST 1 => {disease_dictionary["Random Forest"]}')
        self.type_text(f'\nTEST 2 => {disease_dictionary["SVC"]}')
        self.type_text(f'\nTEST 3 => {disease_dictionary["Naive Bayes"]}')
        self.say_to_user(
            f'\nAfter examining everything I found that you might have : {disease_dictionary["Final Prediction"]}'
        )

        if show_description == True:
            disease_description = self.get_description(
                disease_dictionary["Final Prediction"]
            )
            if disease_description != None:
                self.say_to_user(f"\n\nDisease Description : {disease_description}")
            else:
                self.say_to_user(
                    "\n\nSorry I could not find the description of the disease"
                )

        if show_precautions == True:
            disease_precautions = self.get_precautions(
                disease_dictionary["Final Prediction"]
            )
            if disease_precautions != None:
                self.say_to_user(f"\n\nDisease Precautions : {disease_precautions}\n")
            else:
                self.say_to_user(
                    "\n\nSorry I could not find the precautions of the disease\n"
                )

    def get_diseases(self, disease_dictionary):
        test1 = disease_dictionary["Random Forest"]
        test2 = disease_dictionary["SVC"]
        test3 = disease_dictionary["Naive Bayes"]
        final_prediction = disease_dictionary["Final Prediction"]
        disease_description = self.get_description(final_prediction)

        return {
            "test1": test1,
            "test2": test2,
            "test3": test3,
            "final_prediction": final_prediction,
            "disease_description": disease_description,
        }

    def type_text(self, text):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.004)

    def speak(self, text):
        engine = pyttsx3.init()
        engine.setProperty("Volume", 1.0)
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()

    def say_to_user(self, question, speaker_name=None):
        if speaker_name != None:
            print(f"{speaker_name} => ", end="")
        self.type_text(question)
        self.speak(question)

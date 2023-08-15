import os
import csv
import pandas as pd
import numpy as np
from utils import check_list_match
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import preprocessing

TRAIN_DATASET_PATH = "./dataset/data/Training.csv"
TEST_DATASET_PATH = "./dataset/data/Testing.csv"


class HIRO:
    def __init__(self):
        self.DESCRIPTION_LIST = self.load_description_list()
        self.PRECAUTION_LIST = self.load_precaution_list()
        self.SEVERITY_LIST = self.load_severity_list()

        self.COLUMNS, self.DATASET, self.DATASET_X, self.DATASET_Y = self.load_dataset()

        self.classifier, self.CLASSIFIER_SCORE = self.load_classifier()
        self.encoder = self.load_encoder()

    def introduce(self):
        os.system("cls")
        print(
            "Hello , I am HIRO , your own healthcare companion.I am here to make you fit and fine ^_^"
        )
        name = input(
            "But you have to tell me your name first. what is your name? \n\n[patient name] :: "
        )
        print(f"\nSo hello {name} , let's start with your problem ")

    def load_description_list(self):
        description_list = dict()
        with open("./dataset/main/symptom_Description.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                _description = {row[0]: row[1]}
                description_list.update(_description)
        return description_list

    def load_severity_list(self):
        severity_dictionary = dict()
        with open("./dataset/main/Symptom_severity.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if len(row) == 2:
                    _diction = {row[0]: int(row[1])}
                    severity_dictionary.update(_diction)
        return severity_dictionary

    def load_precaution_list(self):
        precaution_dictionary = dict()
        with open("./dataset/main/symptom_precaution.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precaution_dictionary.update(_prec)
        return precaution_dictionary

    def load_dataset(self):
        # train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
        # test_dataset = pd.read_csv(TEST_DATASET_PATH)

        dataset = pd.read_csv(TRAIN_DATASET_PATH)

        columns = dataset.columns
        columns = columns[:-1]

        dataset_x = dataset[columns]
        dataset_y = dataset["prognosis"]

        return [columns, dataset, dataset_x, dataset_y]

    def load_classifier(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.DATASET_X, self.DATASET_Y, test_size=0.33, random_state=42
        )

        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        scores = cross_val_score(classifier, x_test, y_test, cv=3)

        return [classifier, scores]

    def load_encoder(self):
        encoder = preprocessing.LabelEncoder()
        encoder.fit(self.DATASET_Y)
        return encoder

    def print_disease(self, node):
        node = node[0]
        val = node.nonzero()
        disease = self.encoder.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def get_disease_from_name(self, symptom_name, node=0, depth=1):
        tree_ = self.classifier.tree_
        symptoms_present = []
        feature_name = [
            self.COLUMNS[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptom_name:
                val = 1
            else:
                val = 0
            if val <= threshold:
                self.get_disease_from_name(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                self.get_disease_from_name(tree_.children_right[node], depth + 1)
        else:
            present_disease = self.print_disease(tree_.value[node])
            return present_disease

    def get_probable_symptoms(self, disease):
        reduced_data = self.DATASET.groupby(self.DATASET["prognosis"]).max()
        red_cols = reduced_data.columns
        probable_symptoms = red_cols[reduced_data.loc[disease].values[0].nonzero()]
        return probable_symptoms

    def get_disease_from_arr(symptoms_arr):
        df = pd.read_csv("./dataset/data/Training.csv")
        X = df.iloc[:, :-1]
        y = df["prognosis"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=20
        )
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_arr:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    def is_consult_doctor(self, symptoms_arr, days):
        THRESHOLD = 13

        severity = 0
        for item in symptoms_arr:
            severity = severity + self.SEVERITY_LIST[item]

        if severity > THRESHOLD:
            return True

        return False

    def get_precautions(self, disease):
        return self.PRECAUTION_LIST[disease]

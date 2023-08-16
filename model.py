import re
import csv
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

train = pandas.read_csv("dataset/data/Training.csv")
# test = pandas.read_csv("dataset/data/Testing.csv")

cols = train.columns
cols = cols[:-1]
x = train[cols]
y = train["prognosis"]

# y1 = y

reduced_data = train.groupby(train["prognosis"]).max()

encoder = preprocessing.LabelEncoder()
encoder.fit(y)
# y = encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
# testx = testing[cols]
# testy = testing["prognosis"]
# testy = le.transform(testy)


classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
scores = cross_val_score(classifier, x_test, y_test, cv=3)
# print(scores.mean())

# INITIALIZING THE MODEL
# model = SVC()
# model.fit(x_train, y_train)
# print("model score :", model.score(x_test, y_test))

feature_importance = classifier.feature_importances_

# indices = np.argsort(importances)[::-1]
# features = cols

# severity_dictionary = dict()
# description_list = dict()
# precaution_list=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


# # Calculate condition
def calc_condition(exp, days, severity_list):
    sum = 0
    for item in exp:
        sum = sum + severity_list[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("\nYou should take the consultation from doctor.")
    else:
        print("\nIt might not be that bad but you should take precautions.")


def load_description_list():
    description_list = dict()
    with open("./dataset/main/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)
    return description_list


def load_severity_list():
    severity_dictionary = dict()
    with open("./dataset/main/Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if len(row) == 2:
                _diction = {row[0]: int(row[1])}
                severity_dictionary.update(_diction)
    return severity_dictionary


def load_precaution_list():
    precaution_dictionary = dict()
    with open("./dataset/main/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precaution_dictionary.update(_prec)
    return precaution_dictionary


def display_intro():
    os.system("cls")
    print(
        "Hello , I am HIRO , your own healthcare companion.I am here to make you fit and fine ^_^"
    )
    name = input(
        "But you have to tell me your name first. what is your name? \n\n[patient name] :: "
    )
    print(f"\nSo hello {name} , let's start with your problem ")


def check_match(list, value):
    pred_list = []  # prediction list
    value = value.replace(" ", "_")
    patt = f"{value}"
    regexp = re.compile(patt)
    pred_list = [item for item in list if regexp.search(item)]
    if len(pred_list) > 0:
        return pred_list
    else:
        return []


def sec_predict(symptoms_exp):
    df = pandas.read_csv("./dataset/data/Training.csv")
    X = df.iloc[:, :-1]
    y = df["prognosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20
    )
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = encoder.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def run(tree, feature_names):
    # tree_ = tree.tree_
    # feature_name = [
    #     feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    #     for i in tree_.feature
    # ]

    # print(feature_name)

    symptoms = ",".join(feature_names).split(",")

    # print(chk_dis)

    symptoms_present = []

    while True:
        print(
            "\ncan you please all the symptom you are experiencing from past few days so I can understand you more!"
        )
        symptom_input = input("\n[enter symtoms description here] :: ")
        matched_symptoms = check_match(symptoms, symptom_input)

        if not len(matched_symptoms):
            print("enter valid symptom.")
            continue

        print("\nsearches related to input: \n")
        for num, symptom in enumerate(matched_symptoms):
            print(num, ")", symptom)

        disease_input = input(
            f"select the one you meant (0 - {len(matched_symptoms) - 1}): "
        )
        if not disease_input.isdigit():
            print("please enter a valid input for choice.")
            continue

        disease_input = int(disease_input)
        disease_input = matched_symptoms[disease_input]

        while True:
            try:
                num_days = int(input("\nOkay. From how many days ? : "))
                break
            except:
                print("\nplease enter a valid input.")


# getSeverityDict()
# getDescription()
# getprecautionDict()
# get_info()

severity_list = load_severity_list()
description_list = load_description_list()
precaution_list = load_precaution_list()

# display_intro()
# run(classifier, cols)
# print(severity_list)


tree_ = classifier.tree_
feature_name = [
    cols[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature
]

disease_input = "headache"
num_days = 10
symptoms_present = []


def get_disease_from_name(symptom_name, node=0, depth=1):
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == symptom_name:
            val = 1
        else:
            val = 0
        if val <= threshold:
            get_disease_from_name(tree_.children_left[node], depth + 1)
        else:
            symptoms_present.append(name)
            get_disease_from_name(tree_.children_right[node], depth + 1)
    else:
        present_disease = print_disease(tree_.value[node])
        return present_disease


def get_probable_symptoms(disease):
    red_cols = reduced_data.columns
    probable_symptoms = red_cols[reduced_data.loc[disease].values[0].nonzero()]
    return probable_symptoms


def get_disease_from_arr(symptoms_arr):
    df = pandas.read_csv("./dataset/data/Training.csv")
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


def recurse(node, depth):
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == disease_input:
            val = 1
        else:
            val = 0
        if val <= threshold:
            recurse(tree_.children_left[node], depth + 1)
        else:
            symptoms_present.append(name)
            recurse(tree_.children_right[node], depth + 1)
    else:
        present_disease = print_disease(tree_.value[node])

        red_cols = reduced_data.columns
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

        print(symptoms_given)

        print(
            "\nOkay Now I am going to ask you some question , please answer all of them in yes or no \n"
        )
        symptoms_exp = []
        for syms in list(symptoms_given):
            inp = ""
            print(f"\nAre you experiencing any {syms} ? \n")
            while True:
                inp = input("[yes or no] : ")
                if inp == "yes" or inp == "no":
                    break
                else:
                    print("provide proper answers i.e. (yes/no) : ", end="")
            if inp == "yes":
                symptoms_exp.append(syms)

        second_prediction = sec_predict(symptoms_exp)

        calc_condition(symptoms_exp, num_days, severity_list)
        if present_disease[0] == second_prediction[0]:
            print("\nYou may have ", present_disease[0])
            print(f"\nDESCRIPTION OF DISEASES :", description_list[present_disease[0]])

        else:
            print("\nYou may have ", present_disease[0], "or ", second_prediction[0])
            print(description_list[present_disease[0]])
            print(description_list[second_prediction[0]])

        precution_list = precaution_list[present_disease[0]]
        print("\nTake following measures : \n")
        for i, j in enumerate(precution_list):
            print(i + 1, ")", j)


recurse(0, 1)

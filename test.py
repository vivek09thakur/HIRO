import re
import csv
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

training = pandas.read_csv("dataset/data/Training.csv")
testing = pandas.read_csv("dataset/data/Testing.csv")

cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training["prognosis"]

y1 = y

reduced_data = training.groupby(training["prognosis"]).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
testx = testing[cols]
testy = testing["prognosis"]
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

# INITIALIZING THE MODEL
model = SVC()
model.fit(x_train, y_train)
print("model score :", model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


# Calculate condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("\nYou should take the consultation from doctor.")
    else:
        print("\nIt might not be that bad but you should take precautions.")


# Get description
def getDescription():
    global description_list
    with open("./dataset/main/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open("./dataset/main/Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open("./dataset/main/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def get_info():
    os.system("cls")
    print(
        "Hello , I am HIRO , your own healthcare companion.I am here to make you fit and fine ^_^"
    )
    name = input(
        "But you have to tell me your name first. what is your name? \n\n[patient name] :: "
    )
    print(f"\nSo hello {name} , let's start with your problem ")


def check_pattern(dis_list, inp):
    pred_list = []  # prediction list
    inp = inp.replace(" ", "_")
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pandas.read_csv("./dataset/data/Training.csv")
    X = df.iloc[:, :-1]
    y = df["prognosis"]
    X_train, y_train = train_test_split(
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
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def run(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print(
            "\ncan you please all the symptom you are experiencing from past few days so I can understand you more!"
        )
        disease_input = input("\n[enter symtoms description here] :: ") # parameter 1
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("\nsearches related to input: \n")
            for num, item in enumerate(cnf_dis):
                print(num, ")", item)
            if num != 0:
                try:
                    conf_inp = int(input(f"select the one you meant (0 - {num}): ")) # parameter 2
                except Exception:
                    print("\nplease enter a valid input for choice")
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("enter valid symptom.")

    while True:
        try:
            num_days = int(input("\nOkay. From how many days ? : ")) # parameter 3
            break
        except:
            print("\nplease enter a valid input.")

    def recurse(node,depth):
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
            print("\nOkay Now I am going to ask you some question , please answer all of them in yes or no \n")
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
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("\nYou may have ", present_disease[0])
                print(f"\nDESCRIPTION OF DISEASES :", description_list[present_disease[0]])
            else:
                print("\nYou may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])
            precution_list = precautionDictionary[present_disease[0]]
            print("\nTake following measures : \n")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)
    recurse(0, 1)


getSeverityDict()
getDescription()
getprecautionDict()
get_info()
run(clf, cols)

import pandas as pd

# class support_module:
    
#     def __init__(self,training_data_file,test_data_csv):
#         self.training_data_file = training_data_file
#         self.test_data_csv = test_data_csv
#         with open(self.training_data_file,'r') as f:
#             reader = csv.reader(f)
#         for 
#         pass
    
#     def get_disease_from_sentence(self,sentence):
#         pass
l1 = []
with open('Notebook/dataset/Testing.csv','r') as f:
    reader = pd.read_csv(f)
    for row in reader:
        l1.append(row)
        
# print(l1)
user_input = 'itching'
for item in l1:
    if user_input == item:
        print(item)
from beta.HIRO import HIRO

params = [
    './dataset/data/Training.csv', # training_data
    './dataset/data/Testing.csv', # testing_data
    './dataset/main/Symptom_severity.csv', # serverity_data 
    './dataset/main/symptom_precaution.csv' ,# precaution_data
    './dataset/main/symptom_Description.csv' # description_dat
]

hiro = HIRO(params[0],params[1])
hiro.introduce(patient_name=input('[patient name] :: '))
serverity_list = hiro.getServersity(params[2])
description_list = hiro.getDescription(params[4])
precaution_dict = hiro.getPrecaution(params[3])


while True:
    user_problem = input('\n[can you explain your problem] :: ')
    result = hiro.get_user_problem(user_problem)
    
    if result[0] == 1:
        print(f'\nsearches related to {result[1][0]} :')
        for item_number , item_name in enumerate(result[1]):
            print(f'{item_number} ) {item_name}')
        if item_number !=0:
                confidence_input = hiro.get_choice(input(f'select an option from 1 - {item_number} : '),
                    'please enter choice in digits (0-9)!') 
        else:
            confidence_input = 0
        disease_input = result[1]
    else:
        print('please enter a valid symptoms')
        
    num_days = hiro.get_choice(input("\nOkay. From how many days ? : "),
                    'please enter number of days in digits (0-9)!')
    given_symptoms,first_prediction = hiro.recurse(0,1,disease_input[0],result[2])
    
    print('\nOkay Now I am going to ask you some question,please answer all of them in yes or no')
    symptoms_exp = []
    for symps in given_symptoms:
        choice = input(f'\nAre you experiencing any {symps} ? \nEnter your answer in [yes or no] :: ')
        if choice == 'yes':
            symptoms_exp.append(symps)
            
    second_prediction = hiro.second_prediction(symptoms_exp)
    patient_condition = hiro.calcCondition(symptoms_exp,num_days)
    print(patient_condition[1])
    
    try:
        if first_prediction[0]==second_prediction[0]:
            print(f'''You may have {first_prediction[0]}
                  DISEASE DESCRIPTION : {description_list[first_prediction[0]]}''')
        else:
            print(f'\nYou may have {first_prediction[0]} or {second_prediction[0]}')
            print(f'''
                DISEASE DESCRIPTION [1] : {description_list[first_prediction[0]]}
                DISEASE DESCRIPTION [2] : {description_list[second_prediction[0]]}
            ''')
            precaution_list = precaution_dict[first_prediction[0]]
            for precaution_num,precaution_discription in enumerate(precaution_list):
                print(f'{precaution_num} -- {precaution_discription}')
    except Exception as e:
        print(f'ERROR OCCURED : {e}')
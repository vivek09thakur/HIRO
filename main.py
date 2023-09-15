from hiro.hiro import HIRO
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

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


if __name__ == '__main__':
    
    # KEEP ASKING USER FOR PROBLEM UNTIL HE/SHE ENTERS A VALID SYMPTOMS
    while True:
        user_problem = input('\n[can you explain your problem] :: ')
        result = hiro.get_user_problem(user_problem)

        # IF RESULT IS 1 THEN IT MEANS THAT USER HAS ENTERED A VALID SYMPTOMS
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
            
        
        # IF RESULT IS 2 THEN IT MEANS THAT USER HAS ENTERED A VALID DISEASE
        num_days = hiro.get_choice(input("\nOkay. From how many days ? : "),
                                   'please enter number of days in digit (0-9!')
        given_symptoms,first_prediction = hiro.recurse(0,1,disease_input[0],result[2])
        print('\nOkay Now I am going to ask you some question,please answer all of them in yes or no')
        symptoms_exp = []
        for symps in given_symptoms:
            choice = input(f'\nAre you experiencing any {symps} ? \nEnter your answer in [yes or no] :: ')
            if choice == 'yes':
                symptoms_exp.append(symps) 
                
        
        # START PREDICTING THE DISEASE
        second_prediction = hiro.second_prediction(symptoms_exp)
        patient_condition = hiro.calcCondition(symptoms_exp,num_days)
        hiro.show_patient_condition(patient_condition[1])
        try:
            if first_prediction[0]==second_prediction[0]:
                hiro.show_first_prediction(first_prediction[0],description_list)
            else:
                hiro.show_second_prediction(first_prediction[0],second_prediction[0],description_list)
                precaution_list = precaution_dict[first_prediction[0]]
                for precaution_num,precaution_discription in enumerate(precaution_list):
                    print(f'{precaution_num} -- {precaution_discription}')
        except Exception as e:
            print('ERROR OCCURED : {}'.format(e))
            pass
        
        # IF USER WANTS TO CONTINUE THEN CONTINUE ELSE BREAK THE LOOP
        choice = input('To turn me off you have to say [ I am fine ] :: ')
        if choice.lower() == 'i am fine':
            break
        else:
            continue
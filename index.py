from HIRO.HIRO import HEALTHCARE_COMPANION
from warnings import filterwarnings

filterwarnings('ignore')

params = ['./Notebook/dataset/Training.csv',
    './Notebook/dataset/Testing.csv',
    './Notebook/dataset/symptom_Description.csv',
    './Notebook/dataset/precautions.csv',
    './Notebook/question_response_pairs.json'
]

hiro = HEALTHCARE_COMPANION(params[0],params[1],params[2],params[3],params[4])
hiro.process_training_data(show_accuracy=True)
hiro.build_robust_model()


if __name__ == '__main__':
    hiro.introduce('Guest')
    
    while True:
        user_input = input('\nUSER : ')
        extracted_symptoms = hiro.extract_symptoms(user_input)

        if extracted_symptoms:
            print('\nSYMPTOMS FOUND => ',extracted_symptoms)
            disease = hiro.predict_disease_from_symptoms(extracted_symptoms)
            hiro.show_diseases(disease,show_description=True,show_precautions=True)
        else:
            hiros_response = hiro.talk_to_user(user_input)
            hiro.type_text(f'\nHIRO : {hiros_response}')
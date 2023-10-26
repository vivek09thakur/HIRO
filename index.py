from HIRO.HIRO import HEALTHCARE_COMPANION

TRAINING_DATASET = './Notebook/dataset/Training.csv'
TESTING_DATASET = './Notebook/dataset/Testing.csv'
SYMPTOM_DESCRIPTION = './Notebook/dataset/symptom_Description.csv'
PRECAUTION_DATASET = './Notebook/dataset/precautions.csv'
QUESTION_RESPONSE_DATASET = './Notebook/question_response_pairs.json'

hiro = HEALTHCARE_COMPANION(TRAINING_DATASET, TESTING_DATASET, SYMPTOM_DESCRIPTION, PRECAUTION_DATASET,
                        QUESTION_RESPONSE_DATASET)
hiro.process_training_data(show_models_stats=True)
hiro.build_robust_model()

if __name__ == '__main__': 
    hiro.introduce(ask_for_paitent_name=True)
    
    while True: 
        try:
            user_input = input(f'\nUSER : ')
            extracted_symptoms = hiro.extract_symptoms(user_input,show_extracted_symptoms=True)
            
            if extracted_symptoms:
                disease = hiro.predict_disease_from_symptoms(extracted_symptoms)
                hiro.show_diseases(disease,show_description=True,show_precautions=True)
            else:
                hiros_response = hiro.talk_to_user(user_input)
                hiro.say_to_user(f'{hiros_response}',speaker_name='HIRO')
                
        except KeyboardInterrupt:
            hiro.say_to_user('''Looks like you have interrupted the keyboard,
                    If you don\'t wanna continue then goodbye''')
            
        except Exception as runtime_errors:
            hiro.type_text(f'ERROR OCCURED => {runtime_errors}')
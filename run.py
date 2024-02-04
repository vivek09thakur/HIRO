from HIRO.HIRO import HEALTHCARE_COMPANION

TRAINING_DATA = "DATASET/Training.csv"
TESTING_DATA = "DATASET/Testing.csv"
SYMPTOM_DESCRIPTION = "DATASET/symptom_Description.csv"
PRECAUTION_DATA = "DATASET/precautions.csv"
CHAT_DATASET = "DATASET/intents.json"

hiro = HEALTHCARE_COMPANION(TRAINING_DATA,TESTING_DATA,SYMPTOM_DESCRIPTION,PRECAUTION_DATA, CHAT_DATASET)
hiro.process_training_data(show_models_stats=True)
hiro.build_robust_model()
hiro.introduce(ask_for_paitent_name=True)

if __name__ == '__main__': 
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
            hiro.say_to_user("Keyboard Interrupted!,It seems you don't wanna continue.Have a nice day.")
            break
        
        except Exception as runtime_errors:
            hiro.type_text(f'ERROR OCCURED => {runtime_errors}')
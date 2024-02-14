from src.main import HEALTHCARE_COMPANION

TRAINING_DATA = "src/data/train.csv"
TESTING_DATA = "src/data/test.csv"
PRECAUTION = "src/data/precaution.csv"
DESCRIPTION = "src/data/description.csv"
INTENTS = "src/data/intents.json"
MODEL_PATH = "src/model/chat_model.h5"

hiro = HEALTHCARE_COMPANION(TRAINING_DATA,TESTING_DATA,DESCRIPTION,PRECAUTION,INTENTS,MODEL_PATH)
hiro.process_training_data(show_models_stats=True)
hiro.build_robust_model()
hiro.introduce(ask_for_paitent_name=False) 

if __name__ == "__main__":
    while True: 
        try:
            user_input = input(f'\n<user> ')
            extracted_symptoms = hiro.extract_symptoms(user_input,show_extracted_symptoms=True)
            
            if extracted_symptoms:
                disease = hiro.predict_disease_from_symptoms(extracted_symptoms)
                hiro.show_diseases(disease,show_description=True,show_precautions=True)
            else:
                hiros_response = hiro.talk_to_user(user_input)
                print(f'<hiro> {hiros_response}')
                
        except KeyboardInterrupt:
            print("Keyboard Interrupted! Exiting...")
            break
        
        except Exception as runtime_errors:
            print(f'ERROR OCCURED => {runtime_errors}')
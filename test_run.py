from HIRO.HIRO import HEALTHCARE_COMPANION
from warnings import filterwarnings

filterwarnings('ignore')

parameters = [
    './Notebook/dataset/Training.csv',
    './Notebook/dataset/Testing.csv'
]

hiro = HEALTHCARE_COMPANION(parameters[0],parameters[1])
hiro.process_training_data(show_accuracy=True)
hiro.build_robust_model()


if __name__ == '__main__':
    hiro.introduce('Guest')
    
    while True:
        user_input = input('\nUSER : ')
        # extracted_symptoms = hiro.extract_symptoms(user_input)
        disease = hiro.predict_disease_from_symptoms(user_input)
        hiro.show_diseases("HIRO :",disease)
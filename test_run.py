from HIRO.HIRO import HEALTHCARE_CHATBOT

parameters = [
    './Notebook/dataset/Training.csv',
    './Notebook/dataset/Testing.csv'
]

hiro = HEALTHCARE_CHATBOT(parameters[0],parameters[1])
hiro.introduce('Vivek')
hiro.prepare_model()

if __name__ == '__main__':
    
    user_input = input('Enter the symtoms separated by comma: ')
    disease = hiro.predict_disease_from_symtoms(user_input)
    print(f'You may have {disease}')
from hiro.hiro import HIRO
from utils import menu_selector, type_input, ask_filter_arr

hiro = HIRO()
hiro.introduce()

print(
    "Can you please tell all the symptom you are experiencing from past few days so I can understand you more!"
)
symptom_input = input("\n[enter symtoms description here] :: ")
matched_symptoms = hiro.get_matching_symptoms(symptom_input)
confirm_symptom = menu_selector(matched_symptoms)

num_days = type_input("\nOkay. From how many days ? : ", int)
first_predicted_disease = hiro.get_disease_from_symptom_name(confirm_symptom)
print(first_predicted_disease)
# probable_symptoms = hiro.get_probable_symptoms(first_predicted_disease)
# confirm_symptoms = ask_filter_arr(
#     probable_symptoms, lambda symptom: f"Are you expriencing any {symptom}"
# )
# second_predicted_disease = hiro.get_disease_from_symptom_arr(confirm_symptoms)
# print(second_predicted_disease)

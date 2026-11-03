import unittest

from model.hiro.common import label_for
from model.hiro.nlp import SymptomExtractor, strip_durations

SYMPTOMS = [
    "high_fever", "mild_fever", "headache", "cough", "vomiting", "nausea", "diarrhoea", "chest_pain", "back_pain",
    "stomach_pain", "fatigue", "chills", "throat_irritation", "runny_nose", "breathlessness", "skin_rash", "itching",
    "dizziness", "ear_pain", "loss_of_taste", "loss_of_smell", "difficulty_swallowing", "pale_skin", "joint_pain",
    "abdominal_pain", "knee_pain", "burning_micturition", "sweating", "muscle_pain", "pain_behind_the_eyes",
]


class SymptomExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = SymptomExtractor({k: label_for(k) for k in SYMPTOMS})

    def reported(self, text):
        return self.ex.extract(text)[0]

    def denied(self, text):
        return self.ex.extract(text)[1]

    def test_exact_and_alias(self):
        self.assertEqual(self.reported("I have a headache and high fever"), ["headache", "high_fever"])
        self.assertEqual(self.reported("been throwing up all night and feel queasy"), ["vomiting", "nausea"])
        self.assertEqual(self.reported("I've got loose motions"), ["diarrhoea"])

    def test_fever_modifiers(self):
        self.assertEqual(self.reported("slight fever"), ["mild_fever"])
        self.assertEqual(self.reported("temperature of 103"), ["high_fever"])
        self.assertEqual(self.reported("I'm running a temperature"), ["high_fever"])

    def test_spelling_correction(self):
        self.assertEqual(self.reported("i have feaver and headach"), ["high_fever", "headache"])
        self.assertEqual(self.reported("vomitting and diarhea"), ["vomiting", "diarrhoea"])
        self.assertEqual(self.reported("I feel dizy"), ["dizziness"])

    def test_body_parts_and_coordination(self):
        self.assertEqual(self.reported("pain in my chest"), ["chest_pain"])
        self.assertEqual(self.reported("my back hurts"), ["back_pain"])
        self.assertEqual(sorted(self.reported("pain in my chest and back")), ["back_pain", "chest_pain"])
        self.assertEqual(sorted(self.reported("my head and stomach hurt")), ["headache", "stomach_pain"])
        self.assertEqual(self.reported("my ears hurt"), ["ear_pain"])

    def test_negation_before(self):
        reported, denied = self.ex.extract("no fever but I do have a cough")
        self.assertEqual(reported, ["cough"])
        self.assertEqual(denied, ["high_fever"])
        self.assertEqual(self.denied("I don't have any rash or itching"), ["skin_rash", "itching"])
        self.assertEqual(self.denied("no pain in my chest"), ["chest_pain"])
        self.assertEqual(self.denied("not really dizzy"), ["dizziness"])

    def test_negation_after(self):
        reported, denied = self.ex.extract("the fever is gone but the cough stopped too, still have a headache")
        self.assertEqual(reported, ["headache"])
        self.assertEqual(sorted(denied), ["cough", "high_fever"])

    def test_durations_ignored(self):
        self.assertEqual(strip_durations("headache for 3 days now and fever since yesterday"), "headache and fever")
        self.assertEqual(self.reported("I've had a sore throat for the past two weeks"), ["throat_irritation"])
        self.assertEqual(self.reported("night sweats"), ["sweating"])

    def test_new_symptoms(self):
        self.assertEqual(self.reported("I can't taste anything"), ["loss_of_taste"])
        self.assertEqual(self.reported("it hurts to swallow"), ["difficulty_swallowing"])
        self.assertEqual(self.reported("I look very pale"), ["pale_skin"])
        self.assertEqual(self.reported("earache"), ["ear_pain"])

    def test_overlap_prefers_longest(self):
        self.assertEqual(self.reported("pain behind the eyes"), ["pain_behind_the_eyes"])
        self.assertEqual(self.reported("burning when I pee"), ["burning_micturition"])

    def test_nothing(self):
        self.assertEqual(self.ex.extract("hello there"), ([], []))
        self.assertEqual(self.ex.extract("yes"), ([], []))


if __name__ == "__main__":
    unittest.main()

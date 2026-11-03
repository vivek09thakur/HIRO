from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd

from ..hiro.constants import DATA_DIR, TARGET_COLUMN

RAW_DIR = DATA_DIR / "raw"
SEED = 7
MIN_KEEP = 4
MAX_ATTEMPTS = 200

COLUMN_FIXES = {
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
    "dischromic _patches": "dischromic_patches",
}

DISEASE_RENAMES = {
    "Diabetes ": "Diabetes",
    "Hypertension ": "Hypertension",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Haemorrhoids (piles)",
    "Dimorphic hemorrhoids(piles)": "Haemorrhoids (piles)",
    "Osteoarthristis": "Osteoarthritis",
    "(vertigo) Paroymsal  Positional Vertigo": "Benign paroxysmal positional vertigo",
    "Paralysis (brain hemorrhage)": "Paralysis (brain haemorrhage)",
    "hepatitis A": "Hepatitis A",
    "Chicken pox": "Chickenpox",
}

NEW_SYMPTOMS = ["ear_pain", "loss_of_taste", "difficulty_swallowing", "pale_skin"]

NEW_DISEASES: dict[str, dict] = {
    "Influenza": {
        "symptoms": ["high_fever", "chills", "cough", "headache", "muscle_pain", "fatigue",
                      "throat_irritation", "runny_nose", "sweating", "malaise"],
        "description": "Influenza (flu) is a contagious respiratory infection caused by influenza viruses. It usually starts suddenly with fever, aches and a dry cough, and most people recover within one to two weeks.",
        "precautions": ["rest and drink plenty of fluids", "take paracetamol for fever and aches",
                         "stay home to avoid spreading it", "consult a doctor if breathing becomes difficult"],
    },
    "COVID-19": {
        "symptoms": ["high_fever", "cough", "fatigue", "loss_of_smell", "loss_of_taste", "breathlessness",
                      "throat_irritation", "headache", "muscle_pain", "chills", "diarrhoea"],
        "description": "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Symptoms range from mild cold-like illness to severe pneumonia; loss of smell or taste is a distinctive early sign.",
        "precautions": ["isolate and take a test", "rest and stay hydrated", "monitor oxygen levels and breathing",
                         "seek urgent care if breathing is difficult"],
    },
    "Sinusitis": {
        "symptoms": ["sinus_pressure", "headache", "congestion", "runny_nose", "pain_behind_the_eyes",
                      "cough", "mild_fever", "phlegm", "fatigue", "loss_of_smell"],
        "description": "Sinusitis is inflammation of the sinuses, usually after a cold or because of allergies. It causes facial pressure, a blocked or runny nose and thick mucus, and most cases clear within a few weeks.",
        "precautions": ["use steam inhalation or saline rinses", "take decongestants or pain relief",
                         "drink warm fluids", "consult a doctor if symptoms last over ten days"],
    },
    "Food poisoning": {
        "symptoms": ["vomiting", "diarrhoea", "nausea", "abdominal_pain", "stomach_pain", "cramps",
                      "mild_fever", "dehydration", "loss_of_appetite", "fatigue", "chills"],
        "description": "Food poisoning is illness caused by eating food contaminated with bacteria, viruses or toxins. It typically begins within hours of eating and causes vomiting, diarrhoea and stomach cramps.",
        "precautions": ["sip oral rehydration solution", "eat bland food once vomiting stops",
                         "avoid dairy and fatty food", "consult a doctor if there is blood in stool or signs of dehydration"],
    },
    "Anaemia": {
        "symptoms": ["fatigue", "pale_skin", "breathlessness", "dizziness", "fast_heart_rate", "palpitations",
                      "headache", "cold_hands_and_feets", "brittle_nails", "lack_of_concentration"],
        "description": "Anaemia is a shortage of healthy red blood cells or haemoglobin, so the body receives less oxygen. Iron deficiency is the most common cause; it leads to tiredness, pallor and breathlessness on exertion.",
        "precautions": ["eat iron rich food such as leafy greens and legumes", "take iron supplements if prescribed",
                         "get a blood test to find the cause", "consult a doctor"],
    },
    "Conjunctivitis": {
        "symptoms": ["redness_of_eyes", "watering_from_eyes", "itching", "visual_disturbances",
                      "pain_behind_the_eyes", "yellow_crust_ooze", "puffy_face_and_eyes"],
        "description": "Conjunctivitis (pink eye) is inflammation of the thin membrane covering the eye and inner eyelid. It is caused by infection or allergy and makes the eye red, itchy, watery and sometimes sticky.",
        "precautions": ["clean the eyelids with cooled boiled water", "avoid touching or rubbing the eyes",
                         "do not share towels", "consult a doctor if vision is affected"],
    },
    "Tonsillitis": {
        "symptoms": ["throat_irritation", "difficulty_swallowing", "high_fever", "patches_in_throat",
                      "swelled_lymph_nodes", "headache", "ear_pain", "loss_of_appetite", "chills", "fatigue"],
        "description": "Tonsillitis is inflammation of the tonsils, usually from a viral or bacterial infection. It causes a sore throat, painful swallowing, fever and swollen glands in the neck.",
        "precautions": ["gargle with warm salt water", "take pain relief and rest", "drink plenty of fluids",
                         "consult a doctor if white patches or high fever persist"],
    },
    "Ear infection": {
        "symptoms": ["ear_pain", "high_fever", "headache", "dizziness", "irritability", "loss_of_balance",
                      "congestion", "runny_nose", "mild_fever"],
        "description": "A middle ear infection (otitis media) is inflammation behind the eardrum, often following a cold. It causes earache, fever and sometimes dizziness or temporary hearing loss.",
        "precautions": ["take pain relief", "apply a warm compress to the ear", "keep the ear dry",
                         "consult a doctor if pain lasts over two days or fluid drains from the ear"],
    },
    "Kidney stones": {
        "symptoms": ["back_pain", "abdominal_pain", "burning_micturition", "spotting_urination", "nausea",
                      "vomiting", "dark_urine", "continuous_feel_of_urine", "foul_smell_of_urine", "mild_fever"],
        "description": "Kidney stones are hard deposits of minerals that form in the kidneys. Passing a stone causes severe pain in the back or side that spreads to the lower abdomen, often with blood in the urine.",
        "precautions": ["drink plenty of water", "take prescribed pain relief", "strain urine to catch the stone",
                         "consult a doctor urgently if there is fever or you cannot pass urine"],
    },
    "Appendicitis": {
        "symptoms": ["abdominal_pain", "belly_pain", "nausea", "vomiting", "loss_of_appetite", "mild_fever",
                      "constipation", "distention_of_abdomen", "high_fever"],
        "description": "Appendicitis is inflammation of the appendix. Pain usually starts around the navel and moves to the lower right abdomen within hours. It is a surgical emergency because the appendix can burst.",
        "precautions": ["go to the emergency department immediately", "do not eat or drink",
                         "do not take laxatives or apply heat", "do not delay treatment"],
    },
    "Measles": {
        "symptoms": ["high_fever", "skin_rash", "red_spots_over_body", "cough", "runny_nose", "redness_of_eyes",
                      "watering_from_eyes", "malaise", "loss_of_appetite", "fatigue"],
        "description": "Measles is a highly contagious viral infection. It begins like a cold with high fever, cough and red eyes, followed by a blotchy red rash that spreads from the face down the body.",
        "precautions": ["isolate to avoid spreading it", "rest and drink fluids", "take paracetamol for fever",
                         "consult a doctor and check vaccination status"],
    },
    "Heat exhaustion": {
        "symptoms": ["sweating", "dizziness", "headache", "nausea", "fatigue", "dehydration", "cramps",
                      "fast_heart_rate", "high_fever", "pale_skin", "vomiting"],
        "description": "Heat exhaustion happens when the body overheats, usually after exertion in hot weather. It causes heavy sweating, dizziness, headache and nausea, and can progress to heat stroke if not treated.",
        "precautions": ["move to a cool place and lie down", "sip water or oral rehydration solution",
                         "cool the skin with wet cloths", "seek emergency care if confusion or a very high temperature develops"],
    },
    "Anxiety disorder": {
        "symptoms": ["anxiety", "restlessness", "palpitations", "fast_heart_rate", "sweating", "irritability",
                      "lack_of_concentration", "breathlessness", "dizziness", "shivering", "nausea"],
        "description": "Generalised anxiety disorder involves persistent, excessive worry that is hard to control. It often comes with restlessness, a racing heart, sweating and trouble concentrating or sleeping.",
        "precautions": ["practise slow breathing and regular exercise", "limit caffeine and alcohol",
                         "keep a regular sleep routine", "consult a doctor or therapist"],
    },
}


def clean_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns=COLUMN_FIXES)
    df = df.loc[:, ~df.columns.duplicated()]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().replace(DISEASE_RENAMES)
    return df


def variants(symptoms: list[str], rng: random.Random, count: int) -> list[set[str]]:
    rows = [set(symptoms)]
    optional = symptoms[MIN_KEEP:]
    seen = {frozenset(symptoms)}
    attempts = 0
    while len(rows) < count and attempts < MAX_ATTEMPTS:
        attempts += 1
        drop = rng.randint(1, max(1, len(optional) - 2))
        kept = set(symptoms[:MIN_KEEP]) | set(rng.sample(optional, len(optional) - drop))
        if frozenset(kept) not in seen:
            seen.add(frozenset(kept))
            rows.append(kept)
    return rows


def extend(df: pd.DataFrame, rng: random.Random, per_disease: int) -> pd.DataFrame:
    columns = [c for c in df.columns if c != TARGET_COLUMN] + NEW_SYMPTOMS
    zeros = pd.DataFrame(0, index=df.index, columns=NEW_SYMPTOMS)
    df = pd.concat([df.drop(columns=TARGET_COLUMN), zeros, df[TARGET_COLUMN]], axis=1)

    unknown = {s for d in NEW_DISEASES.values() for s in d["symptoms"]} - set(columns)
    if unknown:
        raise SystemExit(f"Unknown symptoms in NEW_DISEASES: {sorted(unknown)}")

    new_rows = []
    for disease, spec in NEW_DISEASES.items():
        for present in variants(spec["symptoms"], rng, per_disease):
            row = {c: int(c in present) for c in columns}
            row[TARGET_COLUMN] = disease
            new_rows.append(row)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def write_lookup(name: str, extra: dict[str, list[str]]) -> None:
    rows: dict[str, list[str]] = {}
    with open(RAW_DIR / name, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            disease = DISEASE_RENAMES.get(row[0].strip(), row[0].strip())
            rows[disease] = [c.strip() for c in row[1:] if c.strip()]
    rows.update(extra)
    with open(DATA_DIR / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for disease in sorted(rows):
            writer.writerow([disease, *rows[disease]])


def main() -> None:
    rng = random.Random(SEED)
    train = clean_frame(RAW_DIR / "train.csv").drop_duplicates().reset_index(drop=True)
    test = clean_frame(RAW_DIR / "test.csv")
    per_disease = int(round(train.groupby(TARGET_COLUMN).size().mean()))

    train = extend(train, rng, per_disease).drop_duplicates().sort_values(TARGET_COLUMN, kind="stable")
    test = extend(test, rng, 1)

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    write_lookup("description.csv", {d: [s["description"]] for d, s in NEW_DISEASES.items()})
    write_lookup("precaution.csv", {d: s["precautions"] for d, s in NEW_DISEASES.items()})

    print(f"train: {len(train)} rows, {train[TARGET_COLUMN].nunique()} diseases, {len(train.columns) - 1} symptoms")
    print(f"test:  {len(test)} rows")


if __name__ == "__main__":
    main()

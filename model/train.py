import time

from .hiro.classifiers import DiseaseModels
from .hiro.constants import DISEASE_MODELS, INTENT_MODEL, MODEL_DIR
from .hiro.intents import IntentClassifier

SAMPLES = ("hi there", "who made you", "thanks a lot", "my head hurts", "asdfgh")


def main() -> None:
    started = time.perf_counter()
    root = MODEL_DIR.parent

    disease = DiseaseModels.train(evaluate=True)
    disease.save()
    print(f"Disease models -> {DISEASE_MODELS.relative_to(root)}")
    print(f"  {len(disease.classes)} diseases, {len(disease.columns)} symptoms")
    for name, score in disease.scores.items():
        print(f"  CV   {name:<13} {score:.3f}")
    for name, score in disease.evaluate().items():
        print(f"  test {name:<13} {score:.3f}")

    intents = IntentClassifier.train()
    intents.save()
    print(f"Intent model -> {INTENT_MODEL.relative_to(root)}")
    print(f"  {len(intents.tags)} intents")
    for sample in SAMPLES:
        tag, confidence = intents.classify(sample)
        print(f"  {sample!r:<16} -> {tag} ({confidence:.2f})")

    print(f"Done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

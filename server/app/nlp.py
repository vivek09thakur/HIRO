"""Rule-based symptom extraction: aliases, phrase reordering, fuzzy matching and negation."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.86
FUZZY_MIN_LABEL_LEN = 5
NEGATION_WINDOW = 4

NEGATION_CUES = {
    "no", "not", "without", "dont", "don't", "never", "neither", "nor", "havent", "haven't",
    "isnt", "isn't", "aint", "ain't", "didnt", "didn't", "cant", "can't", "hardly", "zero",
}
NEGATION_FILLERS = {
    "have", "got", "feel", "feeling", "any", "much", "a", "an", "the", "of", "really", "been",
    "experiencing", "experience", "had", "or", "some", "more", "longer", "at", "all", "sign", "signs",
}
STOPWORDS = {
    "i", "a", "an", "the", "and", "or", "my", "me", "is", "am", "are", "have", "has", "had", "with",
    "of", "in", "on", "at", "to", "for", "it", "its", "this", "that", "very", "also", "too", "so",
    "but", "yes", "no", "not", "feel", "feeling", "like", "get", "got", "some", "lot", "lots",
}

BODY_PARTS = {
    "stomach": "stomach pain",
    "tummy": "stomach pain",
    "belly": "belly pain",
    "back": "back pain",
    "chest": "chest pain",
    "knee": "knee pain",
    "knees": "knee pain",
    "neck": "neck pain",
    "joint": "joint pain",
    "joints": "joint pain",
    "hip": "hip joint pain",
    "hips": "hip joint pain",
    "abdomen": "abdominal pain",
    "muscle": "muscle pain",
    "muscles": "muscle pain",
    "head": "headache",
}
_PARTS = "|".join(sorted(BODY_PARTS, key=len, reverse=True))
_PAIN = r"(?:pain|pains|ache|aches|aching|hurt|hurts|hurting|sore|soreness|cramping)"
BODY_PART_PATTERNS = [
    re.compile(rf"\b{_PAIN} (?:in|on|at|around|across) (?:my |the |both |lower |upper )?({_PARTS})\b"),
    re.compile(rf"\b(?:my |the )?(?:lower |upper )?({_PARTS}) (?:is |are |has been |have been |keeps? )?(?:hurt|hurts|hurting|aches|aching|paining|painful|sore|killing me)\b"),
]

# (regex, canonical label). Longest match wins when spans overlap.
ALIASES: list[tuple[str, str]] = [
    # fever
    (r"(?:very |really |extremely )?(?:high|severe|strong|bad|burning|raging|10[2-6](?: ?degrees?| ?f)?) (?:degree )?(?:fever|temperature|temp)", "high fever"),
    (r"(?:mild|slight|low|low grade|little|light|minor) (?:fever|temperature|temp)", "mild fever"),
    (r"fever(?:s|ish)?|feverish|febrile|pyrexia|(?:running a |have a |got a )?temperature", "high fever"),
    # digestive
    (r"diarrh?(?:ea|oea)|loose motions?|loose stools?|watery stools?|the runs|dysentery", "diarrhoea"),
    (r"throwing up|threw up|throw up|puk(?:e|ed|ing)|vomit(?:s|ed|ting|ing)?|emesis", "vomiting"),
    (r"nauseous|nauseated|queasy|queasiness|feel(?:ing)? sick|sick to (?:my|the) stomach", "nausea"),
    (r"stomach ?aches?|tummy ?aches?|belly ?aches?|gastric pain", "stomach pain"),
    (r"heart ?burn|acid reflux|reflux|acidic|sour burps|acidity", "acidity"),
    (r"indigestion|dyspepsia|upset stomach", "indigestion"),
    (r"bloat(?:ed|ing)|distended (?:abdomen|stomach|belly)|distension|distention", "distention of abdomen"),
    (r"(?:swelling|swollen) (?:of |in )?(?:my |the )?(?:stomach|abdomen)|abdominal swelling|stomach (?:is )?swollen", "swelling of stomach"),
    (r"gassy|farting|flatulence|passing (?:a lot of )?(?:gas|wind)|lots of gas|gas problem", "passage of gases"),
    (r"constipated|constipation|can'?t poop|hard stools?|no bowel movements?", "constipation"),
    (r"blood(?:y)? (?:in )?(?:my )?(?:stool|stools|poop)|bloody stools?|rectal bleeding", "bloody stool"),
    (r"pain(?:ful)? (?:when|while|during) (?:pooping|passing stool|bowel movements?)|painful bowel movements?", "pain during bowel movements"),
    (r"(?:pain|hurts?|burning) (?:in|around|near) (?:my |the )?(?:anus|anal region|anal area|rectum|bottom)", "pain in anal region"),
    (r"itch(?:y|ing|es)? (?:in|around|near) (?:my |the )?(?:anus|anal region|anal area|bottom)|anal itching", "irritation in anus"),
    (r"no appetite|lost (?:my )?appetite|not hungry|don'?t feel like eating|poor appetite|appetite loss|can'?t eat", "loss of appetite"),
    (r"(?:very|really|always|constantly|extremely) hungry|increased (?:hunger|appetite)|hungry all the time|eating (?:a lot|too much)", "excessive hunger"),
    (r"lost weight|losing weight|weight (?:is )?dropping|weight dropped", "weight loss"),
    (r"gained weight|gaining weight|putting on weight|put on weight", "weight gain"),
    (r"vomiting blood|bleeding (?:in|from) (?:my |the )?stomach|stomach bleeding", "stomach bleeding"),
    (r"mouth ulcers?|ulcers? (?:on|in) (?:my |the )?(?:tongue|mouth)|canker sores?", "ulcers on tongue"),
    # respiratory
    (r"short(?:ness)? of breath|out of breath|breathless(?:ness)?|difficulty breathing|trouble breathing|hard to breathe|can'?t breathe|cannot breathe|breathing (?:difficulty|difficulties|problems?|trouble)|wheez(?:e|ing)", "breathlessness"),
    (r"running nose|runny nose|nose (?:is |keeps )?running|nasal discharge", "runny nose"),
    (r"blocked nose|stuffy nose|stuffed nose|nasal congestion|congested|congestion", "congestion"),
    (r"sore throat|throat (?:pain|ache|is sore|irritation|hurts)|scratchy throat|itchy throat", "throat irritation"),
    (r"white patches (?:in|on) (?:my |the )?throat|patches in (?:my |the )?throat", "patches in throat"),
    (r"sneez(?:e|es|ing)|continuous sneezing", "continuous sneezing"),
    (r"cough(?:s|ing|ed)?", "cough"),
    (r"coughing (?:up )?blood|blood (?:in|when) (?:i )?(?:cough|sputum|phlegm)|hemoptysis|haemoptysis", "blood in sputum"),
    (r"rusty (?:coloured |colored )?(?:sputum|phlegm)|rust coloured (?:sputum|phlegm)", "rusty sputum"),
    (r"thick (?:sputum|phlegm|mucus)|mucoid sputum", "mucoid sputum"),
    (r"phlegm|mucus|sputum", "phlegm"),
    (r"sinus (?:pressure|pain|headache)|pressure (?:in|around) (?:my |the )?sinuses|sinusitis|sinus", "sinus pressure"),
    (r"chest tightness|tight chest|tightness in (?:my |the )?chest|pressure in (?:my |the )?chest|chest (?:feels|is) (?:tight|heavy)|heaviness in (?:my |the )?chest", "chest pain"),
    # general
    (r"chill(?:s|y|ed)?|feel(?:ing)? cold|cold shivers", "chills"),
    (r"shiver(?:s|ing|y)?|trembling|shaking", "shivering"),
    (r"tired(?:ness)?|exhaust(?:ed|ion)|fatigued?|worn out|no energy|drained|sluggish|weak(?:ness)?", "fatigue"),
    (r"lethargic|lethargy|listless", "lethargy"),
    (r"sweaty|sweats|sweating|perspir(?:ing|ation)|night sweats|drenched", "sweating"),
    (r"dizzy|dizziness|light ?headed(?:ness)?|giddy|giddiness|vertigo|woozy", "dizziness"),
    (r"head ?aches?|migraines?|head (?:is )?(?:pounding|throbbing|hurting|hurts)", "headache"),
    (r"malaise|generally unwell|feel(?:ing)? unwell|under the weather|not feeling well|feel(?:ing)? (?:ill|awful|terrible)", "malaise"),
    (r"dehydrat(?:ed|ion)|very thirsty|dry mouth|parched", "dehydration"),
    (r"body ?aches?|body pain|aching all over|everything hurts|muscle aches?|sore muscles", "muscle pain"),
    (r"joint aches?|aching joints|joints? (?:ache|aching)|arthralgia", "joint pain"),
    (r"back ?aches?", "back pain"),
    (r"neck ?aches?", "neck pain"),
    (r"knee aches?", "knee pain"),
    (r"cramp(?:s|ing|ed)?|muscle spasms?", "cramps"),
    (r"bruis(?:e|es|ed|ing)|black and blue", "bruising"),
    (r"muscle (?:wasting|loss)|losing muscle|muscles? (?:are |is )?shrinking", "muscle wasting"),
    (r"stiff(?:ness)?|hard to move|can'?t move (?:properly|well)|movement stiffness", "movement stiffness"),
    (r"stiff neck|neck (?:is |feels )?stiff", "stiff neck"),
    (r"hurts to walk|pain(?:ful)? (?:when|while) walking|limping|limp|can'?t walk (?:properly|well)", "painful walking"),
    # skin
    (r"itch(?:y|es|ing|iness)?|pruritus|scratching", "itching"),
    (r"rash(?:es)?|hives|urticaria|skin (?:eruptions?|rashes|breaking out)", "skin rash"),
    (r"bumps? on (?:my |the )?skin|skin bumps|nodules?|lumps? (?:on|under) (?:my |the )?skin", "nodal skin eruptions"),
    (r"pimples?|acne|zits?|pustules?|boils?", "pus filled pimples"),
    (r"blackheads?", "blackheads"),
    (r"scar(?:s|ring)?|scurring", "scurring"),
    (r"skin (?:is )?peeling|peeling skin|flaky skin|flaking skin|scaly skin", "skin peeling"),
    (r"silver(?:y)? (?:scales|dusting|flakes|patches)", "silver like dusting"),
    (r"discolou?red patches|skin discolou?ration|patches on (?:my |the )?skin|dischromic patches|white patches on (?:my |the )?skin", "dischromic patches"),
    (r"red spots|spots (?:all )?over (?:my |the )?body|red dots|petechiae", "red spots over body"),
    (r"blisters?", "blister"),
    (r"yellow crust(?:s|ing)?|crusty (?:skin|sores?)|oozing|weeping sores?", "yellow crust ooze"),
    (r"yellow(?:ish)? skin|skin (?:is |looks )?yellow|jaundiced?", "yellowish skin"),
    (r"yellow(?:ish)? (?:eyes|sclera)|eyes (?:are |look )?yellow|whites of (?:my )?eyes (?:are )?yellow", "yellowing of eyes"),
    (r"dents in (?:my )?nails|nail pitting|pitted nails|small dents in nails", "small dents in nails"),
    (r"brittle nails|nails (?:are )?brittle|nails (?:keep )?break(?:ing)?|nails (?:are )?cracking", "brittle nails"),
    (r"inflamed nails|nail inflammation|swollen nails|inflammatory nails", "inflammatory nails"),
    (r"dry lips|chapped lips|tingling lips|lips (?:are )?(?:dry|tingling|cracked)", "drying and tingling lips"),
    # eyes
    (r"blurr(?:y|ed) vision|vision (?:is )?blurr(?:y|ed)|can'?t see (?:clearly|properly)|double vision|distorted vision", "blurred and distorted vision"),
    (r"red eyes?|eyes (?:are )?red|bloodshot(?: eyes)?|redness (?:of|in) (?:my |the )?eyes?", "redness of eyes"),
    (r"watery eyes?|eyes (?:are |keep )?watering|tearing (?:up)?|watering from (?:my |the )?eyes", "watering from eyes"),
    (r"pain behind (?:my |the )?eyes?|eye ?aches?|eyes? (?:hurt|hurts|ache)", "pain behind the eyes"),
    (r"sunken eyes|eyes (?:look |are )?sunken|hollow eyes", "sunken eyes"),
    (r"puffy (?:face|eyes)|face (?:is )?puffy|swollen face|face (?:is )?swollen|puffy face and eyes", "puffy face and eyes"),
    (r"visual disturbances?|seeing spots|flashes of light|floaters", "visual disturbances"),
    (r"lost (?:my )?(?:sense of )?smell|can'?t smell|no sense of smell|anosmia|loss of smell", "loss of smell"),
    # swelling
    (r"swollen (?:lymph )?(?:nodes|glands)|swelled lymph nodes|lumps? in (?:my |the )?(?:neck|armpit|groin)", "swelled lymph nodes"),
    (r"swollen legs?|legs? (?:are |is )?swollen|leg swelling", "swollen legs"),
    (r"swollen joints?|joints? (?:are |is )?swollen|joint swelling|swelling joints", "swelling joints"),
    (r"swollen (?:hands|feet|ankles|extremities|fingers|toes)|(?:hands|feet|ankles) (?:are )?swollen|swollen extremities", "swollen extremeties"),
    (r"swollen (?:blood )?vessels|bulging veins|veins (?:are )?swollen", "swollen blood vessels"),
    (r"visible veins|prominent veins|veins (?:on|in) (?:my |the )?(?:calf|calves|legs?)|varicose veins", "prominent veins on calf"),
    (r"enlarged thyroid|goitre|goiter|swollen thyroid|thyroid (?:is )?(?:enlarged|swollen)|lump in (?:my |the )?throat", "enlarged thyroid"),
    (r"fluid (?:overload|retention)|retaining (?:fluid|water)|water retention", "fluid overload"),
    (r"obese|overweight|obesity", "obesity"),
    (r"cold (?:hands|feet)|hands and feet (?:are )?cold|cold hands and feet", "cold hands and feets"),
    # urinary
    (r"burning (?:sensation )?(?:when|while|during|on) (?:peeing|urinating|urination|passing urine)|burning (?:pee|urination|micturition)|painful urination|pain(?:ful)? (?:when|while) (?:peeing|urinating)|dysuria|burning micturition", "burning micturition"),
    (r"peeing (?:a lot|often|frequently|too much|constantly)|frequent urination|urinating (?:a lot|often|frequently)|excessive urination|polyuria", "polyuria"),
    (r"constant(?:ly)? (?:urge|need|feel) to (?:pee|urinate)|feel(?:ing)? like (?:peeing|urinating) all the time|always need to pee|continuous feel of urine", "continuous feel of urine"),
    (r"dark (?:coloured |colored )?(?:urine|pee)|brown (?:urine|pee)|cola coloured urine|tea coloured urine|urine (?:is )?dark", "dark urine"),
    (r"yellow (?:urine|pee)|urine (?:is )?(?:very )?yellow", "yellow urine"),
    (r"foul smelling (?:urine|pee)|smelly (?:urine|pee)|urine smells|bad smelling urine|foul smell of urine", "foul smell of urine"),
    (r"bladder (?:pain|discomfort|pressure)|pressure in (?:my |the )?bladder", "bladder discomfort"),
    (r"spotting (?:when |while |during )?(?:peeing|urinating|urination)|blood (?:in|when) (?:i )?(?:pee|urinate|urine)|spotting urination", "spotting urination"),
    # heart
    (r"palpitations?|heart (?:is )?(?:racing|pounding|fluttering|skipping)|racing heart|pounding heart", "palpitations"),
    (r"fast heart ?beat|fast heart ?rate|rapid heart ?(?:beat|rate)|heart (?:is )?beating (?:fast|quickly)|tachycardia", "fast heart rate"),
    # neuro / mental
    (r"anxious|anxiety|nervous(?:ness)?|panick?(?:y|ing|ed)?|panic attacks?|worried all the time", "anxiety"),
    (r"depress(?:ed|ion|ive)|feel(?:ing)? (?:low|down|hopeless|empty)|sad all the time|no interest in anything", "depression"),
    (r"irritable|irritability|short tempered|cranky|snappy|easily annoyed", "irritability"),
    (r"mood swings?|moody|mood (?:keeps )?changing", "mood swings"),
    (r"restless(?:ness)?|can'?t sit still|fidgety|agitated", "restlessness"),
    (r"can'?t (?:focus|concentrate)|lack of (?:focus|concentration)|trouble concentrating|poor concentration|hard to concentrate", "lack of concentration"),
    (r"slurred speech|slurring(?: my words)?|speech (?:is )?slurred|can'?t speak (?:clearly|properly)", "slurred speech"),
    (r"spinning|room (?:is )?spinning|spinning sensation|spinning movements", "spinning movements"),
    (r"lose (?:my )?balance|losing (?:my )?balance|off balance|loss of balance|keep falling", "loss of balance"),
    (r"unsteady|unsteadiness|wobbly|staggering", "unsteadiness"),
    (r"numb(?:ness)?|one side (?:is |feels )?(?:weak|numb)|weakness (?:on|of) one side|weakness of one body side|face (?:is )?drooping|facial droop", "weakness of one body side"),
    (r"weak(?:ness)? in (?:my |the )?(?:limbs|arms|legs|arms and legs)|limbs (?:are |feel )?weak|weakness in limbs", "weakness in limbs"),
    (r"muscle weakness|muscles (?:are |feel )?weak", "muscle weakness"),
    (r"altered (?:sensorium|consciousness)|confus(?:ed|ion)|disoriented|disorientation|not thinking clearly", "altered sensorium"),
    (r"coma|comatose|unconscious|passed out|blacked out|fainted|fainting", "coma"),
    # endocrine / misc
    (r"high (?:blood )?sugar|sugar (?:level )?(?:is )?(?:high|irregular|fluctuating|unstable)|blood sugar (?:problems?|issues?)|irregular sugar level|diabetic", "irregular sugar level"),
    (r"irregular periods?|missed periods?|periods? (?:is |are )?irregular|menstrual (?:problems?|irregularit(?:y|ies))|heavy periods?|abnormal menstruation|no periods?", "abnormal menstruation"),
    (r"liver failure|acute liver failure", "acute liver failure"),
    (r"blood transfusion|transfused|received blood|receiving blood transfusion", "receiving blood transfusion"),
    (r"unsterile (?:injections?|needles?)|shared needles?|dirty needles?|reused needles?|receiving unsterile injections", "receiving unsterile injections"),
    (r"family history|runs in (?:my |the )?family|hereditary|genetic|my (?:mother|father|parents|brother|sister) (?:has|had|have) it", "family history"),
    (r"drink(?:ing)? (?:a lot|heavily|too much|alcohol)|alcohol(?:ic|ism)?|heavy drinker|history of alcohol consumption", "history of alcohol consumption"),
    (r"unprotected sex|multiple (?:sexual )?partners|extra ?marital|affairs?|extra marital contacts", "extra marital contacts"),
    (r"toxic look|look toxic|typhos", "toxic look (typhos)"),
    (r"internal itching|itching inside", "internal itching"),
]

NGRAM_STOP = STOPWORDS | {"pain", "feel", "have", "very"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    priority: int  # higher wins on ties


class SymptomExtractor:
    def __init__(self, labels: dict[str, str]) -> None:
        self.key_of = {label: key for key, label in labels.items()}
        self.labels = sorted(self.key_of, key=len, reverse=True)
        self.label_patterns = [(re.compile(rf"\b{re.escape(label)}\b"), label) for label in self.labels]
        self.alias_patterns = [
            (re.compile(rf"\b(?:{pattern})\b"), label) for pattern, label in ALIASES if label in self.key_of
        ]
        self.fuzzy_labels = [label for label in self.labels if len(label) >= FUZZY_MIN_LABEL_LEN]

    # ---- public -------------------------------------------------------

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        """Return (reported_keys, denied_keys) in order of appearance."""
        tokens = list(re.finditer(r"[a-z0-9']+", text))
        spans = self._exact_spans(text) + self._alias_spans(text) + self._body_part_spans(text)
        spans += self._fuzzy_spans(text, tokens, {s.label for s in spans})
        chosen = self._resolve(spans)

        reported: list[str] = []
        denied: list[str] = []
        for span in chosen:
            key = self.key_of[span.label]
            target = denied if self._negated(span, tokens, text) else reported
            if key not in target:
                target.append(key)
        return [k for k in reported if k not in denied], denied

    # ---- candidate generation -----------------------------------------

    def _exact_spans(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), label, 3) for pattern, label in self.label_patterns for m in pattern.finditer(text)]

    def _alias_spans(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), label, 2) for pattern, label in self.alias_patterns for m in pattern.finditer(text)]

    def _body_part_spans(self, text: str) -> list[Span]:
        spans = []
        for pattern in BODY_PART_PATTERNS:
            for m in pattern.finditer(text):
                label = BODY_PARTS[m.group(1)]
                if label in self.key_of:
                    spans.append(Span(m.start(), m.end(), label, 2))
        return spans

    def _fuzzy_spans(self, text: str, tokens: list[re.Match], found: set[str]) -> list[Span]:
        spans = []
        words = [t.group() for t in tokens]
        for label in self.fuzzy_labels:
            if label in found:
                continue
            n = len(label.split())
            for i in range(len(words) - n + 1):
                chunk = words[i : i + n]
                if chunk[0] in NGRAM_STOP or chunk[-1] in NGRAM_STOP or min(len(w) for w in chunk) < 4:
                    continue
                candidate = " ".join(chunk)
                if SequenceMatcher(None, candidate, label).ratio() >= FUZZY_THRESHOLD:
                    spans.append(Span(tokens[i].start(), tokens[i + n - 1].end(), label, 1))
                    break
        return spans

    # ---- resolution ---------------------------------------------------

    @staticmethod
    def _resolve(spans: list[Span]) -> list[Span]:
        chosen: list[Span] = []
        for span in sorted(spans, key=lambda s: (-(s.end - s.start), -s.priority, s.start)):
            if all(span.end <= c.start or span.start >= c.end for c in chosen):
                chosen.append(span)
        return sorted(chosen, key=lambda s: s.start)

    @staticmethod
    def _negated(span: Span, tokens: list[re.Match], text: str) -> bool:
        before = [t.group() for t in tokens if t.end() <= span.start]
        for word in reversed(before[-NEGATION_WINDOW:]):
            if word in NEGATION_CUES:
                return True
            if word not in NEGATION_FILLERS:
                return False
        return False

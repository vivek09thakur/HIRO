import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODEL_DIR / "data"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
DESCRIPTION_CSV = DATA_DIR / "description.csv"
PRECAUTION_CSV = DATA_DIR / "precaution.csv"
INTENTS_JSON = DATA_DIR / "intents.json"

DISEASE_MODELS = ARTIFACTS_DIR / "disease_models.joblib"
INTENT_MODEL = ARTIFACTS_DIR / "intent_model.joblib"

TARGET_COLUMN = "prognosis"
MODEL_NAMES = ("Random Forest", "SVC", "Naive Bayes")
TIE_BREAKER = "Random Forest"
TOP_CANDIDATES = 3
RANDOM_STATE = 24
CV_FOLDS = 4
SCORE_TEMPERATURE = 10.0
RF_ESTIMATORS = 200
SVC_C = 1.0
NB_ALPHA = 0.1

INTENT_CONFIDENCE_THRESHOLD = 0.35
FALLBACK_TAG = "fallback"
INTENT_C = 20.0
INTENT_MAX_ITER = 2000
CHAR_NGRAMS = (2, 4)
WORD_NGRAMS = (1, 2)

INTRO_MESSAGE = (
    "Hello, I am HIRO, a symptom checker. Describe what you are feeling and I will ask a few "
    "follow-up questions before suggesting a possible condition. This is not a medical diagnosis. "
    "Say \"that's all\" at any time to see the result."
)
EMERGENCY_NOTICE = "If this is an emergency, stop here and call your local emergency number now. "
NO_SYMPTOMS_YET = "I have not recorded any symptoms yet. Tell me what you are feeling first."
DENIED_ONLY = "Noted that you do not have {symptoms}. What symptoms do you have?"
UNRECOGNISED = "I did not recognise any symptoms in that message."
NOTED_NEW = "I have noted {symptoms}."
NOTED_YES = "Noted, {symptom}."
NOTED_NO = "Okay, no {symptom}."
UNDERSTOOD = "Understood."
QUESTION = "{prefix} Symptoms recorded so far: {symptoms}. Are you also experiencing {asked}?"
NO_DESCRIPTION = "No description is available for this condition."

MIN_SYMPTOMS = 3
MAX_QUESTIONS = 5
MAX_CANDIDATES = 12
DENIED_PENALTY = 0.5
CONFIDENCE_MARGIN = 1.0
CANDIDATE_MARGIN = 0.5

YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "ya", "sure", "correct", "right", "true", "indeed", "affirmative", "absolutely", "definitely"}
NO_WORDS = {"no", "n", "nope", "nah", "not", "never", "none", "false", "negative"}
YES_PHRASES = ("i do", "i am", "i have", "i think so", "a bit", "a little", "sometimes", "kind of", "sort of")
NO_PHRASES = ("i don't", "i dont", "not really", "i do not", "i am not", "i haven't", "i havent", "not at all", "i'm not")
DONE_PATTERN = re.compile(
    r"\b(that'?s? (all|it)|that is (all|it)|nothing else|no more|no other|"
    r"done|finish(ed)?|enough|only (that|this|these)|what do i have|what is it|tell me the result|show (me )?(the )?result)\b"
)
EMERGENCY_PATTERN = re.compile(
    r"\b(can'?t breathe|cannot breathe|not breathing|unconscious|unresponsive|passed out|collapsed|seizure|"
    r"heavy bleeding|bleeding heavily|won'?t stop bleeding|overdose|suicid\w*|kill myself|hurt myself|"
    r"stroke|heart attack|crushing chest pain|severe chest pain|chest pain and (breathless|sweating|can'?t breathe)|"
    r"anaphyla\w*|throat (is )?closing|blue lips|emergency)\b"
)

FUZZY_THRESHOLD = 0.86
FUZZY_MIN_LABEL_LEN = 5
SPELL_THRESHOLD = 0.84
SPELL_MIN_LEN = 4
NEGATION_WINDOW = 4
POST_NEGATION_WINDOW = 4

NEGATION_CUES = {
    "no", "not", "without", "dont", "don't", "never", "neither", "nor", "havent", "haven't",
    "isnt", "isn't", "aint", "ain't", "didnt", "didn't", "cant", "can't", "hardly", "zero", "free",
    "denies", "deny", "lack", "lacking", "nothing",
}
NEGATION_FILLERS = {
    "have", "got", "feel", "feeling", "any", "much", "a", "an", "the", "of", "really", "been",
    "experiencing", "experience", "had", "or", "some", "more", "longer", "at", "all", "sign", "signs",
    "my", "i", "am", "do", "does", "actually", "currently", "anymore", "even", "with",
}
POST_NEGATION = re.compile(
    r"^(?:is |has |have |are |was |were |had |finally |now |already |completely )*"
    r"(?:gone|stopped|subsided|disappeared|cleared(?: up)?|resolved|better now|went away|"
    r"is over|has passed|passed|faded|improved|no more|no longer)\b"
)
COORDINATORS = {"or", "and", "nor", "neither", "either"}
STOPWORDS = {
    "i", "a", "an", "the", "and", "or", "my", "me", "is", "am", "are", "have", "has", "had", "with",
    "of", "in", "on", "at", "to", "for", "it", "its", "this", "that", "very", "also", "too", "so",
    "but", "yes", "no", "not", "feel", "feeling", "like", "get", "got", "some", "lot", "lots",
}
NGRAM_STOP = STOPWORDS | {"pain", "feel", "have", "very"}
WORD_PATTERN = re.compile(r"[a-z0-9']+")

NUMBER_WORDS = r"\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of|several|many|half a"
TIME_UNITS = r"days?|weeks?|months?|years?|hours?|hrs?|nights?|mornings?|minutes?|mins?"
DURATION_PATTERNS = [
    re.compile(
        rf"\b(?:for|since|from|past|last|over the past|over the last|in the last|about|around|nearly|almost)?\s*"
        rf"(?:the )?(?:{NUMBER_WORDS})\s*(?:{TIME_UNITS})\b(?:\s+(?:now|ago|back|straight|already))?"
    ),
    re.compile(
        r"\b(?:since|from|starting|started|began|beginning)\s+(?:last\s+|this\s+|the\s+)?"
        r"(?:yesterday|today|tonight|morning|afternoon|evening|night|week|weekend|month|monday|tuesday|"
        r"wednesday|thursday|friday|saturday|sunday)\b"
    ),
    re.compile(r"\b(?:all day|all night|every day|everyday|every night|on and off|off and on|constantly|nonstop|non stop)\b"),
]

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
    "ear": "ear pain",
    "ears": "ear pain",
    "throat": "throat irritation",
    "eye": "pain behind the eyes",
    "eyes": "pain behind the eyes",
    "bladder": "bladder discomfort",
    "anus": "pain in anal region",
}
_PARTS = "|".join(sorted(BODY_PARTS, key=len, reverse=True))
_PAIN = r"(?:pain|pains|ache|aches|aching|hurt|hurts|hurting|sore|soreness|cramping|discomfort|burning)"
_DET = r"(?:my |the |both |lower |upper |left |right |whole |entire )*"
_MORE = rf"(?:(?:,| and| or| &|, and) {_DET}(?:{_PARTS}))*"
BODY_PART_WORD = re.compile(rf"\b({_PARTS})\b")
BODY_PART_PATTERNS = [
    re.compile(rf"\b{_PAIN} (?:in|on|at|around|across|near|behind) {_DET}(?:{_PARTS}){_MORE}\b"),
    re.compile(
        rf"\b{_DET}(?:{_PARTS}){_MORE} (?:is |are |has been |have been |keeps? |both |really |still )*"
        rf"(?:hurt|hurts|hurting|aches|aching|ache|paining|painful|sore|killing me|throbbing|pounding|burning)\b"
    ),
]

ALIASES: list[tuple[str, str]] = [
    (r"(?:very |really |extremely )?(?:high|severe|strong|bad|burning|raging|10[2-6](?: ?degrees?| ?f)?) (?:degree )?(?:fever|temperature|temp)", "high fever"),
    (r"(?:mild|slight|low|low grade|little|light|minor|small) (?:fever|temperature|temp)", "mild fever"),
    (r"fever(?:s|ish)?|feverish|febrile|pyrexia|(?:running a |have a |got a |got )?temperature(?: of \d+)?|(?:feel(?:ing)? )?hot and cold|burning up", "high fever"),
    (r"diarrh?(?:ea|oea)|loose motions?|loose stools?|watery stools?|the runs|dysentery|upset tummy", "diarrhoea"),
    (r"throwing up|threw up|throw up|puk(?:e|ed|ing)|vomit(?:s|ed|ting|ing)?|emesis|been sick", "vomiting"),
    (r"nauseous|nauseated|queasy|queasiness|feel(?:ing)? sick|sick to (?:my|the) stomach|want to (?:vomit|throw up)|gagging", "nausea"),
    (r"stomach ?aches?|tummy ?aches?|belly ?aches?|gastric pain", "stomach pain"),
    (r"heart ?burn|acid reflux|reflux|acidic|sour burps|acidity|burning in (?:my |the )?chest after eating", "acidity"),
    (r"indigestion|dyspepsia|upset stomach|stomach upset", "indigestion"),
    (r"bloat(?:ed|ing)|distended (?:abdomen|stomach|belly)|distension|distention|feel(?:ing)? full", "distention of abdomen"),
    (r"(?:swelling|swollen) (?:of |in )?(?:my |the )?(?:stomach|abdomen)|abdominal swelling|stomach (?:is )?swollen", "swelling of stomach"),
    (r"gassy|farting|flatulence|passing (?:a lot of )?(?:gas|wind)|lots of gas|gas problem|burping a lot", "passage of gases"),
    (r"constipated|constipation|can'?t poop|hard stools?|no bowel movements?|haven'?t pooped", "constipation"),
    (r"blood(?:y)? (?:in )?(?:my )?(?:stool|stools|poop)|bloody stools?|rectal bleeding|bleeding when (?:i )?poop", "bloody stool"),
    (r"pain(?:ful)? (?:when|while|during) (?:pooping|passing stool|bowel movements?)|painful bowel movements?|hurts to poop", "pain during bowel movements"),
    (r"(?:pain|hurts?|burning) (?:in|around|near) (?:my |the )?(?:anus|anal region|anal area|rectum|bottom)|piles", "pain in anal region"),
    (r"itch(?:y|ing|es)? (?:in|around|near) (?:my |the )?(?:anus|anal region|anal area|bottom)|anal itching", "irritation in anus"),
    (r"no appetite|lost (?:my )?appetite|not hungry|don'?t feel like eating|poor appetite|appetite loss|can'?t eat|not eating|off my food", "loss of appetite"),
    (r"(?:very|really|always|constantly|extremely) hungry|increased (?:hunger|appetite)|hungry all the time|eating (?:a lot|too much)|excessive hunger", "excessive hunger"),
    (r"lost weight|losing weight|weight (?:is )?dropping|weight dropped|clothes (?:are )?loose", "weight loss"),
    (r"gained weight|gaining weight|putting on weight|put on weight", "weight gain"),
    (r"vomiting blood|bleeding (?:in|from) (?:my |the )?stomach|stomach bleeding|blood in (?:my )?vomit", "stomach bleeding"),
    (r"mouth ulcers?|ulcers? (?:on|in) (?:my |the )?(?:tongue|mouth)|canker sores?|sores? in (?:my |the )?mouth", "ulcers on tongue"),
    (r"(?:hard|difficult|painful|trouble|difficulty) (?:to )?swallow(?:ing)?|can'?t swallow|hurts? (?:to|when) (?:i )?swallow|dysphagia|swallowing (?:is )?(?:hard|difficult|painful)|pain (?:when|while|on) swallowing", "difficulty swallowing"),
    (r"short(?:ness)? of breath|out of breath|breathless(?:ness)?|difficulty breathing|trouble breathing|hard to breathe|can'?t breathe|cannot breathe|breathing (?:difficulty|difficulties|problems?|trouble)|wheez(?:e|ing|y)|gasping", "breathlessness"),
    (r"running nose|runny nose|nose (?:is |keeps )?running|nasal discharge|dripping nose", "runny nose"),
    (r"blocked nose|stuffy nose|stuffed nose|nasal congestion|congested|congestion|nose (?:is )?blocked|can'?t breathe through (?:my )?nose", "congestion"),
    (r"sore throat|throat (?:pain|ache|is sore|irritation|hurts|is scratchy)|scratchy throat|itchy throat|strep", "throat irritation"),
    (r"white patches (?:in|on) (?:my |the )?throat|patches in (?:my |the )?throat|spots on (?:my |the )?tonsils|swollen tonsils", "patches in throat"),
    (r"sneez(?:e|es|ing|y)|continuous sneezing|keep sneezing", "continuous sneezing"),
    (r"cough(?:s|ing|ed)?|dry cough|wet cough|hacking", "cough"),
    (r"coughing (?:up )?blood|blood (?:in|when) (?:i )?(?:cough|sputum|phlegm)|hemoptysis|haemoptysis", "blood in sputum"),
    (r"rusty (?:coloured |colored )?(?:sputum|phlegm)|rust coloured (?:sputum|phlegm)", "rusty sputum"),
    (r"thick (?:sputum|phlegm|mucus)|mucoid sputum", "mucoid sputum"),
    (r"phlegm|mucus|sputum|chesty", "phlegm"),
    (r"sinus (?:pressure|pain|headache)|pressure (?:in|around) (?:my |the )?sinuses|sinusitis|sinus|face (?:feels )?(?:heavy|full)|facial pressure", "sinus pressure"),
    (r"chest tightness|tight chest|tightness in (?:my |the )?chest|pressure in (?:my |the )?chest|chest (?:feels|is) (?:tight|heavy)|heaviness in (?:my |the )?chest", "chest pain"),
    (r"chill(?:s|y|ed)?|feel(?:ing)? cold|cold shivers|goosebumps", "chills"),
    (r"shiver(?:s|ing|y)?|trembling|shaking|tremors?", "shivering"),
    (r"tired(?:ness)?|exhaust(?:ed|ion)|fatigued?|worn out|no energy|drained|sluggish|weak(?:ness)?|wiped out|low energy|run down", "fatigue"),
    (r"lethargic|lethargy|listless|sleepy all the time|drowsy", "lethargy"),
    (r"sweaty|sweats|sweating|perspir(?:ing|ation)|night sweats|drenched|clammy", "sweating"),
    (r"dizzy|dizziness|light ?headed(?:ness)?|giddy|giddiness|vertigo|woozy|head (?:is )?spinning", "dizziness"),
    (r"head ?aches?|migraines?|head (?:is )?(?:pounding|throbbing|hurting|hurts|killing me)|splitting head", "headache"),
    (r"malaise|generally unwell|feel(?:ing)? unwell|under the weather|not feeling well|feel(?:ing)? (?:ill|awful|terrible|rough|lousy|crappy|horrible)|not well", "malaise"),
    (r"dehydrat(?:ed|ion)|very thirsty|dry mouth|parched|thirsty all the time", "dehydration"),
    (r"body ?aches?|body pain|aching all over|everything hurts|muscle aches?|sore muscles|whole body (?:hurts|aches|pains)", "muscle pain"),
    (r"joint aches?|aching joints|joints? (?:ache|aching)|arthralgia", "joint pain"),
    (r"back ?aches?|lower back pain", "back pain"),
    (r"neck ?aches?", "neck pain"),
    (r"knee aches?", "knee pain"),
    (r"cramp(?:s|ing|ed)?|muscle spasms?|spasms?", "cramps"),
    (r"bruis(?:e|es|ed|ing)|black and blue|bruise easily", "bruising"),
    (r"muscle (?:wasting|loss)|losing muscle|muscles? (?:are |is )?shrinking|thin(?:ner)? (?:arms|legs)", "muscle wasting"),
    (r"stiff(?:ness)?|hard to move|can'?t move (?:properly|well)|movement stiffness|stiff (?:joints|body|in the morning)", "movement stiffness"),
    (r"stiff neck|neck (?:is |feels )?stiff|can'?t turn (?:my )?neck", "stiff neck"),
    (r"hurts to walk|pain(?:ful)? (?:when|while) walking|limping|limp|can'?t walk (?:properly|well)|trouble walking", "painful walking"),
    (r"itch(?:y|es|ing|iness)?|pruritus|scratching", "itching"),
    (r"rash(?:es)?|hives|urticaria|skin (?:eruptions?|rashes|breaking out)|breaking out", "skin rash"),
    (r"bumps? on (?:my |the )?skin|skin bumps|nodules?|lumps? (?:on|under) (?:my |the )?skin", "nodal skin eruptions"),
    (r"pimples?|acne|zits?|pustules?|boils?|whiteheads?", "pus filled pimples"),
    (r"blackheads?", "blackheads"),
    (r"scar(?:s|ring)?|scurring", "scurring"),
    (r"skin (?:is )?peeling|peeling skin|flaky skin|flaking skin|scaly skin|skin (?:is )?flaking", "skin peeling"),
    (r"silver(?:y)? (?:scales|dusting|flakes|patches)", "silver like dusting"),
    (r"discolou?red patches|skin discolou?ration|patches on (?:my |the )?skin|dischromic patches|white patches on (?:my |the )?skin|light patches", "dischromic patches"),
    (r"red spots|spots (?:all )?over (?:my |the )?body|red dots|petechiae|spots on (?:my |the )?(?:body|skin)", "red spots over body"),
    (r"blisters?|fluid filled bumps", "blister"),
    (r"yellow crust(?:s|ing)?|crusty (?:skin|sores?)|oozing|weeping sores?|honey coloured crust", "yellow crust ooze"),
    (r"yellow(?:ish)? skin|skin (?:is |looks )?yellow|jaundiced?", "yellowish skin"),
    (r"yellow(?:ish)? (?:eyes|sclera)|eyes (?:are |look )?yellow|whites of (?:my )?eyes (?:are )?yellow", "yellowing of eyes"),
    (r"dents in (?:my )?nails|nail pitting|pitted nails|small dents in nails", "small dents in nails"),
    (r"brittle nails|nails (?:are )?brittle|nails (?:keep )?break(?:ing)?|nails (?:are )?cracking|weak nails", "brittle nails"),
    (r"inflamed nails|nail inflammation|swollen nails|inflammatory nails|red around (?:my )?nails", "inflammatory nails"),
    (r"dry lips|chapped lips|tingling lips|lips (?:are )?(?:dry|tingling|cracked)|cracked lips", "drying and tingling lips"),
    (r"pale|pallor|look(?:ing)? (?:pale|washed out|white as a sheet)|skin (?:is |looks )?pale|pasty|ashen|no colour in (?:my )?face", "pale skin"),
    (r"sores? around (?:my |the )?nose|red sore around nose|crusty nose", "red sore around nose"),
    (r"blurr(?:y|ed) vision|vision (?:is )?blurr(?:y|ed)|can'?t see (?:clearly|properly)|double vision|distorted vision|blurry", "blurred and distorted vision"),
    (r"red eyes?|eyes (?:are )?red|bloodshot(?: eyes)?|redness (?:of|in) (?:my |the )?eyes?|pink eye", "redness of eyes"),
    (r"watery eyes?|eyes (?:are |keep )?watering|tearing (?:up)?|watering from (?:my |the )?eyes|teary eyes", "watering from eyes"),
    (r"pain behind (?:my |the )?eyes?|eye ?aches?|eyes? (?:hurt|hurts|ache)|sore eyes", "pain behind the eyes"),
    (r"sunken eyes|eyes (?:look |are )?sunken|hollow eyes|dark circles", "sunken eyes"),
    (r"puffy (?:face|eyes)|face (?:is )?puffy|swollen face|face (?:is )?swollen|puffy face and eyes|swollen eyelids?", "puffy face and eyes"),
    (r"visual disturbances?|seeing spots|flashes of light|floaters|sensitive to light|light sensitivity", "visual disturbances"),
    (r"lost (?:my )?(?:sense of )?smell|can'?t smell|no sense of smell|anosmia|loss of smell|smell (?:is )?gone", "loss of smell"),
    (r"lost (?:my )?(?:sense of )?taste|can'?t taste|no sense of taste|loss of taste|ageusia|taste (?:is )?gone|food tastes (?:like )?(?:nothing|bland|cardboard)|everything tastes bland", "loss of taste"),
    (r"ear ?aches?|pain in (?:my |the )?ears?|ears? (?:is |are )?(?:hurt|hurting|hurts|paining|sore|throbbing)|earache|otalgia|ear infection|blocked ears?|ears? (?:feel )?blocked", "ear pain"),
    (r"swollen (?:lymph )?(?:nodes|glands)|swelled lymph nodes|lumps? in (?:my |the )?(?:neck|armpit|groin)|glands (?:are )?(?:swollen|up)", "swelled lymph nodes"),
    (r"swollen legs?|legs? (?:are |is )?swollen|leg swelling", "swollen legs"),
    (r"swollen joints?|joints? (?:are |is )?swollen|joint swelling|swelling joints", "swelling joints"),
    (r"swollen (?:hands|feet|ankles|extremities|fingers|toes)|(?:hands|feet|ankles) (?:are )?swollen|swollen extremities|swollen extremeties", "swollen extremeties"),
    (r"swollen (?:blood )?vessels|bulging veins|veins (?:are )?swollen", "swollen blood vessels"),
    (r"visible veins|prominent veins|veins (?:on|in) (?:my |the )?(?:calf|calves|legs?)|varicose veins|spider veins", "prominent veins on calf"),
    (r"enlarged thyroid|goitre|goiter|swollen thyroid|thyroid (?:is )?(?:enlarged|swollen)|lump in (?:my |the )?throat|swollen neck", "enlarged thyroid"),
    (r"fluid (?:overload|retention)|retaining (?:fluid|water)|water retention|bloated legs", "fluid overload"),
    (r"obese|overweight|obesity", "obesity"),
    (r"cold (?:hands|feet)|hands and feet (?:are )?cold|cold hands and feet|cold hands and feets|freezing (?:hands|feet)", "cold hands and feets"),
    (r"burning (?:sensation )?(?:when|while|during|on) (?:peeing|urinating|urination|passing urine|i pee)|burning (?:pee|urination|micturition)|painful urination|pain(?:ful)? (?:when|while) (?:peeing|urinating)|dysuria|burning micturition|stings? when (?:i )?pee", "burning micturition"),
    (r"peeing (?:a lot|often|frequently|too much|constantly)|frequent urination|urinating (?:a lot|often|frequently)|excessive urination|polyuria|pee(?:ing)? all the time|going to the toilet a lot", "polyuria"),
    (r"constant(?:ly)? (?:urge|need|feel) to (?:pee|urinate)|feel(?:ing)? like (?:peeing|urinating) all the time|always need to pee|continuous feel of urine|urge to pee|bladder feels full", "continuous feel of urine"),
    (r"dark (?:coloured |colored )?(?:urine|pee)|brown (?:urine|pee)|cola coloured urine|tea coloured urine|urine (?:is )?dark", "dark urine"),
    (r"yellow (?:urine|pee)|urine (?:is )?(?:very )?yellow|bright yellow pee", "yellow urine"),
    (r"foul smelling (?:urine|pee)|smelly (?:urine|pee)|urine smells|bad smelling urine|foul smell of urine|pee smells (?:bad|strong)", "foul smell of urine"),
    (r"bladder (?:pain|discomfort|pressure)|pressure in (?:my |the )?bladder|pelvic pressure", "bladder discomfort"),
    (r"spotting (?:when |while |during )?(?:peeing|urinating|urination)|blood (?:in|when) (?:i )?(?:pee|urinate|urine)|spotting urination|pink urine|red urine", "spotting urination"),
    (r"palpitations?|heart (?:is )?(?:racing|pounding|fluttering|skipping)|racing heart|pounding heart|heart skips", "palpitations"),
    (r"fast heart ?beat|fast heart ?rate|rapid heart ?(?:beat|rate)|heart (?:is )?beating (?:fast|quickly)|tachycardia|pulse (?:is )?(?:fast|high|racing)", "fast heart rate"),
    (r"anxious|anxiety|nervous(?:ness)?|panick?(?:y|ing|ed)?|panic attacks?|worried all the time|on edge|stressed out|stress", "anxiety"),
    (r"depress(?:ed|ion|ive)|feel(?:ing)? (?:low|down|hopeless|empty|numb inside)|sad all the time|no interest in anything|no motivation", "depression"),
    (r"irritable|irritability|short tempered|cranky|snappy|easily annoyed|angry all the time", "irritability"),
    (r"mood swings?|moody|mood (?:keeps )?changing|up and down emotionally", "mood swings"),
    (r"restless(?:ness)?|can'?t sit still|fidgety|agitated|can'?t relax", "restlessness"),
    (r"can'?t (?:focus|concentrate)|lack of (?:focus|concentration)|trouble concentrating|poor concentration|hard to concentrate|brain fog|forgetful", "lack of concentration"),
    (r"slurred speech|slurring(?: my words)?|speech (?:is )?slurred|can'?t speak (?:clearly|properly)|words (?:come out|coming out) wrong", "slurred speech"),
    (r"spinning|room (?:is )?spinning|spinning sensation|spinning movements|everything spins", "spinning movements"),
    (r"lose (?:my )?balance|losing (?:my )?balance|off balance|loss of balance|keep falling|falling over", "loss of balance"),
    (r"unsteady|unsteadiness|wobbly|staggering|stumbling", "unsteadiness"),
    (r"numb(?:ness)?|one side (?:is |feels )?(?:weak|numb)|weakness (?:on|of) one side|weakness of one body side|face (?:is )?drooping|facial droop|droopy face|can'?t move (?:my )?(?:arm|leg|one side)", "weakness of one body side"),
    (r"weak(?:ness)? in (?:my |the )?(?:limbs|arms|legs|arms and legs)|limbs (?:are |feel )?weak|weakness in limbs|legs (?:feel )?(?:weak|like jelly)|arms (?:feel )?weak", "weakness in limbs"),
    (r"muscle weakness|muscles (?:are |feel )?weak", "muscle weakness"),
    (r"altered (?:sensorium|consciousness)|confus(?:ed|ion)|disoriented|disorientation|not thinking clearly|foggy|delirious", "altered sensorium"),
    (r"coma|comatose|unconscious|passed out|blacked out|fainted|fainting|collapsed|not responding", "coma"),
    (r"high (?:blood )?sugar|sugar (?:level )?(?:is )?(?:high|irregular|fluctuating|unstable)|blood sugar (?:problems?|issues?)|irregular sugar level|diabetic|glucose (?:is )?high", "irregular sugar level"),
    (r"irregular periods?|missed periods?|periods? (?:is |are )?irregular|menstrual (?:problems?|irregularit(?:y|ies))|heavy periods?|abnormal menstruation|no periods?|late period|period (?:is )?late", "abnormal menstruation"),
    (r"liver failure|acute liver failure|liver (?:is )?failing", "acute liver failure"),
    (r"blood transfusion|transfused|received blood|receiving blood transfusion|had a transfusion", "receiving blood transfusion"),
    (r"unsterile (?:injections?|needles?)|shared needles?|dirty needles?|reused needles?|receiving unsterile injections|used needles", "receiving unsterile injections"),
    (r"family history|runs in (?:my |the )?family|hereditary|genetic|my (?:mother|father|parents|brother|sister|mom|dad) (?:has|had|have) it", "family history"),
    (r"drink(?:ing)? (?:a lot|heavily|too much|alcohol|every day)|alcohol(?:ic|ism)?|heavy drinker|history of alcohol consumption|binge drinking", "history of alcohol consumption"),
    (r"unprotected sex|multiple (?:sexual )?partners|extra ?marital|affairs?|extra marital contacts|unsafe sex", "extra marital contacts"),
    (r"toxic look|look toxic|typhos|look(?:ing)? (?:really )?(?:ill|sick)", "toxic look (typhos)"),
    (r"internal itching|itching inside|itchy inside", "internal itching"),
]

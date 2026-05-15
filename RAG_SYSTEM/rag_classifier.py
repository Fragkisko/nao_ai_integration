import spacy
import unicodedata
import re
import json
from difflib import SequenceMatcher

nlp = spacy.load("el_core_news_lg")

GREEK_STOPWORDS = {
    "ποσα","ένα","έναν","ένας","αι","ακομα","ακομη","ακριβως","αληθεια","αληθινα","αλλα","αλλαχου","αλλες","αλλη","αλλην","αλλης","αλλιως","αλλιωτικα","αλλο","αλλοι","αλλοιως","αλλοιωτικα","αλλον","αλλος","αλλοτε","αλλου","αλλους","αλλων","αμα","αμεσα","αμεσως","αν","ανα","αναμεσα","αναμεταξυ","ανευ","αντι","αντιπερα","αντις","ανω","ανωτερω","αξαφνα","απ","απεναντι","απο","αποψε","από","αρα","αραγε","αργα","αργοτερο","αριστερα","αρκετα","αρχικα","ας","αυριο","αυτα","αυτες","αυτεσ","αυτη","αυτην","αυτης","αυτο","αυτοι","αυτον","αυτος","αυτοσ","αυτου","αυτους","αυτουσ","αυτων","αφοτου","αφου","αἱ","αἳ","αἵ","αὐτόσ","αὐτὸς","αὖ","α∆ιακοπα","βεβαια","βεβαιοτατα","γάρ","γα","γα^","γε","γι","για","γοῦν","γρηγορα","γυρω","γὰρ","δ'","δέ","δή","δαί","δαίσ","δαὶ","δαὶς","δε","δεν","δι","δι'","διά","δια","διὰ","δὲ","δὴ","δ’","εαν","εαυτο","εαυτον","εαυτου","εαυτους","εαυτων","εγκαιρα","εγκαιρως","εγω","ειθε","ειμαι","ειμαστε","ειναι","εις","εισαι","εισαστε","ειστε","ειτε","ειχα","ειχαμε","ειχαν","ειχατε","ειχε","ειχες","ει∆εμη","εκ","εκαστα","εκαστες","εκαστη","εκαστην","εκαστης","εκαστο","εκαστοι","εκαστον","εκαστος","εκαστου","εκαστους","εκαστων","εκει","εκεινα","εκεινες","εκεινεσ","εκεινη","εκεινην","εκεινης","εκεινο","εκεινοι","εκεινον","εκεινος","εκεινοσ","εκεινου","εκεινους","εκεινουσ","εκεινων","εκτος","εμας","εμεις","εμενα","εμπρος","εν","ενα","εναν","ενας","ενος","εντελως","εντος","εντωμεταξυ","ενω","ενός","εξ","εξαφνα","εξης","εξισου","εξω","επ","επί","επανω","επειτα","επει∆η","επι","επισης","επομενως","εσας","εσεις","εσενα","εστω","εσυ","ετερα","ετεραι","ετερας","ετερες","ετερη","ετερης","ετερο","ετεροι","ετερον","ετερος","ετερου","ετερους","ετερων","ετουτα","ετουτες","ετουτη","ετουτην","ετουτης","ετουτο","ετουτοι","ετουτον","ετουτος","ετουτου","ετουτους","ετουτων","ετσι","ευγε","ευθυς","ευτυχως","εφεξης","εχει","εχεις","εχετε","εχθες","εχομε","εχουμε","εχουν","εχτες","εχω","εως","εἰ","εἰμί","εἰμὶ","εἰς","εἰσ","εἴ","εἴμι","εἴτε","ε∆ω","η","ημασταν","ημαστε","ημουν","ησασταν","ησαστε","ησουν","ηταν","ητανε","ητοι","ηττον","η∆η","θα","ι","ιι","ιιι","ισαμε","ισια","ισως","ισωσ","ι∆ια","ι∆ιαν","ι∆ιας","ι∆ιες","ι∆ιο","ι∆ιοι","ι∆ιον","ι∆ιος","ι∆ιου","ι∆ιους","ι∆ιων","ι∆ιως","κ","καί","καίτοι","καθ","καθε","καθεμια","καθεμιας","καθενα","καθενας","καθενος","καθετι","καθολου","καθως","και","κακα","κακως","καλα","καλως","καμια","καμιαν","καμιας","καμποσα","καμποσες","καμποση","καμποσην","καμποσης","καμποσο","καμποσοι","καμποσον","καμποσος","καμποσου","καμποσους","καμποσων","κανεις","κανεν","κανενα","κανεναν","κανενας","κανενος","καποια","καποιαν","καποιας","καποιες","καποιο","καποιοι","καποιον","καποιος","καποιου","καποιους","καποιων","καποτε","καπου","καπως","κατ","κατά","κατα","κατι","κατιτι","κατοπιν","κατω","κατὰ","καὶ","κι","κιολας","κλπ","κοντα","κτλ","κυριως","κἀν","κἂν","λιγακι","λιγο","λιγωτερο","λογω","λοιπα","λοιπον","μέν","μέσα","μή","μήτε","μία","μα","μαζι","μακαρι","μακρυα","μαλιστα","μαλλον","μας","με","μεθ","μεθαυριο","μειον","μελει","μελλεται","μεμιας","μεν","μερικα","μερικες","μερικοι","μερικους","μερικων","μεσα","μετ","μετά","μετα","μεταξυ","μετὰ","μεχρι","μη","μην","μηπως","μητε","μη∆ε","μιά","μια","μιαν","μιας","μολις","μολονοτι","μοναχα","μονες","μονη","μονην","μονης","μονο","μονοι","μονομιας","μονος","μονου","μονους","μονων","μου","μπορει","μπορουν","μπραβο","μπρος","μἐν","μὲν","μὴ","μὴν","να","ναι","νωρις","ξανα","ξαφνικα","ο","οι","ολα","ολες","ολη","ολην","ολης","ολο","ολογυρα","ολοι","ολον","ολονεν","ολος","ολοτελα","ολου","ολους","ολων","ολως","ολως∆ιολου","ομως","ομωσ","οποια","οποιαν","οποιαν∆ηποτε","οποιας","οποιας∆ηποτε","οποια∆ηποτε","οποιες","οποιες∆ηποτε","οποιο","οποιοι","οποιον","οποιον∆ηποτε","οποιος","οποιος∆ηποτε","οποιου","οποιους","οποιους∆ηποτε","οποιου∆ηποτε","οποιο∆ηποτε","οποιων","οποιων∆ηποτε","οποι∆ηποτε","οποτε","οποτε∆ηποτε","οπου","οπου∆ηποτε","οπως","οπωσ","ορισμενα","ορισμενες","ορισμενων","ορισμενως","οσα","οσα∆ηποτε","οσες","οσες∆ηποτε","οση","οσην","οσην∆ηποτε","οσης","οσης∆ηποτε","οση∆ηποτε","οσο","οσοι","οσοι∆ηποτε","οσον","οσον∆ηποτε","οσος","οσος∆ηποτε","οσου","οσους","οσους∆ηποτε","οσου∆ηποτε","οσο∆ηποτε","οσων","οσων∆ηποτε","οταν","οτι","οτι∆ηποτε","οτου","ου","ουτε","ου∆ε","οχι","οἱ","οἳ","οἷς","οὐ","οὐδ","οὐδέ","οὐδείσ","οὐδεὶς","οὐδὲ","οὐδὲν","οὐκ","οὐχ","οὐχὶ","οὓς","οὔτε","οὕτω","οὕτως","οὕτωσ","οὖν","οὗ","οὗτος","οὗτοσ","παλι","παντοτε","παντου","παντως","παρ","παρά","παρα","παρὰ","περί","περα","περι","περιπου","περισσοτερο","περσι","περυσι","περὶ","πια","πιθανον","πιο","πισω","πλαι","πλεον","πλην","ποια","ποιαν","ποιας","ποιες","ποιεσ","ποιο","ποιοι","ποιον","ποιος","ποιοσ","ποιου","ποιους","ποιουσ","ποιων","πολυ","ποσες","ποση","ποσην","ποσης","ποσοι","ποσος","ποσους","ποτε","που","πουθε","πουθενα","ποῦ","πρεπει","πριν","προ","προκειμενου","προκειται","προπερσι","προς","προσ","προτου","προχθες","προχτες","πρωτυτερα","πρόσ","πρὸ","πρὸς","πως","πωσ","σαν","σας","σε","σεις","σημερα","σιγα","σου","στα","στη","στην","στης","στις","στο","στον","στου","στους","στων","συγχρονως","συν","συναμα","συνεπως","συνηθως","συχνα","συχνας","συχνες","συχνη","συχνην","συχνης","συχνο","συχνοι","συχνον","συχνος","συχνου","συχνους","συχνων","συχνως","σχε∆ον","σωστα","σόσ","σύ","σύν","σὸς","σὺ","σὺν","τά","τήν","τί","τίς","τίσ","τα","ταυτα","ταυτες","ταυτη","ταυτην","ταυτης","ταυτο,ταυτον","ταυτος","ταυτου","ταυτων","ταχα","ταχατε","ταῖς","τα∆ε","τε","τελικα","τελικως","τες","τετοια","τετοιαν","τετοιας","τετοιες","τετοιο","τετοιοι","τετοιον","τετοιος","τετοιου","τετοιους","τετοιων","τη","την","της","τησ","τι","τινα","τιποτα","τιποτε","τις","τισ","το","τοί","τοι","τοιοῦτος","τοιοῦτοσ","τον","τος","τοσα","τοσες","τοση","τοσην","τοσης","τοσο","τοσοι","τοσον","τοσος","τοσου","τοσους","τοσων","τοτε","του","τουλαχιστο","τουλαχιστον","τους","τουτα","τουτες","τουτη","τουτην","τουτης","τουτο","τουτοι","τουτοις","τουτον","τουτος","τουτου","τουτους","τουτων","τούσ","τοὺς","τοῖς","τοῦ","τυχον","των","τωρα","τό","τόν","τότε","τὰ","τὰς","τὴν","τὸ","τὸν","τῆς","τῆσ","τῇ","τῶν","τῷ","υπ","υπερ","υπο","υποψη","υποψιν","υπό","υστερα","φετος","χαμηλα","χθες","χτες","χωρις","χωριστα","ψηλα","ω","ωραια","ως","ωσ","ωσαν","ωσοτου","ωσπου","ωστε","ωστοσο","ωχ","ἀλλ'","ἀλλά","ἀλλὰ","ἀλλ’","ἀπ","ἀπό","ἀπὸ","ἀφ","ἂν","ἃ","ἄλλος","ἄλλοσ","ἄν","ἄρα","ἅμα","ἐάν","ἐγώ","ἐγὼ","ἐκ","ἐμόσ","ἐμὸς","ἐν","ἐξ","ἐπί","ἐπεὶ","ἐπὶ","ἐστι","ἐφ","ἐὰν","ἑαυτοῦ","ἔτι","ἡ","ἢ","ἣ","ἤ","ἥ","ἧς","ἵνα","ὁ","ὃ","ὃν","ὃς","ὅ","ὅδε","ὅθεν","ὅπερ","ὅς","ὅσ","ὅστις","ὅστισ","ὅτε","ὅτι","ὑμόσ","ὑπ","ὑπέρ","ὑπό","ὑπὲρ","ὑπὸ","ὡς","ὡσ","ὥς","ὥστε","ὦ","ᾧ","∆α","∆ε","∆εινα","∆εν","∆εξια","∆ηθεν","∆ηλα∆η","∆ι","∆ια","∆ιαρκως","∆ικα","∆ικο","∆ικοι","∆ικος","∆ικου","∆ικους","∆ιολου","∆ιπλα","∆ιχως"
}


DB_ROUTE_THRESHOLD = 0.8
LLM_FALLBACK_THRESHOLD = 0.4

def normalize_text(text):
    text = text.lower()

    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # remove punctuation and normalize whitespace
    text = re.sub(r"[^\w\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# normalized stopwords set (used for clean text)
NORMALIZED_STOPWORDS = { normalize_text(w) for w in GREEK_STOPWORDS }


def process_text(text):
    text = normalize_text(text)
    doc = nlp(text)

    full_tokens = []
    clean_tokens = []

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_
        lemma_norm = normalize_text(lemma)

        # full_text (λεματοποιημένο + normalized)
        full_tokens.append(lemma_norm)

        # clean_text (χωρίς stopwords)
        if lemma_norm not in NORMALIZED_STOPWORDS:
            clean_tokens.append(lemma_norm)

    full_text = " ".join(full_tokens)
    clean_text = " ".join(clean_tokens)

    return full_text, clean_text

def tokenize(text: str) -> list[str]:
    # lowercase
    text = text.lower()
    
    # remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # split
    tokens = text.split()
    
    return tokens

def load_keyword_map():
    return {
        "courses": {
            "μαθημα": 3,
            "μαθηματα": 3,
            "υλη": 2,
            "εξεταση": 3,
            "εργασια": 2,
            "ects": 1,
            "mail": 1
        },

        "professors": {
            "καθηγητης": 3,
            "διδασκων": 3,
            "διδασκαλια": 2,
            "υπευθυνος": 2,
            "επικοινωνια": 2,
            "email": 2
        },

        "postgraduate": {
            "μεταπτυχιακο": 3,
            "διδακτορικο": 3,
            "διδακτορικες σπουδες": 3,
            "μεταπτυχιακες σπουδες": 3,
            "msc": 2,
            "phd": 2
        },

        "undergraduate": {
            "προπτυχιακο": 3,
            "πτυχιο": 3,
            "προγραμμα": 2,
            "πτυχιακο προγραμμα": 2,
            "πρακτικη ασκηση": 3,
            "πρακτικη": 2,
            "ECTS": 2,
        },

        "general_info": {
            "ανακοινωση": 3,
            "οδηγος": 3,
            "ERASMUS": 2,
            "εγγραφες": 2,
            "εξεταστικη": 2,
            "ωρολογιο": 2,
            "ιστορικος": 2,
            "αναδρομος": 2,
            "ιστορικη": 2,
            "ιστορικη αναδρομη": 2,
            "διαρκεια": 1,
            "ποσα": 1,
            "τυπικα": 1,
            "εξαμηνα": 1,
            "τυπικ": 1,
        },

        "facilities": {
            "κτηριο": 3,
            "αιθουσα": 3,
            "εργαστηριο": 3,
            "λεωφορειο": 3,
            "52": 2,
            "βρισκεται": 2,
            "υποδομη": 2,
            "χαρτης": 2,
            "αμφιθεατρο": 2,
            "κεντρικο": 1,
            "λεωφορειο": 1,
            "παω": 1,
            "μετακινηση": 1,
            "παει": 1,

        }
    }

# raw and normalized keyword maps (precomputed for speed and consistency)
RAW_KEYWORD_MAP = load_keyword_map()

def lemma_phrase(phrase: str) -> str:
    doc = nlp(normalize_text(phrase))
    toks = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        lemma_norm = normalize_text(tok.lemma_)
        if lemma_norm and lemma_norm not in NORMALIZED_STOPWORDS:
            toks.append(lemma_norm)
    return " ".join(toks)

# precompute normalized+lemmatized keyword map for fast matching
NORMALIZED_KEYWORD_MAP = {
    intent: { lemma_phrase(k): w for k, w in keywords.items() }
    for intent, keywords in RAW_KEYWORD_MAP.items()
}

text = "Αυτό είναι ένα Τεστ, με Τόνους!"

full, clean = process_text(text)

print("FULL:", full)
print("CLEAN:", clean)
print("TOKENS:", tokenize(clean))

ENTITY_MAP = {
    "transport": ["λεωφορειο", "μετρο"],
    "course": ["αλγοριθμοι", "βασεις δεδομενων"],
    "location": ["σχολη", "κτηριο", "αμφιθεατρο"],
    "person": ["καθηγητης", "διδασκων"]
}

def extract_entities(text: str, normalized: str):
    doc = nlp(text)

    # --- 1. PROPN extraction (βελτιωμένο) ---
    name_candidates = []
    current = []

    for token in doc:
        if token.pos_ == "PROPN" or (
            token.text and token.text[0].isupper() and normalize_text(token.text) not in NORMALIZED_STOPWORDS):
            current.append(token.text)
        else:
            if current:
                name_candidates.append(" ".join(current))
                current = []

    if current:
        name_candidates.append(" ".join(current))

    # --- include spaCy NER results if available ---
    for ent in doc.ents:
        if ent.text not in name_candidates:
            name_candidates.append(ent.text)

    # --- 2. Token-based keyword extraction ---
    tokens = tokenize(normalized)
    joined = " ".join(tokens)

    entities = {}

    for entity_type, keywords in ENTITY_MAP.items():
        for kw in keywords:
            kw_norm = normalize_text(kw)
            if " " in kw_norm:
                if kw_norm in joined:
                    entities.setdefault(entity_type, []).append(kw)
            else:
                if kw_norm in tokens:
                    entities.setdefault(entity_type, []).append(kw)

    # --- 3. Συνδυασμός ---
    return {
        "name_candidates": name_candidates,
        "detected_entities": entities
    }

def classify_query(text: str):
    # 1. πάρε full & clean text
    full_text, clean_text = process_text(text)

    # (αν θες ξεχωριστό normalize κράτα το, αλλιώς full_text αρκεί)
    normalized = full_text

    # tokenize clean_text and use a set for fast membership checks
    tokens = tokenize(clean_text)
    token_set = set(tokens)
    joined = " ".join(tokens)

    # use precomputed normalized+lemmatized keyword map
    norm_keyword_map = NORMALIZED_KEYWORD_MAP

    scores = {}
    total_weights = { intent: sum(keywords.values()) if keywords else 0 for intent, keywords in norm_keyword_map.items() }

    # 2. scoring πάνω στο clean_text (πιο καθαρό), υποστηρίζει φράσεις
    for intent, keywords in norm_keyword_map.items():
        score = 0
        total_weight = total_weights.get(intent, 0)

        for kw, weight in keywords.items():
            if " " in kw:
                if kw in joined:
                    score += weight
            else:
                if kw in token_set:
                    score += weight

        scores[intent] = score

    # 3. επιλογή top intents
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_intent, max_score = sorted_intents[0]

    # 4. fuzzy fallback όταν δεν υπάρχει ξεκάθαρη αντιστοίχιση
    if max_score == 0:
        fuzzy_scores = {}
        for intent, keywords in norm_keyword_map.items():
            fscore = 0
            for kw, weight in keywords.items():
                if " " in kw:
                    sim = SequenceMatcher(None, kw, joined).ratio()
                else:
                    sim = max((SequenceMatcher(None, kw, t).ratio() for t in tokens), default=0)
                if sim >= 0.7:
                    fscore += sim * weight
            fuzzy_scores[intent] = fscore

        fuzzy_sorted = sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True)
        fuzzy_best, fuzzy_score = fuzzy_sorted[0]
        if fuzzy_score > 0:
            best_intent = fuzzy_best
            max_score = fuzzy_score

    # 5. confidence (weighted) — unknown when no score
    if max_score == 0 or total_weights.get(best_intent, 0) == 0:
        best_intent = "unknown"
        confidence = 0.0
    else:
        confidence = float(max_score) / float(max(1, total_weights.get(best_intent, 1)))
        confidence = min(1.0, confidence)

    # only single best intent is returned (best_intent)

    # 6. entities (καλύτερα από full_text)
    entities = extract_entities(text, clean_text)

    return {
        "full_text": full_text,
        "clean_text": clean_text,
        "intent": best_intent,
        "entities": entities,
        "confidence": confidence
    }
if __name__ == "__main__":
    try:
        while True:
            query = input("Ερώτηση (Enter για έξοδο): ").strip()
            if not query:
                print("Έξοδος.")
                break
            result = classify_query(query)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("-" * 50)
    except KeyboardInterrupt:
        print("\nΈξοδος.")
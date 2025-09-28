# src/retriever.py
import unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _normalize(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))  # sans accents
    # singularisations simples
    rules = [
        (r"\bexamens?\b", "examen"),
        (r"\bevaluations?\b", "evaluation"),
        (r"\bformations?\b", "formation"),
        (r"\bprogrammes?\b", "programme"),
        (r"\bfilieres?\b", "filiere"),
        (r"\bsessions?\b", "session"),
        (r"\bdiplomes?\b", "diplome"),
        (r"\bnotes?\b", "note"),
        (r"\bmodalites?\b", "modalite"),
        (r"\bidentifiants?\b", "identifiant"),
        (r"\bmasters?\b", "master"),
        (r"\blicences?\b", "licence"),
        (r"\bproblemes?\b", "probleme"),
        (r"\bdifficultes?\b", "difficulte"),
    ]
    for pat, rep in rules:
        t = re.sub(pat, rep, t)
    # variantes d'écriture
    t = re.sub(r"\buv[\-\s]?bf\b", "uvbf", t)
    t = re.sub(r"\bmdp\b", "mot de passe", t)
    # separateurs
    t = re.sub(r"[_/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Stopwords FR (liste courte, personnalisable)
_RAW_FR_STOP = {
    "alors","au","aucun","aussi","autre","aux","avec","avoir","bon","car","ce","cela","ces","cet","cette","ceux",
    "chaque","comme","comment","dans","de","des","du","elle","elles","en","encore","est","et","etc","etre","eux",
    "faire","fois","hors","il","ils","je","la","le","les","leur","lui","mais","malgre","me","meme","mes","moi",
    "mon","ne","nos","notre","nous","on","ou","par","parce","pas","peu","plus","pour","pourquoi","quand","que",
    "quel","quelle","quelles","quels","qui","sans","se","ses","si","sont","sur","ta","te","tes","toi","ton",
    "tous","tout","toute","toutes","tres","tu","un","une","vos","votre","vous","y"
}


# Normaliser pour être cohérent avec le preprocessor
FR_STOPWORDS = {_normalize(w) for w in _RAW_FR_STOP}

class Retriever:
    def __init__(self, docs, ngram_range=(1,2), min_df=1, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            preprocessor=_normalize,
            token_pattern=r"(?u)\b\w+\b",
            stop_words=FR_STOPWORDS
        )
        self.doc_term = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k=3):
        if not query: return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.doc_term).ravel()
        idxs = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0.0]

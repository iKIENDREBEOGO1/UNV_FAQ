# src/retriever.py (extrait)
import unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # enlever les accents
    t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))
    # normalisations domaine: pluriels -> singuliers simples
    #   examens -> examen, formations -> formation, etc.
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
        (r"\bdifficultes?\b", "difficulte")
    ]
    for pat, rep in rules:
        t = re.sub(pat, rep, t)
    # variantes écriture
    t = re.sub(r"\buv[\-\s]?bf\b", "uvbf", t)
    t = re.sub(r"\bmdp\b", "mot de passe", t)
    # espaces propres
    t = re.sub(r"[_/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

class Retriever:
    def __init__(self, docs, ngram_range=(1,2), min_df=1, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            preprocessor=_normalize,        # <<<<<< clé pour sing/pluriel + accents
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_term = self.vectorizer.fit_transform(docs)
    def search(self, query: str, top_k=3):
        if not query:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.doc_term).ravel()
        idxs = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0]

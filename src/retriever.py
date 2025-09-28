# src/retriever.py
import re
import unicodedata
from typing import Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 1) Normalisation commune ----------
def _strip_accents_lower(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = "".join(
        c for c in unicodedata.normalize("NFKD", t)
        if not unicodedata.combining(c)
    )
    return t

# règles simples singulier/pluriel + harmonisation domaine
_SINGULAR_RULES = [
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

def _domain_normalize(text: str) -> str:
    t = _strip_accents_lower(text)
    for pat, rep in _SINGULAR_RULES:
        t = re.sub(pat, rep, t)
    # variantes d'écriture
    t = re.sub(r"\buv[\-\s]?bf\b", "uvbf", t)
    t = re.sub(r"\bmdp\b", "mot de passe", t)
    # séparateurs -> espace
    t = re.sub(r"[_/]", " ", t)
    # tokens propres
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- 2) Stopwords FR minimalistes ----------
# (Tu peux étoffer la liste si besoin)
_FR_STOPWORDS = {
    "alors","au","aucun","aussi","autre","aux","avec","avoir","bon","car","ce","cela","ces","cet","cette","ceux",
    "chaque","comme","comment","dans","de","des","du","elle","elles","en","encore","est","et","etc","etre","eux",
    "faire","fois","hors","il","ils","je","la","le","les","leur","lui","mais","malgre","me","meme","mes","moi",
    "mon","ne","nos","notre","nous","on","ou","par","parce","pas","peu","plus","pour","pourquoi","quand","que",
    "quel","quelle","quelles","quels","qui","sans","se","ses","si","sont","sur","ta","te","tes","toi","ton",
    "tous","tout","toute","toutes","tres","tu","un","une","vos","votre","vous","y"
}
# On normalise la stoplist avec la même fonction
_FR_STOPWORDS = { _domain_normalize(w) for w in _FR_STOPWORDS }

def _clean(text: str) -> str:
    """Normalisation + retrait stopwords (côté documents et requêtes)."""
    t = _domain_normalize(text)
    # tokenisation simple: mots alphanumériques
    tokens = re.findall(r"\b\w+\b", t)
    # retirer les stopwords
    tokens = [tok for tok in tokens if tok not in _FR_STOPWORDS]
    return " ".join(tokens)


# ---------- 3) Retriever ----------
class Retriever:
    def __init__(self, docs: Iterable[str], ngram_range=(1, 2), min_df=1, max_df=0.95):
        """
        Retriever TF-IDF + cosinus, avec prétraitement externe (normalisation + stopwords).
        - docs : liste/Series de textes (question + réponse + mots_cles).
        """
        # On prétraite les documents AVANT vectorisation
        self._docs_raw: List[str] = list(docs)
        self._docs_clean: List[str] = [_clean(d) for d in self._docs_raw]

        # Le vectorizer ne fait plus de lowercase ni de stopwords : déjà fait.
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=False,       # déjà fait dans _clean
            preprocessor=None,     # déjà fait
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None        # déjà retirés
        )
        self.doc_term = self.vectorizer.fit_transform(self._docs_clean)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Retourne [(index_doc, score)] triés par score décroissant, taille <= top_k.
        """
        if not query:
            return []

        q_clean = _clean(query)
        if not q_clean:
            return []

        q_vec = self.vectorizer.transform([q_clean])
        scores = cosine_similarity(q_vec, self.doc_term).ravel()
        idxs = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0.0]  # ✅ parenthèse fermée

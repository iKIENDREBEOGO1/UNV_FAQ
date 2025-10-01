# src/retriever.py
import re
import unicodedata
from typing import Iterable, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------ Normalisation ------------
def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # enlever accents
    t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))
    # pluriels -> singuliers (règles simples domaine)
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
    # nettoyages
    t = re.sub(r"[_/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    """Tokenisation simple (après _normalize)."""
    t = _normalize(text)
    return re.findall(r"\b\w+\b", t)


def _keyword_list(cell) -> List[str]:
    """
    Transforme la cellule 'mots_cles' (ex: 'examen; presentiel; en ligne')
    en liste de formes normalisées, en gardant aussi les tokens multi-mots éclatés.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    kws_raw = [k.strip() for k in s.split(";") if k.strip()]
    out: List[str] = []
    for k in kws_raw:
        kn = _normalize(k)
        if kn:
            out.append(kn)  # forme entière normalisée
            out.extend(re.findall(r"\b\w+\b", kn))  # tokens
    # dédoublonnage en gardant l'ordre
    return list(dict.fromkeys(out))


def _keyword_overlap_score(query_tokens: List[str], kw_norm_list: List[str]) -> float:
    """
    Proportion de mots-clés présents dans la requête.
    nb_matches / nb_keywords (0..1). Si pas de mots-clés, renvoie 0.
    """
    if not kw_norm_list:
        return 0.0
    qset = set(query_tokens)
    matches = sum(1 for k in kw_norm_list if k in qset)
    return matches / len(kw_norm_list)


# ------------ Retriever ------------
class Retriever:
    def __init__(
        self,
        docs: Iterable[str],
        faq_df: Optional["object"] = None,  # pandas.DataFrame attendu, mais typé optionnel
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        keyword_weight: float = 0.30,  # 30% mots-clés par défaut
        threshold: float = 0.0         # seuil minimal sur le score final
    ):
        """
        Retriever TF-IDF + cosinus, avec score optionnel de recouvrement des mots-clés.
        - docs       : liste/Series de textes (ex: question + reponse + mots_cles)
        - faq_df     : DataFrame contenant au moins la colonne 'mots_cles'
        - keyword_weight : poids du score mots-clés (0..1). Si faq_df=None => ignoré
        - threshold  : filtre les résultats dont le score final < threshold
        """
        self.has_keywords = faq_df is not None and "mots_cles" in faq_df.columns
        self.keyword_weight = float(keyword_weight if self.has_keywords else 0.0)
        self.threshold = float(threshold)

        # TF-IDF avec normalisation via preprocessor
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            preprocessor=_normalize,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_term = self.vectorizer.fit_transform(docs)

        # Si on a le DF, pré-calculer les mots-clés normalisés
        self._kw_lists: List[List[str]] = []
        if self.has_keywords:
            self._kw_lists = [_keyword_list(faq_df.loc[i, "mots_cles"]) for i in range(len(faq_df))]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        if not query:
            return []

        # 1) Score cosinus
        q_vec = self.vectorizer.transform([query])
        cos_scores = cosine_similarity(q_vec, self.doc_term).ravel()

        # 2) Score mots-clés (si dispo)
        if self.keyword_weight > 0.0:
            q_tokens = _tokenize(query)
            combined: List[Tuple[int, float]] = []
            for i, cos_s in enumerate(cos_scores):
                kw_s = _keyword_overlap_score(q_tokens, self._kw_lists[i])
                final = (1.0 - self.keyword_weight) * float(cos_s) + self.keyword_weight * float(kw_s)
                if final >= self.threshold:
                    combined.append((int(i), final))
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined[:top_k]

        # Sinon: cosinus pur
        idxs = cos_scores.argsort()[::-1]
        out: List[Tuple[int, float]] = []
        for i in idxs:
            s = float(cos_scores[i])
            if s < self.threshold:
                break
            out.append((int(i), s))
            if len(out) == top_k:
                break
        return out

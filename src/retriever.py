from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, docs, ngram_range=(1,2), min_df=1, max_df=0.95, **kwargs):
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        self.doc_texts = docs.fillna("").astype(str).tolist()
        self.doc_mat = self.vectorizer.fit_transform(self.doc_texts)

    def search(self, query: str, top_k=3):
        if not query:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_mat)[0]
        top = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[:top_k]
        return top

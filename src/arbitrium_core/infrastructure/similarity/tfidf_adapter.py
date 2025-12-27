from typing import Any

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arbitrium_core.ports.similarity import SimilarityEngine


class TfidfSimilarityEngine(SimilarityEngine):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]) -> Any:
        return self.vectorizer.transform(texts)

    def is_fitted(self) -> bool:
        if not self._fitted:
            return False
        try:
            self.vectorizer.transform(["test"])
            return True
        except NotFittedError:
            return False

    def compute_similarity(
        self, query_vector: Any, corpus_vectors: Any
    ) -> list[float]:
        scores = cosine_similarity(query_vector, corpus_vectors)
        result: list[float] = scores.flatten().tolist()
        return result

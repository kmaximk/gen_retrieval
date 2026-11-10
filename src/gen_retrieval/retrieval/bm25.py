import math
from collections import Counter, defaultdict
from ..tokenization import TokenizerBM25


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = TokenizerBM25()
        self.corpus_size = 0
        self.avg_dl = 0.0
        self.doc_lens: list[int] = list()
        self.inverted_index = defaultdict(list)
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]):
        self.corpus_size = len(corpus)
        df = defaultdict(int)
        total = 0

        for idx, text in enumerate(corpus):
            tokens = self.tokenizer.tokenize(text)
            ln = len(tokens)
            self.doc_lens.append(ln)
            total += ln
            tf = Counter(tokens)
            for token, count in tf.items():
                self.inverted_index[token].append((idx, count))
                df[token] += 1
        self.avg_dl = total / self.corpus_size if self.corpus_size else 0.0
        for token, freq in df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> dict[int, float]:
        tokens = self.tokenizer.tokenize(query)
        scores = defaultdict(float)
        for token in tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            for idx, tf in self.inverted_index[token]:
                doc_len = self.doc_lens[idx]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))
                scores[idx] += idf_val * (num / den)
        return scores

    def batcher(self, queries: list[str], topk: int = 100) -> list[list[int]]:
        batches: list = []
        for q in queries:
            scores = self.score(q)
            top_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:topk]
            batches.append(top_docs)
        return batches

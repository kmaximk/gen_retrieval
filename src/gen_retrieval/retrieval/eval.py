from gen_retrieval.retrieval.bm25 import BM25
from gen_retrieval.retrieval.metrics import recall_k
import json
import numpy as np


def load_json_or_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)
        if first_line.startswith("{") and not first_line.endswith("}"):
            return json.load(f)
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                return data


"""
# 1. Загрузка данных датасета LIMIT (формат json/huggingface)
# Репозиторий авторов: https://github.com/google-deepmind/limit
print("Загрузка корпуса и запросов...")
Для теста подставьте пути к файлам датасета LIMIT:
with open('data/corpus.json') as f: corpus = json.load(f)
with open('data/queries.json') as f: queries = json.load(f)
# Пример структуры:
corpus = [
    "Jon likes Quokkas and Apples.",
    "Ovid likes Quokkas and Rabbits.",
    "Leslie likes Apples and Candy.",
]
queries = {"q1": "Who likes Quokkas?"}
qrels = {"q1": [0, 1]}
"""

raw_corpus = load_json_or_jsonl('src/gen_retrieval/data/limit/corpus.jsonl')
raw_queries = load_json_or_jsonl('src/gen_retrieval/data/limit/queries.jsonl')

# corpus
if isinstance(raw_corpus, dict):
    doc_ids = list(raw_corpus.keys())
    corpus_texts = [
        raw_corpus[d]["text"] if isinstance(raw_corpus[d], dict) else raw_corpus[d]
        for d in doc_ids
    ]
else:
    doc_ids = [str(item.get("_id", item.get("id", idx))) for idx, item in enumerate(raw_corpus)]
    corpus_texts = [item.get("text", item.get("title", "")) for item in raw_corpus]
doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

# query
if isinstance(raw_queries, dict):
    query_ids = list(raw_queries.keys())
    query_texts = [
        raw_queries[q]["text"] if isinstance(raw_queries[q], dict) else raw_queries[q]
        for q in query_ids
    ]
else:
    query_ids = [str(item.get("_id", item.get("id", idx))) for idx, item in enumerate(raw_queries)]
    query_texts = [item.get("text", "") for item in raw_queries]

# relevant
qrels = {}
if qrels_file.endswith(".json") or qrels_file.endswith(".jsonl"):
    raw_qrels = load_json_or_jsonl(qrels_file)
    if isinstance(raw_qrels, dict):
        qrels = {str(k): set(v.keys() if isinstance(v, dict) else v) for k, v in raw_qrels.items()}
    else:
        for row in raw_qrels:
            qid = str(row.get("query-id", row.get("query_id")))
            did = str(row.get("corpus-id", row.get("corpus_id")))
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(did)
else:
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] != "query-id":
                qid, did = parts[0], parts[1]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(did)


model_bm25 = BM25(k1=1.5, b=0.75)
model_bm25.fit(corpus_texts)
top_predictions = model_bm25.batcher(query_texts, topk=100)
for k in [2, 10, 100]:
    recalls = []
    for q_idx, qid in enumerate(query_ids):
        gold_docs = qrels.get(qid, set())
        gold_indices = {doc_id_to_idx[d] for d in gold_docs if d in doc_id_to_idx}
        if not gold_indices:
            continue
        preds_k = set(top_predictions[q_idx][:k])
        hits = len(gold_indices.intersection(preds_k))
        recalls.append(hits / len(gold_indices))
    print(f"Recall@{k:3d}: {np.mean(recalls) * 100:.2f}%")

"""
embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(
    corpus_texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True
)
query_embeddings = embedder.encode(
    query_texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save("doc_embeddings.npy", doc_embeddings)
np.save("query_embeddings.npy", query_embeddings)
"""

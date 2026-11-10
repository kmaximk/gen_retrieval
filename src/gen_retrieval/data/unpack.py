import os
import json
import glob

# VIBECODE BTW

# Клонируем репозиторий авторов статьи
if not os.path.exists("limit"):
    !git clone https://github.com/google-deepmind/limit.git

# Определяем путь к данным внутри репозитория
repo_path = "limit"
print("Файлы в репозитории:", os.listdir(repo_path))

# Поиск путей к файлам корпуса, запросов и qrels
def find_file(patterns):
    for p in patterns:
        matches = glob.glob(os.path.join(repo_path, "**", p), recursive=True)
        if matches:
            return matches[0]
    return None

corpus_file = find_file(["*corpus*.json*", "*corpus*.tsv"])
queries_file = find_file(["*queries*.json*", "*queries*.tsv"])
qrels_file = find_file(["*qrels*.json*", "*qrels*.tsv", "*test*.tsv"])

print(f"Найден корпус: {corpus_file}")
print(f"Найдены запросы: {queries_file}")
print(f"Найдены qrels: {qrels_file}")

def load_json_or_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        # Проверяем JSONL (построчный)
        first_line = f.readline().strip()
        f.seek(0)
        if first_line.startswith("{") and not first_line.endswith("}"):
            # Цельный JSON
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

raw_corpus = load_json_or_jsonl(corpus_file)
raw_queries = load_json_or_jsonl(queries_file)

# Парсинг корпуса
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

# Парсинг запросов
if isinstance(raw_queries, dict):
    query_ids = list(raw_queries.keys())
    query_texts = [
        raw_queries[q]["text"] if isinstance(raw_queries[q], dict) else raw_queries[q]
        for q in query_ids
    ]
else:
    query_ids = [str(item.get("_id", item.get("id", idx))) for idx, item in enumerate(raw_queries)]
    query_texts = [item.get("text", "") for item in raw_queries]

# Парсинг qrels
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
    # TSV-формат (BEIR): query-id \t corpus-id \t score
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] != "query-id":
                qid, did = parts[0], parts[1]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(did)

print(f"Успешно загружено: {len(corpus_texts)} документов и {len(query_texts)} запросов.")
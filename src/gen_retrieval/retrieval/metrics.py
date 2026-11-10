def recall_k(predictions: list[list[int]], relevant: dict[str, list[int]], k: int) -> float:
    scores = []
    for query, pr in zip(relevant.keys(), predictions):
        gold = set(relevant[query])
        if not gold:
            continue
        retrieve_k = set(pr[:k])
        hits = len(gold.intersection(retrieve_k))
        scores.append(hits / len(gold))
    return sum(scores) / len(scores) if scores else 0.0

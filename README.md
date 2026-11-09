# Generative Retrieval

## Структура

Пока это каркас. Папки предназначены для следующих частей проекта:

```text
src/gen_retrieval/
├── data/             чтение и подготовка корпуса, запросов и qrels
├── embeddings/       получение и сохранение векторов документов и запросов
├── retrieval/        BM25, exact dense и ANN-поиск
│   └── tiger/        модель, обучение и поиск «запрос → SID документа»
└── tokenization/     построение SID через RQ-KMeans и RQ-VAE
```

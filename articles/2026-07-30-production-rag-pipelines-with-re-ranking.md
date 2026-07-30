---
title: Production RAG Pipelines with Re-ranking: Patterns, Pitfalls, and Code
tags:
  - RAG
  - re-ranking
  - scalable architectures
  - hybrid retrieval
  - MLOps
  - LLMs
author: Rehan Malik
---

# Production RAG Pipelines with Re-ranking: Patterns, Pitfalls, and Code

![Production RAG Pipelines with Re-ranking](../images/production-rag-pipelines-with-re-ranking.jpg)

By Rehan Malik

## TL;DR

- **Hybrid retrieval (dense + sparse) is now table stakes for scalable RAG**
- **Cross-encoder re-ranking improves answer quality, but introduces latency tradeoffs**
- **Caching, batching, and async processing are non-negotiable for real-world throughput**
- **Monitoring, evaluation, and frequent retriever refresh are required, not optional**

## Prerequisites

If you want to run the code or build something similar, you'll need:

- Python 3.9+ (I'm using 3.10)
- `faiss-cpu` >= 1.7.4 for dense vector search
- `elasticsearch` >= 8.0.0 for sparse + hybrid search
- `transformers` >= 4.36 for cross-encoder re-ranking
- `redis` >= 4.5.0 for embedding cache (optional)
- `ray` >= 2.7.0 for batch/async processing (optional)
- CUDA GPU for cross-encoder speed (optional but recommended)
- Docker/Kubernetes or some cloud setup for scaling (if you actually want to productionize)

## Introduction

Retrieval-Augmented Generation (RAG) is past the toy phase. I see teams tripping over the same practical problems: slow queries, brittle re-ranking, retrievers that don't update with new data, and pipelines that stall under real load. The hype over LLMs is giving way to harder questions: can I get reliable answers fast enough, with data that's fresh, and a pipeline that stays up under scale?

Hybrid retrieval (dense plus sparse), cross-encoder re-ranking, and orchestrating generators are now the backbone of modern RAG. Getting these pieces right is non-trivial. Here's how I approach the architecture, what's tripped me up, and actual code that works.

## Technical Deep Dive: Hybrid Retrieval and Re-ranking

### Dense Retrieval with FAISS

I start with dense retrieval using FAISS. Sentence transformers like `BAAI/bge-base-en` are my go-to for embeddings. FAISS gives fast vector search.

```python
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
docs = [
    "RAG is a retrieval-augmented generation method.",
    "FAISS enables fast vector search.",
    "Hybrid retrieval combines dense and sparse methods."
]

doc_embeddings = embedder.encode(docs, normalize_embeddings=True)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(np.array(doc_embeddings))

query = "What is hybrid retrieval?"
query_embedding = embedder.encode([query], normalize_embeddings=True)

scores, idxs = index.search(np.array(query_embedding), 3)
results = [docs[i] for i in idxs[0]]
print("Dense retrieval results:", results)
```

**What I've learned:** FAISS is fast and simple, but only gives similarity scores. Embeddings have to be normalized. Dense retrieval is weak on rare keywords or jargon, so I never depend on it alone.

### Sparse Retrieval with BM25 (Elasticsearch)

For BM25, Elasticsearch does the job. You can index your docs and query with BM25. This is the minimal wiring:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

for i, doc in enumerate(docs):
    es.index(index="rag_docs", id=i, document={"text": doc})

resp = es.search(index="rag_docs", query={"match": {"text": "hybrid retrieval"}})
bm25_results = [hit["_source"]["text"] for hit in resp["hits"]["hits"]]
print("BM25 results:", bm25_results)
```

**My take:** BM25 is reliable for keyword-heavy queries, but doesn't understand semantics. I use hybrid setups (like Elasticsearch's ELSER or custom rerank logic) for anything serious.

### Re-ranking with CrossEncoder (Transformers)

Re-ranking is where the boost in answer quality happens. Cross-encoders (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) take a query-document pair and spit out a relevance score.

```python
from transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
query = "What is hybrid retrieval?"
candidates = results + bm25_results

pairs = [(query, doc) for doc in candidates]

scores = cross_encoder.predict(pairs)
reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
print("Reranked results:", reranked[:3])
```

**Reality check:** Cross-encoders are slow and scale poorly if you feed them too many candidates. I typically grab top-10 from each retriever, batch up pairs, and run scoring async when latency matters.

## Architecture Patterns

The typical production RAG pipeline I build is a three-stage system:

1. **Retriever(s):** Hybrid search (dense via FAISS, sparse via Elasticsearch)
2. **Re-ranker:** Cross-encoder model (preferably GPU or batched CPU)
3. **Generator:** LLM (OpenAI, TGI, vLLM) that builds answers from reranked contexts

I split this into microservices:

- **Retrieval service:** Exposes `/search`, runs dense (FAISS) + sparse (BM25/Elasticsearch), merges top-k
- **Re-rank service:** Accepts candidate pairs, batches and scores via cross-encoder, returns top contexts
- **LLM service:** Generates answer, acts as REST or gRPC
- **Cache layer:** Redis for embedding/document cache
- **Monitoring:** Prometheus/Grafana for latency, error rates, candidate pool sizes

**Architecture diagram (in plain text):**

```text
User Query
   |
   v
Retrieval Service (FAISS + Elasticsearch)
   |
   v
Candidate Pool (top-k from both)
   |
   v
Re-rank Service (CrossEncoder)
   |
   v
Generator Service (LLM like OpenAI, TGI, vLLM)
   |
   v
Final Answer
```

**Scaling:** I run retrievers and re-rankers as separate pods in Kubernetes, use GPU for the cross-encoder, and batch scoring through Ray or Dask. Caching embeddings in Redis drops latency for repeated queries.

## Lessons Learned

- **Latency:** Re-ranking is the main bottleneck. If you don't batch and async the scoring, you'll bottleneck everything. Ray works well for this.
- **Cold starts:** Sparse retrievers need index refresh when you update data. Forgetting this leads to stale answers.
- **Chunking:** Overlapping chunks reduce context loss but bloat the index. I stick to 20% overlap for balance.
- **Evaluation:** Latency, cost per query, and retriever freshness matter as much as answer accuracy.
- **Caching:** Embedding cache in Redis is a huge win, especially for repeated queries.

### Async Batch Processing Example

Batching re-ranking with Ray lets me scale to hundreds of pairs/sec.

```python
import ray
from transformers import CrossEncoder

ray.init(ignore_reinit_error=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@ray.remote
def score_pair(pair):
    return cross_encoder.predict([pair])[0]

query = "What is hybrid retrieval?"
candidates = results + bm25_results
pairs = [(query, doc) for doc in candidates]

score_objs = [score_pair.remote(pair) for pair in pairs]
scores = ray.get(score_objs)
reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
print("Async reranked results:", reranked[:3])
```

## Key Takeaways

1. Use hybrid retrieval (dense + sparse), never just one.
2. Batch and cache re-ranking with cross-encoders or you'll hit latency walls.
3. Automate retriever refresh and monitor candidate pool sizes.
4. Evaluate pipelines on latency and accuracy, not just one.
5. Split retrieval, re-ranking, and generation into separate services for easier scaling.

## Further Reading

- [ColBERT: Efficient and Effective Passage Search](https://github.com/stanford-futuredata/ColBERT)
- [BAAI BGE Embedding Models](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [Elasticsearch ELSER Hybrid Retrieval](https://www.elastic.co/blog/introducing-elser)
- [Cohere Re-ranker Docs](https://docs.cohere.com/docs/rerank)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [BEIR Benchmark](https://github.com/beir-dev/BEIR)
- [Ray for distributed Python](https://docs.ray.io/en/latest/)
- [vLLM: Fast, scalable LLM inference](https://github.com/vllm-project/vllm)
- [LlamaIndex: Unified RAG Framework](https://github.com/jerryjliu/llama_index)
- [LangChain: Modular LLM Framework](https://github.com/langchain-ai/langchain)

<!-- <script type='application/ld+json'>{"@context":"https://schema.org","@type":"TechArticle","headline":"Production RAG Pipelines with Re-ranking: Patterns, Pitfalls, and Code","author":{"@type":"Person","name":"Rehan Malik"},"datePublished":"2024-06-06"}</script> -->

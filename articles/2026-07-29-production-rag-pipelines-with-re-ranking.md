---
title: Building Production-Grade RAG Pipelines with Re-ranking at Scale 
tags: RAG, NLP, AI/ML, re-ranking, production pipelines 
author: Rehan Malik 
---

# Building Production-Grade RAG Pipelines with Re-ranking at Scale
![Production RAG Pipelines with Re-ranking](../images/production-rag-pipelines-with-re-ranking.jpg)

## TL;DR
- Retrieval-Augmented Generation (RAG) combines retrieval and generation for context-aware NLP tasks. 
- Re-ranking retrieved documents improves relevance and overall quality. 
- Techniques like dense vector retrieval and cross-encoder re-ranking are foundational to scaling RAG pipelines. 

## Prerequisites
To follow this article, ensure you have: 
- Python 3.9+ 
- `transformers` (v4.30+) 
- `sentence-transformers` (v2.2+) 
- `faiss-cpu` or `faiss-gpu` 
- A working understanding of retrieval-based NLP and RAG principles 

---

## Introduction
Retrieval-Augmented Generation (RAG) pipelines are at the core of modern NLP systems that demand both factual accuracy and dynamic context integration. By combining retrieval systems with generative models, RAG pipelines can augment large language models with external, domain-specific knowledge. 

One challenge I've faced when operationalizing RAG workflows is ensuring the relevance of retrieved documents. Out-of-the-box retrieval approaches often bring back noisy or suboptimal results. Re-ranking addresses this by improving how retrieved documents are prioritized, which directly affects the quality of the final output. 

I'll walk through how I build scalable, production-grade RAG systems with dense vector retrieval, cross-encoder re-ranking, and integration into a complete pipeline.

---

## Technical Deep Dive

### Dense Vector Retrieval
The retrieval process starts with dense vector representations of documents and queries, which allow us to compute semantic similarity. `sentence-transformers` simplifies this.

Here's how I set up dense vector encoding for both documents and queries:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun is shining brightly in the clear blue sky.",
    "The cat purrs contentedly on my lap."
]

# Encode documents into dense vectors
document_embeddings = model.encode(documents)

# Example query
query = "What is the color of the sky?"

# Encode query as a dense vector
query_embedding = model.encode(query)

# Compute similarity scores using dot product
similarity_scores = np.dot(document_embeddings, query_embedding)

# Map scores back to documents
ranked_docs = sorted(
    zip(documents, similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

print("Ranked documents and scores:")
for doc, score in ranked_docs:
    print(f"{score:.4f} - {doc}")
```

**Key challenges:** 
- Selecting the right pre-trained model for embeddings depends on your domain. For general text, models like `all-MiniLM-L6-v2` or `multi-qa-MiniLM-L6-v2` work well. 
- Dense retrieval is limited by the initial set of embeddings, errors here propagate downstream. 

---

### Re-ranking Retrieved Documents
Dense retrieval provides an initial ranking, but it's not always sufficient. Cross-encoders improve this by re-scoring based on joint query-document representations. Unlike vector-based retrieval, cross-encoders consider fine-grained interactions between the query and each document.

Here's how I implemented re-ranking with Hugging Face's `transformers` library:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a cross-encoder model for re-ranking
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example query and retrieved documents
query = "What is the color of the sky?"
retrieved_documents = [
    "The sun is shining brightly in the clear blue sky.",
    "The quick brown fox jumps over the lazy dog.",
    "The cat purrs contentedly on my lap."
]

# Re-rank documents using the cross-encoder
def re_rank(query, documents):
    scores = []
    with torch.no_grad():
        for doc in documents:
            inputs = tokenizer(query, doc, return_tensors='pt', padding=True, truncation=True)
            logits = model(**inputs).logits
            scores.append(logits.item())
    return np.argsort(scores)[::-1] # Sort indices by descending scores

# Get re-ranked indices
re_ranked_indices = re_rank(query, retrieved_documents)

# Display re-ranked results
print("Re-ranked documents:")
for idx in re_ranked_indices:
    print(retrieved_documents[idx])
```

**Key considerations:** 
- Cross-encoders are slower than dense retrieval because they require processing each query-document pair independently. 
- For production, consider caching frequent query results or limiting re-ranking to top-k retrieved candidates. 

---

### Integrating Re-ranking into a RAG Pipeline
Here's how I structure a RAG pipeline to handle both dense retrieval and re-ranking.

1. **Document Indexing**: Use dense embeddings and store them in a vector database like FAISS. 
2. **Query Retrieval**: Generate a query embedding and retrieve top-N similar documents based on cosine similarity in the vector space. 
3. **Re-ranking**: Pass the top-N documents to a cross-encoder for re-scoring and reordering. 
4. **Generation**: Use the re-ranked documents as input context for a generative model. 

Below is a simplified implementation using FAISS for retrieval and re-ranking:

```python
import faiss

# Build a FAISS index for dense vector retrieval
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Retrieve top-3 documents
top_k = 3
D, I = index.search(np.array([query_embedding]), top_k)

# Retrieve the top-k documents from the index
retrieved_docs = [documents[i] for i in I[0]]

# Re-rank the retrieved documents
re_ranked_indices = re_rank(query, retrieved_docs)

# Final re-ranked documents
final_ranking = [retrieved_docs[idx] for idx in re_ranked_indices]

print("Final re-ranked documents:")
for doc in final_ranking:
    print(doc)
```

This modular setup makes the pipeline easy to scale and experiment with, whether by swapping out models or adjusting parameters like `k` (number of top documents retrieved). 

**Bottlenecks:** 
- Scaling re-ranking can become computationally expensive. For large document collections, you'll need to leverage distributed FAISS or approximate search techniques. 
- Generative models often require fine-tuning with retrieved context to produce coherent and accurate outputs. 

---

## Lessons Learned
From building and deploying RAG pipelines, here's what stands out: 
- **Iterate on the retrieval model**: Tuning the dense retriever model can reduce the burden on re-ranking and improve end-to-end efficiency. 
- **Re-ranking can't fix everything**: If the retrieval step is poor, re-ranking won't salvage it. Garbage in, garbage out still applies. 
- **Fine-tune your LLM**: Generic generative models, no matter how advanced, benefit from domain-specific fine-tuning with in-context training data. 

---

## Key Takeaways
1. **Dense retrieval is foundational**, but it's not perfect. Pair it with advanced re-ranking techniques for production-grade relevance. 
2. **Cross-encoders are slow**, optimize your pipeline by caching results or reducing the size of the candidate set for re-ranking. 
3. **Scalability matters**: Scaling across large document collections requires robust indexing and retrieval systems like FAISS. 

---

## Resources for Further Learning
- [SentenceTransformers Documentation](https://www.sbert.net/) 
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) 
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) 

By Rehan Malik

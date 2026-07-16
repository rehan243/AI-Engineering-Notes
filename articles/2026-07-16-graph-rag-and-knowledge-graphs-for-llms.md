---
title: When Vector Search Isn't Enough: Building Graph RAG Systems
tags: Graph RAG, Knowledge Graphs, LLMs, Vector Search
author: Rehan Malik
---

# When Vector Search Isn't Enough: Building Graph RAG Systems

![Graph RAG and Knowledge Graphs for LLMs](../images/graph-rag-and-knowledge-graphs-for-llms.jpg)

## TL;DR

- Vector search is limited for complex queries needing multi-hop or explicit relationships.
- Graph RAG marries knowledge graphs and LLMs for richer retrieval.
- Real Graph RAG systems require hybrid architectures and careful orchestration across retrieval layers.

## Prerequisites

To replicate what I'm explaining, you'll need:

- Python 3.9+
- Neo4j (or similar graph DB)
- Pinecone (or similar vector DB)
- LangChain
- Familiarity with LLMs and graph databases

## Introduction

LLMs have shifted how I approach retrieval. Vector search works well for semantic similarity, but it collapses when queries demand reasoning across multiple relationships or explicit structure. Graph RAG (Retrieval-Augmented Generation with graphs) steps in here, harnessing the structure of knowledge graphs alongside LLM flexibility.

## Technical Deep Dive

I'll show how to stand up a simple Graph RAG system. My go-to stack is Neo4j for the graph, Pinecone for vectors, and LangChain to tie them together.

### Step 1: Setting Up Databases

First, I load up some toy data in Neo4j. The schema is basic: entities and relationships.

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    session.run("""
        CREATE (a:Entity {name: 'Entity A'})
        CREATE (b:Entity {name: 'Entity B'})
        CREATE (c:Entity {name: 'Entity C'})
        CREATE (a)-[:RELATED_TO]->(b)
        CREATE (b)-[:RELATED_TO]->(c)
    """)
```

Make sure you've started Neo4j locally or adjust the connection URI as needed.

### Step 2: Hybrid Retrieval

Now, I combine vector search and graph traversal. Here's a minimal runnable setup:

```python
import pinecone
from langchain.embeddings import OpenAIEmbeddings

# Initialize Pinecone
pinecone.init(api_key='YOUR_API_KEY', environment='us-west1-gcp')
index = pinecone.Index('my-index')
embeddings = OpenAIEmbeddings()

def hybrid_retrieval(query: str):
    # Vector search
    query_embedding = embeddings.embed_query(query)
    vector_results = index.query(query_embedding, top_k=5)['matches']
    vector_names = [match['id'] for match in vector_results]

    # Graph search
    graph_names = []
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query
            RETURN e.name AS name
        """, query=query)
        graph_names = [record['name'] for record in result]
    
    return vector_names, graph_names

query = "Entity A"
vector_results, graph_results = hybrid_retrieval(query)
print("Vector results:", vector_results)
print("Graph results:", graph_results)
```

Note: Pinecone needs an index with vector embeddings already loaded, and `OpenAIEmbeddings` requires credentials.

### Step 3: Augmenting LLMs with Graph RAG

With retrieval done, I need to funnel both sets of results to the LLM. LangChain lets me wrap this logic:

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

class HybridRetriever:
    def __init__(self, vector_results, graph_results):
        self.vector_results = vector_results
        self.graph_results = graph_results

    def get_relevant_documents(self, query):
        docs = []
        for name in self.vector_results + self.graph_results:
            docs.append({"page_content": f"Entity: {name}"})
        return docs

retriever = HybridRetriever(vector_results, graph_results)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
result = qa_chain.run(query)
print("LLM Response:", result)
```

This is a simplified example. In production, you need to handle deduplication, ranking, and richer document construction, but this works for basic connectivity.

## Architecture

Here's how the architecture flows:

1. **User Query** enters the system.
2. **Hybrid Retrieval Layer**: Both Pinecone (vector) and Neo4j (graph) run retrieval.
3. **Orchestration Layer**: LangChain combines the results into context.
4. **LLM Layer**: The context is handed to GPT-4 (or similar).
5. **Response** goes back to the user.

Text diagram:

```
+---------------+
| User Query |
+---------------+
        |
        v
+----------------------+
| Hybrid Retrieval | 
| (Pinecone & Neo4j) |
+----------------------+
        |
        v
+---------------+
| Orchestration |
| (LangChain) |
+---------------+
        |
        v
+---------------+
| LLM (GPT-4) |
+---------------+
        |
        v
+---------------+
| Response |
+---------------+
```

## Lessons Learned

Building Graph RAG systems, I found:

- Real power comes from orchestrating vector and graph retrieval, not just stacking them.
- Tuning parameters for each layer is tricky; relevance and context often trade off.
- The choice of graph and vector DBs matters for scaling and latency.

## Key Takeaways

- Vector search can't handle queries needing explicit relationships or multi-hop reasoning.
- Graph RAG systems need hybrid architectures with tight orchestration.
- Retrieval orchestration is the hardest and most important part.

## Further Reading

If you want to dive more into Graph RAG, procedural knowledge, or LLM orchestration, check out:

- [Reflecting Process Expertise in Procedural Material Generation](http://arxiv.org/abs/2607.13318v1)
- [Agent Hacks Agent: Autoresearch for Production-Agent Red-Teaming](http://arxiv.org/abs/2607.11698v1)
- [RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM](http://arxiv.org/abs/2607.11683v1)

By Rehan Malik

<!-- <script type='application/ld+json'>{"@context":"https://schema.org","@type":"TechArticle","headline":"When Vector Search Isn't Enough: Building Graph RAG Systems","author":{"@type":"Person","name":"Rehan Malik"},"datePublished":"2023-04-01"}</script> -->

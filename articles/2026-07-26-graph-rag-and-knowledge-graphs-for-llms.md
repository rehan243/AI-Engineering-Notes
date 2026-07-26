```yaml
title: Graph RAG and Knowledge Graphs for LLMs 
tags: [Graph RAG, Knowledge Graphs, LLMs, AI, Machine Learning] 
author: Rehan Malik 
```

# When Vector Search Isn't Enough: Building Graph RAG Systems

![Graph RAG and Knowledge Graphs for LLMs](../images/graph-rag-and-knowledge-graphs-for-llms.jpg)

## TL;DR

- Vector search works well for basic retrieval but can't handle multi-hop or relational queries.
- Graph RAG (Retrieval-Augmented Generation) pairs LLMs with knowledge graphs for richer context and reasoning.
- Production systems need careful graph schema design, efficient retrieval, and prompt engineering.
- I show how to use Neo4j with Python for a real Graph RAG workflow.

## Prerequisites

To follow along, you'll need:
- Python 3.9 or newer
- Neo4j running locally or in the cloud
- `neo4j`, `openai`, and `python-dotenv` libraries
- Access to OpenAI GPT models or similar

## Introduction

Vector search changed retrieval, especially alongside LLMs. Embeddings let me find relevant chunks, but as soon as a query gets relational (tracing causality, connecting entities across events), vector search hits limits. For example, questions like "How did leadership changes at Company X affect financial outcomes over the last decade?" require traversing relationships, not just getting the best-matching document.

This is where Graph RAG comes in. I combine knowledge graphs (Neo4j) with LLMs, using the graph for structured retrieval and the LLM for synthesis. I'll walk through setup, architecture, and a hands-on demo.

## Technical Deep Dive

I always start with Neo4j as the graph backend, then use OpenAI's GPT for reasoning. Here's a minimal, runnable workflow.

### Step 1: Define the Knowledge Graph

Set up the schema in Neo4j. This is just a sample for demonstration:

```cypher
CREATE (LeadershipChange:Event {name: "Leadership Change", year: 2020});
CREATE (CEO:Person {name: "John Doe", title: "CEO"});
CREATE (Revenue:Metric {year: 2020, value: "5B USD"});
CREATE (LeadershipChange)-[:INVOLVED]->(CEO);
CREATE (LeadershipChange)-[:IMPACTED]->(Revenue);
```

This snippet defines an event (leadership change), the CEO involved, and the revenue metric impacted.

### Step 2: Querying the Graph with Python

Connect to Neo4j, run Cypher queries, and pass the results to an LLM. Here's a complete example:

```python
from neo4j import GraphDatabase
import openai
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class GraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_graph(self, cypher_query):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

# Setup connection
graph_rag = GraphRAG("bolt://localhost:7687", "neo4j", "password")

cypher_query = """
MATCH (e:Event)-[:INVOLVED]->(p:Person)
MATCH (e)-[:IMPACTED]->(m:Metric)
RETURN e.name AS event, p.name AS person, m.value AS metric
"""

graph_data = graph_rag.query_graph(cypher_query)

# Format data for GPT
openai.api_key = OPENAI_API_KEY
prompt = f"""Given this data:
{graph_data}

Explain how leadership changes affected revenue outcomes."""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=150
)

print(response.choices[0].text.strip())
```

This pulls relational data from Neo4j and lets GPT synthesize an answer. The output depends on model and input, but typically GPT will narrate the influence based on the graph.

### Step 3: Multi-Hop Retrieval

To handle queries that need traversing multiple relationships (like external events affecting leadership changes), expand the Cypher query:

```cypher
MATCH (e:Event)-[:INVOLVED]->(p:Person)
MATCH (e)-[:IMPACTED]->(m:Metric)
MATCH (e)-[:RELATED_TO]->(external:Event)
RETURN e.name AS event, p.name AS person, m.value AS metric, external.name AS external_event
```

Now the graph traversal reaches external events, and the LLM can connect even more dots.

---

## Production Architecture Patterns

Here's how I typically structure a Graph RAG system:

- **Graph Database:** Neo4j or Amazon Neptune for structured knowledge.
- **Embedding Index:** Pinecone, Weaviate, or similar, for unstructured text retrieval.
- **LLM API:** OpenAI, Anthropic, or custom models for reasoning.
- **Orchestrator:** LangChain or custom Python logic to tie everything together.

**Text architecture diagram:**
- Client (chatbot or dashboard) sends a query.
- Orchestrator retrieves graph data (Cypher) and maybe embeddings.
- Results are formatted and sent to LLM.
- LLM synthesizes and the answer goes back to the client.

---

## Common Pitfalls and My Fixes

- **Latency:** Neo4j queries can be slow on big graphs. Index nodes and relationships, cache frequent queries, profile the Cypher.
- **Prompt confusion:** LLMs sometimes misunderstand raw graph data. I use templated prompts, with clear context and relevant fields only.
- **Hardcoded schema:** If I lock the schema too tightly, evolving requirements become painful. Favor flexible node types and relationship labels.
- **Fragmented orchestration:** Mixing graph and vector search without a unified orchestrator leads to inconsistent results. I keep all routing logic in one place.

---

## Lessons Learned

- Graph RAG lets me solve complex reasoning tasks, but only if the graph is well-designed. Overly tangled relationships are a nightmare to maintain and query.
- Prompt engineering is everything. The LLM's reasoning quality depends on how I present the graph data and context.
- Always test traversal logic with real data. Cypher queries can look right but miss key relationships if the data isn't well-structured.

---

## Key Takeaways

1. Graph RAG solves multi-hop, structured retrieval problems that vector search can't.
2. Neo4j works well for graph management, but you must optimize queries as your graph grows.
3. LLM synthesis quality hinges on prompt engineering and how graph data is formatted.

---

## Further Reading

- [From Static Bibliometrics to Dynamic Knowledge Graphs: An LLM-Powered Framework for Modernizing STI Analytics](http://arxiv.org/abs/2607.21327v1)
- [NVIDIA-Labs OO Agents: Native Python Object-Oriented Agents](http://arxiv.org/abs/2607.20709v1)
- [Language-Specific vs Cross-Lingual Knowledge Graphs for Implicit Aspect Identification in Arabic](http://arxiv.org/abs/2607.20056v1)

---

<!-- <script type='application/ld+json'>{"@context":"https://schema.org","@type":"TechArticle","headline":"When Vector Search Isn't Enough: Building Graph RAG Systems","author":{"@type":"Person","name":"Rehan Malik"},"datePublished":"2023-10-17"}</script> -->

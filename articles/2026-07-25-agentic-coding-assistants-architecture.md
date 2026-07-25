---
tags: [ai, ml, agentic-assistants, architecture, coding, engineering]
author: Rehan Malik
---

# How Agentic Coding Assistants Actually Work: Under the Hood

![Agentic Coding Assistants Architecture](../images/agentic-coding-assistants-architecture.jpg)

---

**TL;DR**
- Agentic coding assistants couple LLMs, code synthesis, tool APIs, and orchestration to autonomously tackle coding tasks.
- Real breakthroughs come from chaining, task breakdown, and skill-based communication between modules.
- Most systems break because of fragile prompts, weak tool validation, and clumsy state handling.
- If you want practical scale, focus on Python async orchestration, parsing/validation, and robust error management.

---

## Prerequisites

You need:
- Python 3.10+ (for async and typing)
- Access to OpenAI API or any LLM backend
- Install `openai`, `pydantic`, `aiohttp`, `requests`, `fastapi` (`pip install ...`)
- Familiarity with async Python, REST APIs, and LLM prompt design

---

## Introduction

Agentic coding assistants aren't just autocomplete bots. They read a task, reason about how to break it down, call tools (like APIs or package installers), check their own outputs, and keep iterating until the job is done. This is finally bridging the gap between raw code generation and real, actionable coding support.

Recent papers like [The Autonomous Agency Scale](http://arxiv.org/abs/2607.17947v1) and [SkillComm](http://arxiv.org/abs/2607.11972v1) show the shift from single-shot completions to self-guided workflows. If you're building these systems, you need to know how agents actually reason, orchestrate tools, and avoid dead ends.

---

## Technical Deep Dive

Here's what the core agent loop looks like, and how I'd code it practically.

### 1. Coding Agent Core Loop

The basic loop:

1. Receive user request (e.g., "Write a function to fetch weather data").
2. Decompose the task ("Install requests", "Write fetch_weather", "Handle errors").
3. For each step, generate code or call tools.
4. Validate outputs, retry/refine as needed.
5. Aggregate results for the user.

#### Code Example: Async Agent Task Decomposition

This is a minimal, runnable async agent loop.

```python
import asyncio
from typing import Dict
import openai

openai.api_key = 'YOUR_API_KEY'

async def call_llm(prompt: str) -> str:
    # Run OpenAI API in a thread (since it's blocking)
    response = await asyncio.to_thread(
        lambda: openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful coding assistant."},
                      {"role": "user", "content": prompt}]
        )
    )
    return response.choices[0].message['content']

async def validate_code(code: str) -> bool:
    # Simple forbidden keyword check
    forbidden = ['os.system', 'exec', 'eval']
    return not any(f in code for f in forbidden)

async def agentic_coding_task(user_request: str) -> Dict[str, str]:
    decomposition_prompt = f"Decompose this coding task into steps: {user_request}"
    steps = await call_llm(decomposition_prompt)
    step_list = [s.strip() for s in steps.split('\n') if s.strip()]

    results = {}
    for step in step_list:
        code_prompt = f"Write Python code for: {step}"
        code = await call_llm(code_prompt)
        if not await validate_code(code):
            code = "# Code rejected: forbidden operation detected."
        results[step] = code
    return results

async def main():
    user_request = "Write a function to fetch weather data from OpenWeatherMap API and handle errors."
    results = await agentic_coding_task(user_request)
    for step, code in results.items():
        print(f"Step: {step}\nCode:\n{code}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

*This loop takes a request, breaks it down, generates code per step, and checks for simple forbidden actions. It works as a skeleton for agentic assistants.*

---

### 2. Tool Calling via API Integration

Agents dynamically call APIs/tools. The structure separates tool definitions, calling logic, and orchestration.

#### Code Example: Tool API Registry and Invocation

Here's how I'd build a registry and a basic caller:

```python
import requests
from pydantic import BaseModel

class Tool(BaseModel):
    name: str
    endpoint: str
    method: str

TOOLS = [
    Tool(name='weather_api', endpoint='https://api.openweathermap.org/data/2.5/weather', method='GET'),
]

def call_tool(tool: Tool, params: dict) -> dict:
    if tool.method == 'GET':
        response = requests.get(tool.endpoint, params=params)
        return response.json()
    return {}

# Example usage:
params = {'q': 'London', 'appid': 'your_openweather_api_key'}
weather_tool = TOOLS[0]
result = call_tool(weather_tool, params)
print(result) # Should print dict with weather info
```

*The agent can choose which tool to use, with a clear registry and separation from the orchestration logic.*

---

### 3. Output Parsing and Error Handling

Agents need to parse API/tool outputs and handle failures gracefully.

#### Code Example: Parsing and Error Handling

```python
def parse_weather_output(output: dict) -> str:
    try:
        temp = output['main']['temp']
        description = output['weather'][0]['description']
        return f"Temperature: {temp}C, Description: {description}"
    except (KeyError, IndexError) as e:
        return f"Parsing error: {str(e)}"

# Simulated output
fake_output = {
    'main': {'temp': 20},
    'weather': [{'description': 'clear sky'}]
}
parsed = parse_weather_output(fake_output)
print(parsed) # Prints: Temperature: 20C, Description: clear sky
```

---

## Architecture Patterns

Here's how this fits together in practice.

### Diagram (textual):

- Agents sit in the center.
- User requests come in via HTTP endpoints.
- Each agent instance handles a request, decomposes it, orchestrates LLM calls and tool invocations.
- There's a registry of tools (API endpoints, code sandboxes, installers).
- Outputs are parsed/validated.
- State (intermediate results, retry counts, user context) is tracked per request, often in Redis or a DB.

**Typical flow:**
1. User prompt -> agent instance started
2. Agent calls LLM to break down task
3. For each step:
   - Calls LLM or a tool
   - Validates output
   - Stores results
   - Handles errors (retry, escalate, ask user)
4. Returns aggregated output

**Key components:**
- Async orchestrator (asyncio, Celery, etc)
- Tool registry (class/DB)
- LLM wrapper (OpenAI, Azure, etc)
- Output parser/validator
- State manager (Redis, DB, or local)

---

## Lessons Learned

From my own engineering:

- **Prompting is fragile.** LLMs skip instructions. Always validate their output and restrict dangerous actions.
- **Async is mandatory.** Blocking I/O will stall everything. Use asyncio or proper task queues.
- **APIs/tools fail a lot.** Error responses and timeouts must be handled, or the agent gets stuck. Build retries and parsing in from day one.
- **State is messy.** You need to track retries, intermediate results, and context outside the agent logic (usually Redis or a DB).
- **Skill-driven chaining is powerful but tricky.** SkillComm-style workflows let agents adapt and collaborate, but break easily if state or error handling is weak.

---

## Key Takeaways

- Always check LLM outputs and tool responses before using them.
- Structure agents for decomposition, action, and retries, not just code generation.
- Use async patterns and solid error handling if you care about scale.
- Explicit tool registries and clear output parsing prevent silent failures.
- External state tracking is crucial for reliability and debugging.

---

## Further Reading

- [The Autonomous Agency Scale: A Behavioral Framework for Measuring Self-Directed Behavior in AI Systems](http://arxiv.org/abs/2607.17947v1)
- [Towards Reliable AI-Assisted Analog Design: Template-Constrained LLM Agents for SAR ADC Generation](http://arxiv.org/abs/2607.14165v1)
- [SkillComm: Skill-Driven Semantic Communication for Sequential Workflows via Incremental Token Transmission](http://arxiv.org/abs/2607.11972v1)

---

By Rehan Malik

---

<!-- <script type='application/ld+json'>{"@context":"https://schema.org","@type":"TechArticle","headline":"How Agentic Coding Assistants Actually Work: Under the Hood","author":{"@type":"Person","name":"Rehan Malik"},"datePublished":"2024-06-24"}</script> -->

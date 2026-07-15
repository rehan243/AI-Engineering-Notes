---
tags: AI, Model Context Protocol, Tool Use, LangChain, LlamaIndex 
author: Rehan Malik 
---

# Building MCP-Native AI Applications from Scratch

![Model Context Protocol and Tool Use](../images/model-context-protocol-and-tool-use.jpg)

## TL;DR
* Model Context Protocol (MCP) provides a structured way to manage context for AI models interacting with external tools and systems.
* Building MCP-native applications revolves around context management, tool invocation, and state handling.
* Technologies like LangChain, LlamaIndex, and OpenAI's function calling are central to implementation.
* Challenges include managing context truncation, validating tool input, and handling external API limits.

## Prerequisites
To get the most out of this article, you should have:
- Python 3.9+ installed 
- Libraries: `langchain`, `pydantic`, and `openai` 
- A working understanding of Python and AI model development 

## Introduction
Managing context in AI models becomes significantly harder as applications start interfacing with external tools, databases, and APIs. The Model Context Protocol (MCP) is designed to organize these interactions, ensuring a consistent flow of information between the model and the tools it uses. I'll walk through the process of building MCP-native AI applications, breaking down the concepts with concrete examples.

## Technical Deep Dive

To ground this discussion, I'll start with an example. The task is simple: build a tool to fetch the current weather for a given location, while making it callable by an AI model.

### Step 1: Defining the Tool
LangChain simplifies integrating tools with AI models through decorators and schemas. Here's how I define a weather-fetching tool:

```python
from langchain.tools import tool
from pydantic import BaseModel, ValidationError, validator

# Define input schema for the tool
class WeatherInput(BaseModel):
    location: str

    @validator('location')
    def validate_location(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Location must be a non-empty string.")
        return value

# Register the tool with LangChain
@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    # Simulated weather response
    return f"The weather in {location} is sunny."

# Example usage
if __name__ == "__main__":
    # Valid input
    print(get_weather.run({"location": "New York"})) 

    # Invalid input
    try:
        print(get_weather.run({"location": ""}))
    except ValidationError as e:
        print(f"Validation Error: {e}")
```

### Explanation:
1. **Input Validation**: I use Pydantic's `BaseModel` and validators to ensure the tool receives valid input.
2. **Tool Registration**: The `@tool` decorator registers the function with LangChain, making it callable by an AI agent.

### Step 2: Integrating the Tool with an Agent
Once a tool is defined, the next step is to integrate it into an agent using LangChain's `AgentExecutor`.

```python
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

# Define the tool in LangChain's Tool format
weather_tool = Tool(
    name="get_weather",
    func=get_weather.run,
    description="Fetches the current weather for a given location."
)

# Define the AI model (LLM)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize the agent with the tool
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent="zero-shot-react-description", # Uses a predefined LLM prompting strategy
    verbose=True
)

# Using the agent to invoke the tool
if __name__ == "__main__":
    user_input = "What's the weather in London?"
    response = agent.run(user_input)
    print(response)
```

### Key Points Here:
1. The `Tool` wrapper provides metadata about the tool, like its name and description. This metadata is used to inform the AI about the tool's purpose and how to use it.
2. The `initialize_agent` function ties everything together, enabling the AI model to dynamically invoke the tool in response to input.

### Step 3: Managing Context
MCP is fundamentally about managing context effectively (what the model knows and "remembers"). In LangChain, the `memory` module can be used to store and recall context.

```python
from langchain.memory import ConversationBufferMemory

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Pass memory into the agent
agent_with_memory = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)

# Simulate a conversation with context
if __name__ == "__main__":
    agent_with_memory.run("What's the weather in Paris?")
    agent_with_memory.run("How about in Berlin?")
```

Here, the `ConversationBufferMemory` tracks previous interactions and makes them available to the agent, allowing for better contextual responses.

## Architecture

MCP-native applications typically follow a layered architecture. Here's a simple breakdown:

### 1. Presentation Layer
Handles user interactions. This could be an app, website, or CLI interface. For instance, user inputs like "What's the weather in New York?" are routed here.

### 2. Orchestration Layer
Manages the AI model, tools, and context. This is the layer where frameworks like LangChain or custom MCP implementations reside. It ensures:
- Context is properly tracked (e.g., via memory modules).
- Tools are invoked as needed based on the model's decisions.

### 3. Data Layer
Handles external data, APIs, and storage systems. For example:
- A weather API for fetching live data.
- A database for storing user interactions or results.

**ASCII Diagram:**
```
+-------------------+
| Presentation |
| Web/Mobile/CLI |
+-------------------+
         |
         v
+-------------------+
| Orchestration |
| LangChain/Agent |
+-------------------+
         |
         v
+-------------------+
| Data Layer |
| APIs/DBs/Storage |
+-------------------+
```

## Lessons Learned
Building MCP-native AI applications is not without challenges. Here's what I've learned:
1. **Context Truncation**: LLMs have context length limits, so you need strategies like summarization or chunking to manage long conversations.
2. **Schema Validation**: Without strict validation, invalid or unexpected inputs can break your tools. Pydantic makes this straightforward.
3. **Rate Limits**: Many APIs have strict usage policies. Implement caching and retry logic to avoid hitting limits during high-traffic periods.
4. **Monitoring**: Always log invocations and responses. Debugging tool behavior requires visibility into what's happening at each step.

## Key Takeaways
- **LangChain and MCP**: LangChain simplifies the complexities of MCP. Use its tools, agents, and memory modules to manage context effectively.
- **Validation and Error Handling**: Valid input is critical for reliable tool use; always validate your schemas.
- **Context Planning**: Anticipate and mitigate issues around context length and memory constraints.

## Further Reading
If you want to dig deeper, here are some useful resources:
- ["Track, Rank, Crack: Epistemic Working Memory Scales Multi-Hop Reasoning in Language Agents"](http://arxiv.org/abs/2607.12267v1) 
- ["RCWT: Measuring Task-Budget Displacement from Coordination Content in LLM Calls"](http://arxiv.org/abs/2607.12216v1) 
- ["TerraRepair: A Tool-Grounded LLM Agent for Infrastructure-as-Code Repair"](http://arxiv.org/abs/2607.11390v1) 

By Rehan Malik

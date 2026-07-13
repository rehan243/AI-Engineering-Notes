---
tags: [MCP, AI agents, Tool Use, Protocols, Python, Architecture]
author: Rehan Malik
---

# Building MCP-Native AI Applications from Scratch: Protocol, Tools, and Hands-On Code

![Model Context Protocol and Tool Use](../images/model-context-protocol-and-tool-use.jpg)

## TL;DR

- MCP (Model Context Protocol) structures model interaction with external tools and APIs, using explicit schemas and JSON calls.
- The protocol enables tool-augmented models and agentic workflows, boosting reliability and context management.
- Building MCP-native apps means defining tools, handling structured model-driven calls, and returning results safely.
- I'll show practical code for tool wiring, JSON-based calls, and security handling.

## Prerequisites

You'll need:

- Python 3.10+
- `openai` or `llama-cpp-python` (for LLMs)
- `pydantic` (`pip install pydantic`)
- FastAPI (`pip install fastapi uvicorn`)
- Familiarity with function calling in LLM APIs

## Introduction: Why MCP and Tool Use Matter

AI models only get useful when they can operate real-world tools. The Model Context Protocol (MCP) is a standard for structuring these interactions. MCP lets models request operations on tools in a predictable, safe way, handling responses and context as tasks change. This is critical for agentic systems (AutoGPT, LangChain agents) where reliability and composability need real engineering.

Most function-calling APIs and agent frameworks converge on: tool definitions, structured calls, and tool results. MCP formalizes this pattern. If you're building with open-source models or cloud APIs, going MCP-native gives you better reasoning, safety, and debugging.

## Technical Deep Dive

I focus on the full stack: defining tools, receiving and validating calls, executing, and returning results. All code is real and runnable.

### Defining Tools MCP-Style

I use Pydantic for schema definition. This keeps parameter types explicit and validation tight.

```python
from pydantic import BaseModel, Field
from typing import List

class ToolParameter(BaseModel):
    name: str
    description: str
    type: str
    required: bool

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    scopes: List[str] = Field(default_factory=list)

# Example: calculator tool definition
calculator_tool = ToolDefinition(
    name="calculator",
    description="Perform arithmetic operations",
    parameters=[
        ToolParameter(
            name="operation",
            description="Type of operation: add, subtract, multiply, divide",
            type="string",
            required=True
        ),
        ToolParameter(
            name="a",
            description="First number",
            type="float",
            required=True
        ),
        ToolParameter(
            name="b",
            description="Second number",
            type="float",
            required=True
        ),
    ],
    scopes=["math"]
)

print(calculator_tool.json(indent=2))
```

This reflects MCP's tool registration: explicit names, parameter types, scopes for permission.

### Handling Model Tool Calls

When models output structured tool calls (often JSON), I validate and dispatch them.

```python
from pydantic import BaseModel, ValidationError

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict

# Simulated model tool call as JSON
model_call_json = '''
{
  "tool_name": "calculator",
  "arguments": {
    "operation": "multiply",
    "a": 7.5,
    "b": 2
  }
}
'''

try:
    call = ToolCall.parse_raw(model_call_json)
    print(f"Parsed tool call: {call}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

Strict schema checks here are crucial. Sometimes I add pre-processing to fix parameter names when models hallucinate.

### Executing and Returning Tool Results

Here's a FastAPI endpoint for agentic flows. The server receives a tool call, executes, and returns a result MCP-style.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CalculatorArgs(BaseModel):
    operation: str
    a: float
    b: float

class ToolResult(BaseModel):
    result: float
    success: bool
    error: str = ""

@app.post("/tool/calculator")
def calculator_tool_endpoint(args: CalculatorArgs):
    try:
        if args.operation == "add":
            res = args.a + args.b
        elif args.operation == "subtract":
            res = args.a - args.b
        elif args.operation == "multiply":
            res = args.a * args.b
        elif args.operation == "divide":
            if args.b == 0:
                raise ValueError("Division by zero")
            res = args.a / args.b
        else:
            raise ValueError(f"Unknown operation {args.operation}")
        return ToolResult(result=res, success=True)
    except Exception as e:
        return ToolResult(result=0, success=False, error=str(e))

# Run with: uvicorn myfile:app
# Test with: curl -X POST "http://127.0.0.1:8000/tool/calculator" -H "Content-Type: application/json" -d '{"operation": "multiply", "a": 7.5, "b": 2}'
```

Agentic pipelines call this endpoint when the model emits a calculator tool call. If I integrate with frameworks (LangChain, Semantic Kernel), the flow is: define, validate, execute, return MCP-compliant results.

## Architecture: MCP Native Agentic Application

Here's how the stack fits together, in ASCII:

```
+---------------+ +---------------------+ +-------------------+
| AI Model |-->| MCP Tool Call |-->| Tool Server |
| (LLM, agent) | | (JSON/function) | | (FastAPI, etc) |
+---------------+ +---------------------+ +-------------------+
        ^ ^ ^
        | <--- Tool Result (JSON) --- |
+--------------------------------------------------------------+
| Context Management / Agent Memory |
+--------------------------------------------------------------+
```

- AI model (LLM or agent) emits a tool call in MCP format (JSON).
- Agent stack parses and validates, checks permissions, dispatches to the tool server (could be a microservice).
- Tool server executes, returns structured results (JSON, per MCP spec).
- Agent updates memory/context and optionally prompts the model again.

This separation gives auditability, testability, and safety, especially when tools have elevated access.

## Lessons Learned

From real MCP pipelines and integrating with LLMs, here's what trips me up and what solves it:

- **Schema drift is frequent.** Models invent parameter names or types. Strict Pydantic schemas help, but sometimes I need pre-processing to fix fields.
- **Security is non-negotiable.** Always describe tool permissions and scopes explicitly. Never let the model call arbitrary endpoints or code.
- **Context evolution can blow up.** For long agentic tasks, context grows fast. Scoped verification helps but sometimes I have to prune memory or chunk context.
- **OpenAI function calling isn't always portable.** For open-source models, custom prompting is needed to make them output structured tool calls. I start with simple instructions: output JSON with these fields.
- **Debugging tool calls is way easier with structure.** Log every call and result. When something fails, structured formats make diagnosis fast.

## Key Takeaways

1. **Define tools with explicit schemas and scopes.**
2. **Validate all model inputs, fail gracefully, log errors.**
3. **Build tool execution as endpoints or microservices, not inline code.**
4. **Audit tool calls and memory evolution, especially for long tasks.**
5. **Never allow the model to call arbitrary code or endpoints: permissions and scope checks matter.**

## Further Reading

- [Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift](http://arxiv.org/abs/2607.09175v1) 
- [Present but Rescaled: Chat-to-Agent Transfer of Additive Activation Steering](http://arxiv.org/abs/2607.09156v1)
- [Mitigating Taint-Style Vulnerabilities in MCP Servers via Security-Aware Tool Descriptions](http://arxiv.org/abs/2607.07461v1)

By Rehan Malik

<!-- <script type='application/ld+json'>{"@context":"https://schema.org","@type":"TechArticle","headline":"Building MCP-Native AI Applications from Scratch: Protocol, Tools, and Hands-On Code","author":{"@type":"Person","name":"Rehan Malik"},"datePublished":"2024-06-19"}</script> -->

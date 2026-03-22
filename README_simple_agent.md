# simple-agent.py

A minimal but complete LangGraph agent that searches the web and summarizes results. This is the starting point for understanding how modern AI agents work — the agent decides what to do, calls tools, observes results, and loops until it has an answer. You write zero orchestration logic.

## What it does

The agent takes a user question, decides to call the Tavily web search tool, gets back search results, then calls a custom summarize tool to condense those results into three bullet points. It outputs a final synthesized answer. The key thing to understand is that you do not tell the agent to search first and summarize second — it figures that out on its own based on the tool descriptions.

## How it works

The LLM is Groq's hosted Llama 3.3 70B. It supports native function calling, which means the model outputs structured JSON that LangGraph intercepts and routes to the right tool. This is more reliable than older ReAct-style agents that parsed tool calls out of free-form text.

`create_react_agent` is a single function from `langgraph.prebuilt` that replaces the old `AgentExecutor` from `langchain.agents`. LangChain deprecated `AgentExecutor` in version 1.0. If you see import errors for `AgentExecutor`, this is why.

The `@tool` decorator turns a regular Python function into something the agent can call. The docstring is not just documentation — it is the description the LLM reads to decide when and how to call the function. Write it clearly and specifically.

`agent.invoke()` runs the full ReAct loop synchronously. If you need streaming output, replace it with `agent.stream()`.

## Setup

```
pip install langgraph langchain-groq langchain-community tavily-python python-dotenv
```

Create a `.env` file:

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

Get a free Groq key at console.groq.com. Get a free Tavily key at tavily.com — the free tier gives 1,000 searches per month.

## Groq free tier limits

| Model | Tokens per minute | Tokens per day |
|---|---|---|
| llama-3.3-70b-versatile | 12,000 | 100,000 |
| llama-3.1-8b-instant | 6,000 | 500,000 |

Each agent run uses roughly 1,000 to 3,000 tokens. Use `llama-3.1-8b-instant` during development and switch to the 70B model for final runs.

## Changing the question

Edit the `content` field in the final `agent.invoke()` call. That is the only thing you need to change to ask a different question.

## What to try next

Add a Wikipedia tool from `langchain-community`. Enable streaming with `.stream()` instead of `.invoke()`. Pass conversation history across multiple calls to give the agent memory.

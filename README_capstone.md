# capstone.py

A multi-framework pipeline that combines LangChain, LlamaIndex, and CrewAI into a single workflow. A CrewAI manager agent orchestrates four workers: one uses a LangChain tool for live web search, one uses LlamaIndex tools for PDF retrieval and SQL queries over CSV data, one synthesizes the combined outputs, and one writes the final report. Each framework handles the task it is best suited for rather than forcing one framework to do everything.

## Prerequisites

Run `multi_doc_research.py` at least once to build the vector index in `./storage/pdf_vector/`. The pipeline reuses that index.

## What it does

On startup, LlamaIndex settings are initialized once globally to avoid a PyTorch meta tensor error that occurs when the embedding model is loaded multiple times in parallel. The crew then runs a hierarchical process where the manager delegates tasks to workers in sequence. The final report is saved to `./output/capstone_report.md`. If a rate limit is hit, the script waits and retries automatically up to three times.

## Why three frameworks

LangChain's Tavily integration is the simplest way to add live web search to any agent system. LlamaIndex's vector index and SQL query engine are purpose-built for document retrieval and structured data queries — they handle chunking, embedding, storage, and retrieval with much less setup than building these from scratch in LangChain or CrewAI. CrewAI's hierarchical crew provides the orchestration layer that coordinates when each tool is called and how the outputs are combined.

## How the integration works

LangChain and LlamaIndex tools are exposed to CrewAI as `BaseTool` subclasses. Each tool's `_run()` method calls into the respective framework. CrewAI does not need to know anything about the underlying framework — it just calls `_run()` and gets a string back. This pattern means you can wrap any Python function, API call, or framework as a CrewAI tool without any framework-specific adapters.

**Why LlamaIndex settings are loaded at module level.** The PDF tool and SQL tool both use HuggingFace embeddings. When CrewAI runs multiple tool calls concurrently, both tools try to initialize the embedding model simultaneously. PyTorch cannot copy an uninitialized meta tensor this way and raises an error. Loading `LISettings.embed_model` once at the top of the file before any tools run means both tools reuse the already-loaded model.

**Hierarchical vs sequential process.** In `Process.sequential`, tasks have explicit agent assignments and run in a fixed order. In `Process.hierarchical`, tasks have no agent assignment — the manager reads each agent's role and goal and decides who to give each task to. The manager also validates outputs before proceeding to the next task. This makes the crew more flexible but uses roughly three times as many LLM calls.

**Why `planning=False`.** CrewAI's `planning=True` feature creates an internal planning agent that defaults to OpenAI. There is no way to override this without disabling planning entirely. The manager agent's backstory provides equivalent structure by telling it to follow strict delegation steps.

**Rate limit handling.** The `task_callback` function sleeps 20 seconds after each task completes. Combined with `max_rpm=5`, this spaces out the LLM calls enough to stay under Groq's per-minute token limit. The outer retry loop handles the rare case where a burst still triggers a 429 error.

## Setup

```
pip install crewai crewai-tools litellm langchain-community llama-index-core llama-index-llms-groq llama-index-embeddings-huggingface sqlalchemy pandas python-dotenv
```

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

## Changing the topic

Set `TOPIC` near the middle of the file. All four tasks reference this variable. The output file is always `./output/capstone_report.md`.

## Expected runtime

Each task sleeps 20 seconds after completion. With four tasks plus manager overhead, expect 3 to 5 minutes per run on Groq's free tier. On a paid tier with higher rate limits, remove the `task_callback` and raise `max_rpm` to 30.

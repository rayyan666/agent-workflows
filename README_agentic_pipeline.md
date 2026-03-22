# agentic_pipeline.py

An agentic query pipeline that builds on the multi-document RAG setup and adds four new capabilities: sub-question decomposition, multi-step reasoning over multiple indexes, SQL queries over CSV data, and persistent conversation memory. This represents the difference between a retrieval system and an agent — the agent plans, chooses tools, and reasons across sources autonomously.

## Prerequisites

Run `multi_doc_research.py` at least once before running this script. The pipeline reuses the indexes saved to `./storage/` — it does not rebuild them.

## What it does

The script loads the saved indexes, loads the CSV into an in-memory SQLite database, sets up four tools, and starts an interactive agent that maintains conversation history across queries. The four demo queries at startup showcase each concept, then the script drops into a live session where you can ask anything.

## How each concept works

**Sub-question decomposition.** `SubQuestionQueryEngine` takes a complex question, uses the LLM to break it into sub-questions, routes each sub-question to the appropriate tool, and synthesizes the answers into a single response. Watch the verbose output — you will see lines like `[pdf_research] Q: What are diffusion models?` showing exactly how the question was decomposed and routed.

**SQL over CSV.** `NLSQLTableQueryEngine` translates natural language into SQL, executes it against a SQLite table, and returns the result. This is far more accurate than vector search for aggregation questions like "total GPU hours" or "which project has the highest cost." The verbose output prints the generated SQL so you can learn from it.

**Multi-step reasoning.** `AgentWorkflow` can call multiple tools in sequence within a single query. When you ask about both the PDF content and the billing data in one question, the agent calls `pdf_details`, then `billing_sql`, then synthesizes both results. You do not orchestrate this — the agent decides the sequence.

**Conversation memory.** `ChatMemoryBuffer` stores the conversation history up to a token limit. The `Context` object carries this history across multiple `agent.run()` calls. When you ask "based on what you just told me...", the agent has the full prior exchange available and does not need to re-query.

**Why `LLMQuestionGenerator` instead of the default.** `SubQuestionQueryEngine.from_defaults()` tries to import an OpenAI-specific question generator. Since this setup uses Groq, you must pass a `LLMQuestionGenerator` explicitly. This uses your existing Groq LLM to generate sub-questions.

**Why `asyncio.run()` instead of `asyncio.new_event_loop()`.** `AgentWorkflow.run()` is a coroutine that internally calls `asyncio.create_task()`, which requires a running event loop. `asyncio.new_event_loop()` creates a loop but does not set it as running. `asyncio.run()` creates, starts, and closes the loop correctly.

## Setup

```
pip install llama-index-core llama-index-llms-groq llama-index-embeddings-huggingface sqlalchemy pandas python-dotenv
```

```
GROQ_API_KEY=gsk_...
```

## Changing the data

The column names from the CSV are printed at startup. Update the `billing_sql` tool description to match your actual column names — the LLM reads this description to decide whether to use the tool and to understand what data it contains.

## Good questions to ask

"Which project has the highest total GPU hours?" triggers a SQL aggregation with GROUP BY. "What is the mathematical formulation for the forward diffusion process?" triggers a specific PDF vector search. "Compare what the PDF says about DDPM with what Wikipedia says about generative models" triggers sub-question decomposition across two sources. Chaining two questions where the second references the first tests the memory.

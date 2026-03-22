# research_crew.py

A three-agent sequential crew that researches a topic, analyzes the findings, and writes a formatted report. This introduces CrewAI's core model: multiple specialized agents with defined roles, each handling a distinct phase of work, with outputs flowing from one agent to the next.

## What it does

A researcher agent uses web search and PDF semantic search to gather material. An analyst agent receives that material and extracts insights, comparisons, and practical implications. A writer agent turns the analysis into a structured Markdown report and saves it to `./output/report.md`. Each agent has a role, goal, backstory, and tool list that shapes how it approaches its task.

## Prerequisites

Run `multi_doc_research.py` at least once to build the vector index in `./storage/pdf_vector/`. The `PDFSearchTool` reuses that index.

## How it works

**Agents vs tasks.** An agent defines who is doing the work — their expertise, goals, and available tools. A task defines what needs to be done and what the output should look like. Agents are reusable across tasks. Tasks are specific to a run.

**Sequential process.** `Process.sequential` runs tasks in order. Each task receives the output of the previous task as context. This is the simplest coordination model — no manager, no dynamic delegation.

**Backstory matters.** The backstory is not just flavor text. It is part of the system prompt the LLM receives when executing a task. A researcher with a "meticulous, cross-reference everything" backstory produces different output than a generic assistant. Think of it as persona-level prompting.

**Tool descriptions for routing.** Each tool's `description` field is read by the agent to decide when to call it. If the description is too vague, the agent either calls the wrong tool or skips it. The descriptions here specify the input format, output format, and the type of questions each tool is suited for.

**`context=[previous_task]`.** Adding a task to another task's `context` list means the second task's prompt includes the full output of the first task. Without this, the analyst would not see the researcher's findings.

**`output_file`.** Setting `output_file` on a task tells CrewAI to save the task's final output to that path automatically, regardless of whether the agent uses a file writing tool. The writer agent also uses `FileWriterTool` explicitly, which gives it more control over the filename and content.

**`memory=False`.** CrewAI's built-in memory feature requires an OpenAI key to power the memory analysis agent. With Groq, set `memory=False` on all agents and the crew. Task context passing via `context=[]` handles everything needed for this sequential pipeline.

## Setup

```
pip install crewai crewai-tools litellm langchain-community python-dotenv
```

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

LiteLLM is required for Groq support in CrewAI. Without it, CrewAI cannot resolve the `groq/` model prefix.

## Changing the topic

Set `TOPIC` near the bottom of the file. The researcher, analyst, and writer tasks all reference this variable. The output file is always `./output/report.md`.

## Rate limits

This crew makes roughly 6 to 10 LLM calls per run depending on how many tool calls the researcher makes. At Groq's 25 RPM limit, a typical run completes in under two minutes. If you hit rate limits, lower `max_rpm` on the crew.

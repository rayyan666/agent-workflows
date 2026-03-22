# multi_doc_research.py

A multi-source RAG (Retrieval-Augmented Generation) assistant that indexes PDF files, web pages, and CSV data, then routes each question to the most relevant source. This is the foundation of most production AI applications — instead of sending raw user questions to an LLM, you retrieve relevant context from your own data first.

## What it does

On first run, the script loads your documents, splits them into chunks, embeds each chunk using a local HuggingFace model, and saves the vector indexes to disk. On every subsequent run, it loads from disk in seconds with no API calls. When you ask a question, a `RouterQueryEngine` reads the question, uses the LLM to pick the best source, retrieves the top matching chunks, and synthesizes an answer. Each answer includes source citations with page numbers.

## Data sources

Place your PDF files in `./data/`. The script also fetches four Wikipedia pages on AI topics and loads a CSV from `./data/IndiaAI_BillingAndUsageDailyData.csv`. Change the URLs and CSV path to point at your own data.

## How it works

`SimpleDirectoryReader` loads each PDF page as a separate document with metadata including the filename and page number. The page number is stored as `page_label` internally — the script copies it to `page_number` so citations are readable.

`SentenceSplitter` breaks documents into overlapping chunks before indexing. `chunk_size=512` means each chunk is roughly 512 tokens. `chunk_overlap=50` means adjacent chunks share 50 tokens of context, so a sentence split across a boundary appears in both chunks. These two values are the biggest levers for retrieval quality.

`VectorStoreIndex` embeds each chunk and stores the vectors in memory. At query time it finds the top-k most semantically similar chunks to your question and sends them to the LLM as context. `SummaryIndex` does not use embeddings — it chains all chunks together and is better for broad "summarize everything" questions.

`RouterQueryEngine` wraps multiple query engines as tools. The tool descriptions are what the LLM reads to decide which engine to use. Vague descriptions cause wrong routing — be specific about what data each engine covers.

`LiveSearchQueryEngine` is a custom engine that wraps Tavily search. It exists because `RouterQueryEngine` only accepts `QueryEngineTool` objects, not `FunctionTool`. Wrapping Tavily in a `CustomQueryEngine` makes it compatible.

Embeddings use `BAAI/bge-small-en-v1.5` from HuggingFace. This model downloads once (~130MB), runs entirely offline, and requires no API key.

## Setup

```
pip install llama-index-core llama-index-llms-groq llama-index-embeddings-huggingface llama-index-readers-file llama-index-readers-web langchain-community beautifulsoup4 pandas pypdf tavily-python python-dotenv
```

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

## Index persistence

Indexes are saved to `./storage/`. Subsequent runs skip embedding entirely. If you add new documents or change your data, delete the relevant subfolder in `./storage/` and let it rebuild. The `_load_or_build` function handles this automatically.

## Relevance scores

The score next to each citation is cosine similarity between your query embedding and the chunk embedding. Scores above 0.75 indicate a strong match. Scores below 0.60 suggest the retrieved chunks are only loosely related to your question, which often means the chunk size is too coarse or the query is too broad.

## Rate limits

The `SummaryIndex` with `tree_summarize` mode fires many LLM calls in parallel and hits Groq's per-minute token limit quickly on large PDFs. The script uses `compact` mode instead, which batches chunks into fewer calls. If you still hit rate limits, reduce `similarity_top_k` from 3 to 2.

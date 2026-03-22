import os
import pandas as pd
from langchain_community.tools.tavily_search import TavilySearchResults
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine, CustomQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

pdf_docs = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data()
for doc in pdf_docs:
    page_label = doc.metadata.get("page_label")
    if page_label:
        doc.metadata["page_number"] = int(page_label)
    doc.metadata["source_type"] = "pdf"
print(f"Loaded {len(pdf_docs)} PDF pages")

web_docs = BeautifulSoupWebReader().load_data(urls=[
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
])
print(f"Loaded {len(web_docs)} web pages")

df = pd.read_csv("./data/IndiaAI_BillingAndUsageDailyData.csv")
csv_docs = [
    Document(text=row.to_string(), metadata={"source": "csv", "row": i})
    for i, row in df.iterrows()
]
print(f"Loaded {len(csv_docs)} CSV rows")

def _load_or_build(persist_dir, docs, label):
    if os.path.exists(persist_dir):
        print(f"Loading {label} index from {persist_dir}...")
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))
    print(f"Building {label} index...")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir)
    return index

pdf_vector_index = _load_or_build("./storage/pdf_vector", pdf_docs, "PDF vector")
web_index        = _load_or_build("./storage/web", web_docs, "web")
csv_index        = _load_or_build("./storage/csv", csv_docs, "CSV")

pdf_summary_index = SummaryIndex.from_documents(pdf_docs)

pdf_summary_engine = pdf_summary_index.as_query_engine(response_mode="compact")
pdf_vector_engine  = pdf_vector_index.as_query_engine(similarity_top_k=3)
web_engine         = web_index.as_query_engine(similarity_top_k=3)
csv_engine         = csv_index.as_query_engine(similarity_top_k=3)

class LiveSearchQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str: str) -> str:
        results = TavilySearchResults(
            max_results=3,
            api_key=os.environ["TAVILY_API_KEY"],
        ).invoke(query_str)
        content = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])
        response = Settings.llm.complete(
            f"Answer this question using the search results below.\n\n"
            f"Question: {query_str}\n\nSearch results:\n{content}"
        )
        return str(response)

live_search_engine = LiveSearchQueryEngine()

tools = [
    QueryEngineTool.from_defaults(
        query_engine=pdf_summary_engine,
        name="pdf_overview",
        description="Use for broad questions about the PDFs: summaries, overviews, topic coverage.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=pdf_vector_engine,
        name="pdf_details",
        description="Use for specific questions about PDF content: formulas, methods, results, definitions.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=web_engine,
        name="web_knowledge",
        description="Use for general AI/ML knowledge: machine learning, reinforcement learning, neural networks.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=csv_engine,
        name="structured_data",
        description="Use for questions about the IndiaAI billing CSV: GPU usage, instances, daily hours, trends.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=live_search_engine,
        name="live_web_search",
        description="Use as a last resort for questions not covered by PDFs, web index, or CSV. Searches the live web.",
    ),
]

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
    verbose=True,
)

questions = [
    "Give me a high-level summary of the PDF documents.",
    "What is the mathematical formulation used for diffusion?",
    "What is reinforcement learning?",
    "What trends do you see in the CSV data?",
    "What is the latest version of LlamaIndex?",
]

for q in questions:
    print(f"\n{'─'*60}")
    print(f"Q: {q}")
    response = router_engine.query(q)
    print(f"A: {response}")
    if hasattr(response, "source_nodes"):
        seen = set()
        for node in response.source_nodes:
            src      = node.metadata.get("file_name") or node.metadata.get("source", "web")
            page     = node.metadata.get("page_number")
            score    = f"{node.score:.2f}" if node.score is not None else "n/a"
            page_str = f" | Page: {page}" if page else ""
            key      = f"{src}{page_str}"
            if key not in seen:
                seen.add(key)
                print(f"   Source: {src}{page_str} | Score: {score}")
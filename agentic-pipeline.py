import os
import asyncio
import pandas as pd
from sqlalchemy import create_engine
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SQLDatabase,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine, NLSQLTableQueryEngine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

print("Loading indexes from ./storage/ ...")
pdf_vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage/pdf_vector"))
web_index        = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage/web"))
csv_index        = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage/csv"))
print("All indexes loaded")

df = pd.read_csv("./data/IndiaAI_BillingAndUsageDailyData.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True)
sql_engine = create_engine("sqlite:///:memory:")
df.to_sql("billing", sql_engine, if_exists="replace", index=False)
sql_database    = SQLDatabase(sql_engine, include_tables=["billing"])
sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["billing"], verbose=True)

pdf_engine = pdf_vector_index.as_query_engine(similarity_top_k=3)
web_engine = web_index.as_query_engine(similarity_top_k=3)

sub_question_tools = [
    QueryEngineTool.from_defaults(
        query_engine=pdf_engine,
        name="pdf_research",
        description="Use for specific questions about the PDF research papers: diffusion models, architectures, formulas, methods.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=web_engine,
        name="web_knowledge",
        description="Use for general AI/ML concepts from Wikipedia: machine learning, reinforcement learning, neural networks.",
    ),
]

question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=sub_question_tools,
    question_gen=question_gen,
    verbose=True,
    use_async=False,
)

agent_tools = [
    QueryEngineTool.from_defaults(
        query_engine=sub_question_engine,
        name="multi_source_research",
        description="Use for complex questions that need information from both PDF papers and web knowledge combined.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=pdf_engine,
        name="pdf_details",
        description="Use for specific questions about PDF content: formulas, results, architectures, sections.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="billing_sql",
        description="Use for any question about the IndiaAI billing CSV: total GPU hours, costs, instance types, usage trends.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=web_engine,
        name="web_knowledge",
        description="Use for general AI/ML concepts: machine learning, reinforcement learning, neural networks.",
    ),
]

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=agent_tools,
    llm=Settings.llm,
    verbose=True,
    timeout=120,
    system_prompt=(
        "You are an expert research assistant with access to AI research papers, "
        "web knowledge, and a billing database. Follow these rules strictly: "
        "(1) Always answer directly. "
        "(2) For SQL questions, use billing_sql and report the exact number. "
        "(3) For PDF questions, quote or paraphrase the actual retrieved content with citations. "
        "(4) For comparison questions, structure your answer: From the PDFs / From Wikipedia / Key differences. "
        "(5) For memory questions, refer back to specific facts from earlier in the conversation. "
        "(6) End every answer with the source tool used in brackets."
    ),
)

ctx = Context(agent)

async def _chat_async(query: str) -> str:
    result = await agent.run(query, ctx=ctx)
    return str(result)

def chat(query: str) -> str:
    return asyncio.run(_chat_async(query))

print("\n" + "═" * 60)
print("Project 4 — Agentic Query Pipeline")
print("═" * 60)

demos = [
    ("Sub-question decomposition",    "Compare what the PDF says about diffusion models with what Wikipedia says about generative AI."),
    ("SQL over CSV",                   "What is the total number of GPU hours used across all billing records?"),
    ("Multi-step reasoning",           "What AI techniques are in the PDFs, and which billing project has the highest daily GPU usage?"),
    ("Memory",                         "Based on the GPU usage you just told me, which project is likely training the largest model?"),
]

for concept, query in demos:
    print(f"\n{'─' * 60}")
    print(f"[{concept}]")
    print(f"You: {query}")
    print(f"\nAgent: {chat(query)}")

print("\n" + "═" * 60)
print("Interactive session — type 'quit' to exit")
print("═" * 60)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue
    print(f"\nAgent: {chat(user_input)}")

print("\nSession ended.")
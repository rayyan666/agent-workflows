import os
import time
import pandas as pd
from sqlalchemy import create_engine
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from llama_index.core import Settings as LISettings
from llama_index.llms.groq import Groq as LIGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

LISettings.llm = LIGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"], temperature=0)
LISettings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2,
)

class LangChainSearchTool(BaseTool):
    name: str = "Live Web Search"
    description: str = (
        "Search the live web for current information on any topic. "
        "Input: a search query. Output: top results with URLs and content."
    )
    def _run(self, query: str) -> str:
        from langchain_community.tools.tavily_search import TavilySearchResults
        results = TavilySearchResults(max_results=3, api_key=os.environ["TAVILY_API_KEY"]).invoke(query)
        return "\n\n".join([f"[{r['url']}]\n{r['content']}" for r in results])

class LlamaIndexPDFTool(BaseTool):
    name: str = "PDF Knowledge Search"
    description: str = (
        "Search local AI research PDF papers for concepts, formulas, methods, and findings. "
        "Input: a specific question. Output: relevant passages with page citations."
    )
    def _run(self, query: str) -> str:
        from llama_index.core import StorageContext, load_index_from_storage
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage/pdf_vector"))
        engine = index.as_query_engine(similarity_top_k=3)
        response = engine.query(query)
        citations = [
            f"[{n.metadata.get('file_name','pdf')} p.{n.metadata.get('page_number','?')}]"
            for n in response.source_nodes
        ]
        return f"{response}\n\nSources: {', '.join(citations)}"

class LlamaIndexSQLTool(BaseTool):
    name: str = "Billing Data SQL Query"
    description: str = (
        "Query the IndiaAI billing and usage CSV data using natural language. "
        "Handles total GPU hours, costs by project, usage trends, instance types. "
        "Input: a data question. Output: exact numbers from SQL."
    )
    def _run(self, query: str) -> str:
        from llama_index.core import SQLDatabase
        from llama_index.core.query_engine import NLSQLTableQueryEngine
        df = pd.read_csv("./data/IndiaAI_BillingAndUsageDailyData.csv")
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True)
        engine = create_engine("sqlite:///:memory:")
        df.to_sql("billing", engine, if_exists="replace", index=False)
        sql_db = SQLDatabase(engine, include_tables=["billing"])
        sql_engine = NLSQLTableQueryEngine(sql_database=sql_db, tables=["billing"], verbose=False)
        return str(sql_engine.query(query))

web_researcher = Agent(
    role="Web Research Specialist",
    goal="Find the most current and relevant information from the web about AI topics using live search.",
    backstory=(
        "You are an expert at finding cutting-edge AI research and developments online. "
        "You use web search to find the latest papers, benchmarks, and real-world applications. "
        "You always include source URLs."
    ),
    tools=[LangChainSearchTool()],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=3,
)

data_analyst = Agent(
    role="AI Data & Research Analyst",
    goal="Extract precise insights from local research papers and billing data using PDF search and SQL queries.",
    backstory=(
        "You are a data scientist specializing in extracting insights from research papers and structured data. "
        "You use semantic search over PDFs for research findings and SQL for exact numbers. "
        "You always cite your sources precisely."
    ),
    tools=[LlamaIndexPDFTool(), LlamaIndexSQLTool()],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=4,
)

synthesizer = Agent(
    role="Research Synthesizer",
    goal="Combine web research and data analysis into a coherent synthesis that highlights connections and insights.",
    backstory=(
        "You are a senior AI researcher who excels at combining information from multiple sources — "
        "live web data, academic papers, and structured datasets — into clear, actionable insights."
    ),
    tools=[],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=2,
)

writer = Agent(
    role="Technical Report Writer",
    goal="Write a polished, well-structured technical report from the synthesized research.",
    backstory=(
        "You are a technical writer with expertise in AI topics. "
        "You write clear, professional reports that are accurate, well-cited, and follow a consistent structure."
    ),
    tools=[],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=2,
)

manager = Agent(
    role="AI Pipeline Manager",
    goal="Coordinate the research pipeline end-to-end and deliver a final report.",
    backstory=(
        "You manage a 4-agent AI research pipeline. Follow this exact order: "
        "Step 1: Delegate web research to the Web Research Specialist. "
        "Step 2: Delegate PDF and data analysis to the AI Data & Research Analyst. "
        "Step 3: Delegate synthesis to the Research Synthesizer. "
        "Step 4: Delegate final writing to the Technical Report Writer. "
        "Never skip steps. Validate each output before proceeding."
    ),
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=True,
)

TOPIC = "diffusion models and their applications in AI"

web_research_task = Task(
    description=(
        f"Search the live web for the latest developments on: '{TOPIC}'. "
        "Find at least 3 recent sources. Include URLs and key findings. Focus on the past year."
    ),
    expected_output="A web research brief with 3+ sources, URLs, key findings, and recent trends. Minimum 200 words.",
)

data_research_task = Task(
    description=(
        f"Research '{TOPIC}' using the local PDF knowledge base. "
        "Also query the billing data: which project has the highest GPU usage and what instance types are used? "
        "Use both the PDF Search and SQL Query tools."
    ),
    expected_output="A data research brief with PDF findings with page citations and SQL query results. Minimum 200 words.",
    context=[web_research_task],
)

synthesis_task = Task(
    description=(
        "Synthesize the web research and data research into a unified analysis.\n"
        "Cover: what the web says vs what the PDFs say, how the billing data relates to the topic, "
        "top 3 actionable insights for AI engineers, and open questions worth investigating."
    ),
    expected_output="A synthesis document covering all four areas above. Minimum 250 words.",
    context=[web_research_task, data_research_task],
)

writing_task = Task(
    description=(
        f"Write a complete technical report on '{TOPIC}' using all research above.\n\n"
        "Structure:\n"
        f"# {TOPIC.title()}\n"
        "## Executive Summary\n"
        "## Key Findings\n"
        "## Data Insights\n"
        "## Technical Deep Dive\n"
        "## Practical Implications\n"
        "## References\n\n"
        "Save to: ./output/capstone_report.md"
    ),
    expected_output="A complete Markdown report saved to ./output/capstone_report.md. Minimum 600 words, all sources cited.",
    context=[web_research_task, data_research_task, synthesis_task],
    output_file="./output/capstone_report.md",
)

os.makedirs("./output", exist_ok=True)

def task_callback(output):
    print("\n[rate limiter] Task complete — waiting 20s...\n")
    time.sleep(20)

crew = Crew(
    agents=[web_researcher, data_analyst, synthesizer, writer],
    tasks=[web_research_task, data_research_task, synthesis_task, writing_task],
    process=Process.hierarchical,
    manager_agent=manager,
    planning=False,
    planning_llm=llm,
    verbose=True,
    memory=False,
    max_rpm=5,
    task_callback=task_callback,
)

print("\n" + "═" * 60)
print("CAPSTONE — Multi-Framework AI Research Pipeline")
print(f"Topic: {TOPIC}")
print("Frameworks: LangChain + LlamaIndex + CrewAI")
print("═" * 60 + "\n")

result = None
for attempt in range(3):
    try:
        result = crew.kickoff()
        break
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            wait = 40 * (attempt + 1)
            print(f"\nRate limit hit — waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
        else:
            raise

if result:
    print("\n" + "═" * 60)
    print("Capstone complete. Report saved to ./output/capstone_report.md")
    print("═" * 60)
    print(result)
else:
    print("Max retries reached — wait a minute and run again.")
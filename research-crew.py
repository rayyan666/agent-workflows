import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import FileWriterTool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
    max_tokens=2048,
)

class TavilySearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information about any topic. "
        "Input: a search query string. Output: top results with URLs and content."
    )
    def _run(self, query: str) -> str:
        from langchain_community.tools.tavily_search import TavilySearchResults
        results = TavilySearchResults(max_results=4, api_key=os.environ["TAVILY_API_KEY"]).invoke(query)
        return "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])

class PDFSearchTool(BaseTool):
    name: str = "PDF Research Search"
    description: str = (
        "Search local PDF research papers for AI/ML concepts, formulas, methods, and findings. "
        "Input: a specific question or keyword. Output: relevant passages with page citations."
    )
    def _run(self, query: str) -> str:
        from llama_index.core import StorageContext, Settings, load_index_from_storage
        from llama_index.llms.groq import Groq
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"], temperature=0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage/pdf_vector"))
        engine = index.as_query_engine(similarity_top_k=3)
        response = engine.query(query)
        citations = [f"[{n.metadata.get('file_name','pdf')} p.{n.metadata.get('page_number','?')}]" for n in response.source_nodes]
        return f"{response}\n\nSources: {', '.join(citations)}"

researcher = Agent(
    role="AI Research Specialist",
    goal="Gather comprehensive, accurate information about AI/ML topics from academic papers and current web sources.",
    backstory=(
        "You are a PhD-level AI researcher with 10 years of experience synthesizing machine learning papers. "
        "You are meticulous about accuracy and always cross-reference multiple sources. "
        "You prefer primary sources over web articles when available."
    ),
    tools=[PDFSearchTool(), TavilySearchTool()],
    llm=llm,
    verbose=True,
    memory=False,
    max_iter=5,
)

analyst = Agent(
    role="AI Research Analyst",
    goal="Critically analyze research findings, identify key insights, compare approaches, and highlight practical implications.",
    backstory=(
        "You are a senior AI analyst at a top tech consultancy. "
        "You excel at breaking down complex research into clear insights, spotting trends across papers, "
        "and explaining what findings mean for real-world applications."
    ),
    tools=[],
    llm=llm,
    verbose=True,
    memory=False,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Transform research analysis into a polished, well-structured technical report that is accurate and well-cited.",
    backstory=(
        "You are a technical writer who has authored dozens of AI research summaries for academic and industry audiences. "
        "Your reports follow a consistent structure: Executive Summary, Key Findings, Technical Deep Dive, "
        "Practical Implications, and References."
    ),
    tools=[FileWriterTool()],
    llm=llm,
    verbose=True,
    memory=False,
)

TOPIC = "diffusion models and their applications in image generation"

research_task = Task(
    description=(
        f"Research the topic: '{TOPIC}'.\n"
        "Use PDF Research Search to find what local papers say — include page citations.\n"
        "Use Web Search to find the latest developments and real-world applications from the past year.\n"
        "Compile all findings into a structured research brief."
    ),
    expected_output=(
        "A research brief with: PDF findings with page citations, "
        "web findings with URLs, key technical concepts, recent developments. Minimum 400 words."
    ),
    agent=researcher,
)

analysis_task = Task(
    description=(
        f"Analyze the research brief about '{TOPIC}'.\n"
        "Identify the 3-5 most important insights.\n"
        "Compare different approaches or methods mentioned.\n"
        "Highlight what has changed recently vs what the papers describe.\n"
        "Identify gaps or open questions.\n"
        "Explain practical implications for AI engineers."
    ),
    expected_output=(
        "Structured analysis with: top insights, approach comparison, "
        "recent vs established findings, open questions, engineering implications. Minimum 300 words."
    ),
    agent=analyst,
    context=[research_task],
)

writing_task = Task(
    description=(
        f"Write a polished technical report about '{TOPIC}' using the research brief and analysis.\n\n"
        "Structure:\n"
        "# [Topic Title]\n"
        "## Executive Summary\n"
        "## Key Findings\n"
        "## Technical Deep Dive\n"
        "## Recent Developments\n"
        "## Practical Implications\n"
        "## References\n\n"
        "Save the report to './output/report.md' using the File Writer tool."
    ),
    expected_output="A complete Markdown report saved to ./output/report.md. Minimum 600 words, properly cited.",
    agent=writer,
    context=[research_task, analysis_task],
    output_file="./output/report.md",
)

os.makedirs("./output", exist_ok=True)

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,
    memory=False,
    max_rpm=25,
)

print("\n" + "═" * 60)
print(f"CrewAI Research Crew — Topic: {TOPIC}")
print("Agents: Researcher → Analyst → Writer")
print("═" * 60 + "\n")

result = crew.kickoff()
print("\n" + "═" * 60)
print("Crew finished. Report saved to ./output/report.md")
print("═" * 60)
print(result)
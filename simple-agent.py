import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = TavilySearchResults(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def summarize(text: str) -> str:
    """Summarizes a long block of text into 3 concise bullet points."""
    response = llm.invoke(
        f"Summarize the following into exactly 3 bullet points:\n\n{text}"
    )
    return response.content

tools = [search_tool, summarize]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful research assistant. Search for current info, then summarize."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What are the latest developments in LangChain in 2025?"}]
})

print("\n── Final Answer ──")
print(result["messages"][-1].content)
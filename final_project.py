from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from typing_extensions import Annotated
import operator
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun

# --- new: Google + Scholar deps ---
from googleapiclient.discovery import build
from serpapi import GoogleSearch

# ---------- State ----------
class S(TypedDict):
    messages: Annotated[List[Any], operator.add]
    query: str
    region: str
    discipline: Optional[str]
    timeframe: Optional[str]
    findings: Annotated[List[Dict[str, Any]], operator.add]
    syntheses: Annotated[List[str], operator.add]
    policy_gaps: Annotated[List[Dict[str, Any]], operator.add]
    topic_ideas: Annotated[List[Dict[str, Any]], operator.add]
    recommendations: Annotated[List[Dict[str, Any]], operator.add]
    result: Optional[Dict[str, Any]]

# ---------- LLM & Tools ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

tools = [
    TavilySearchResults(max_results=8),  # needs TAVILY_API_KEY
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)),
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=4000)),
]
react = create_react_agent(llm, tools)

# ---------- Prompts ----------
SYN = ChatPromptTemplate.from_messages([
    ("system", "Synthesize crisp, region-aware insights for {region} (discipline: {discipline}, timeframe: {timeframe})."),
    ("human", "Summarize these findings into <=10 bullets with inline source tags:\n{findings}"),
])
GAPS = ChatPromptTemplate.from_messages([
    ("system", "Extract concrete policy gaps (name, description, who_is_affected, evidence, urgency, feasibility, levers)."),
    ("human", "Region: {region}\nDiscipline: {discipline}\nBullets:\n{bullets}")
])
TOPICS = ChatPromptTemplate.from_messages([
    ("system", "Propose 8–12 policy-relevant academic topics (method, data, novelty, impact, quick wins, refs)."),
    ("human", "Region: {region}\nDiscipline: {discipline}\nTimeframe: {timeframe}\nGaps:\n{gaps}")
])
RECS = ChatPromptTemplate.from_messages([
    ("system", "Translate topics into policy recommendations with steps, stakeholders, and 12-month metrics."),
    ("human", "Region: {region}\nTopics:\n{topics}")
])
RANK = ChatPromptTemplate.from_messages([
    ("system", "Score topics on Relevance(0.35), Recency(0.25), Need(0.40). Return sorted JSON with WeightedScore."),
    ("human", "Topics:\n{topics}\nGaps:\n{gaps}")
])

# ---------- Helpers: Google & Scholar ----------
def google_search(query: str, num: int = 5) -> List[Dict[str, str]]:
    """
    Google Custom Search (Programmable Search Engine).
    Requires env: GOOGLE_API_KEY, GOOGLE_CSE_ID
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return []  # silently skip if not configured

    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    items = res.get("items", []) or []
    out = []
    for i in items:
        out.append({
            "title": i.get("title", ""),
            "snippet": i.get("snippet", ""),
            "link": i.get("link", ""),
        })
    return out

def scholar_search(query: str, num: int = 5) -> List[Dict[str, str]]:
    """
    Google Scholar via SerpAPI.
    Requires env: SERPAPI_API_KEY
    """
    serp_key = os.getenv("SERPAPI_API_KEY")
    if not serp_key:
        return []  # silently skip if not configured

    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serp_key,
        "num": num
    }
    results = GoogleSearch(params).get_dict()
    org = results.get("organic_results", []) or []
    out = []
    for r in org[:num]:
        out.append({
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "link": r.get("link", ""),
        })
    return out

# ---------- Nodes ----------
def n_search(state: S)->S:
    """
    Hybrid retriever:
    - ReAct agent with Tavily + Wikipedia + arXiv (your original setup)
    - PLUS Google Custom Search
    - PLUS Google Scholar via SerpAPI
    Merges all into a single 'findings' list.
    """
    # 1) ReAct agent (Tavily/Wiki/ArXiv)
    prompt = (f"Find up to 12 recent items about '{state['query']}' in {state['region']} "
              f"({state.get('timeframe') or 'last 24 months'}). Include 1–2 sentence gist, date, and link.")
    react_out = react.invoke({
        "messages": [
            SystemMessage(content=(
                "Prefer official/government, multilaterals, think tanks, and peer-reviewed venues. "
                "Include titles, dates, and URLs."
            )),
            HumanMessage(content=prompt)
        ]
    })
    findings_text = react_out["messages"][-1].content
    findings: List[Dict[str, Any]] = [{"source": "ReAct(Tavily/Wiki/ArXiv)", "summary": findings_text}]

    # 2) Google Custom Search (if configured)
    query_google = f"{state['query']} {state['region']} policy regulation site:.gov OR site:.org OR site:.edu"
    g_results = google_search(query_google, num=6)
    for g in g_results:
        findings.append({
            "source": "Google",
            "summary": f"{g['title']} — {g['snippet']} ({g['link']})"
        })

    # 3) Google Scholar via SerpAPI (if configured)
    query_scholar = f"{state['query']} {state['region']} policy"
    s_results = scholar_search(query_scholar, num=6)
    for s in s_results:
        findings.append({
            "source": "Scholar",
            "summary": f"{s['title']} — {s['snippet']} ({s['link']})"
        })

    return {"findings": findings}

def n_synthesize(state:S)->S:
    chain = SYN | llm | StrOutputParser()
    merged = "\n".join(f"- [{x.get('source','web')}] {x['summary']}" for x in state["findings"])
    return {"syntheses":[chain.invoke({
        "findings": merged[:8000],
        "region": state["region"],
        "discipline": state.get("discipline") or "(any)",
        "timeframe": state.get("timeframe") or "last 24 months",
    })]}

def n_gaps(state:S)->S:
    chain = GAPS | llm | StrOutputParser()
    out = chain.invoke({
        "region": state["region"],
        "discipline": state.get("discipline") or "(any)",
        "bullets": "\n".join(state["syntheses"])[:8000]
    })
    return {"policy_gaps":[{"raw": out}]}

def n_topics(state:S)->S:
    chain = TOPICS | llm | StrOutputParser()
    gaps_raw = (state["policy_gaps"][0]["raw"] if state["policy_gaps"] else "[]")[:8000]
    out = chain.invoke({
        "region": state["region"],
        "discipline": state.get("discipline") or "(any)",
        "timeframe": state.get("timeframe") or "last 24 months",
        "gaps": gaps_raw
    })
    return {"topic_ideas":[{"raw": out}]}

def n_recs(state:S)->S:
    chain = RECS | llm | StrOutputParser()
    topics_raw = (state["topic_ideas"][0]["raw"] if state["topic_ideas"] else "[]")[:8000]
    out = chain.invoke({
        "region": state["region"],
        "topics": topics_raw
    })
    return {"recommendations":[{"raw": out}]}

def n_rank(state:S)->S:
    chain = RANK | llm | StrOutputParser()
    topics_raw = (state["topic_ideas"][0]["raw"] if state["topic_ideas"] else "[]")[:8000]
    gaps_raw = (state["policy_gaps"][0]["raw"] if state["policy_gaps"] else "[]")[:8000]
    ranked = chain.invoke({
        "topics": topics_raw,
        "gaps": gaps_raw
    })
    recs_raw = (state["recommendations"][0]["raw"] if state["recommendations"] else "")
    return {"result":{
        "query": state["query"],
        "region": state["region"],
        "ranked_topics": ranked,
        "recommendations": recs_raw,
        "synthesis": "\n".join(state["syntheses"]),
    }}

# ---------- Build & Run ----------
def build_graph():
    g = StateGraph(S)
    g.add_node("search", n_search)
    g.add_node("synthesize", n_synthesize)
    g.add_node("gaps", n_gaps)
    g.add_node("topics", n_topics)
    g.add_node("recs", n_recs)
    g.add_node("rank", n_rank)

    g.set_entry_point("search")
    g.add_edge("search","synthesize")
    g.add_edge("synthesize","gaps")
    g.add_edge("gaps","topics")
    g.add_edge("topics","recs")
    g.add_edge("recs","rank")
    g.add_edge("rank", END)

    return g.compile(checkpointer=MemorySaver())

def run(query:str, region:str, discipline:Optional[str]=None, timeframe:str="last 24 months"):
    app = build_graph()
    init = {
        "messages":[HumanMessage(content=f"User wants topics for '{query}' in {region}.")],
        "query": query, "region": region, "discipline": discipline, "timeframe": timeframe,
        "findings": [], "syntheses": [], "policy_gaps": [], "topic_ideas": [], "recommendations": [], "result": None
    }
    out = app.invoke(init)
    return out["result"]

if __name__ == "__main__":
    # Optional: sanity check for keys (won't block if Google/SerpAPI are missing)
    missing = [v for v in ("OPENAI_API_KEY","TAVILY_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"Warning: missing env vars {missing}. ReAct(Tavily/Wiki/ArXiv) may fail.")

    print(run("AI for health data governance", "Pakistan", "public policy", "last 18 months"))

# AI Agents in LangGraph

This repository contains a small collection of notebooks and a final Python script that explore **agentic workflows with [LangGraph](https://github.com/langchain-ai/langgraph)**, LangChain, OpenAI, Google Gemini, and web search tools (Tavily, Wikipedia, Arxiv, Google, DuckDuckGo).

The goal is educational: each notebook focuses on one core idea (ReAct agents, LangGraph components, persistence, human-in-the-loop, multi-step essay writer), and `final_project.py` ties everything together into a more realistic research agent.

---

## Contents

- `1.Simple_react_agent_scratch.ipynb`  
  Build a **from-scratch ReAct-style agent** on top of Google Gemini, with a tiny toolset (calculator and dog-weight lookup) and manual action parsing.

- `2.LangGraph_Components.ipynb`  
  Introduces **LangGraph building blocks**:
  - `StateGraph`, typed state
  - nodes, edges, conditional edges
  - a Gemini-based agent that can decide when to call a search tool.

- `3.Agentic _Search .ipynb`  
  Demonstrates **agentic search** using:
  - Tavily API for high-level answers
  - DuckDuckGo search as a fallback
  - basic web scraping with `requests` + `BeautifulSoup`
  - JSON pretty-printing for structured data.

- `4.Persistence_and_Streaming.ipynb`  
  Shows **persistence and streaming with LangGraph**:
  - `SqliteSaver` and `AsyncSqliteSaver` checkpointers
  - keeping conversation state across runs
  - streaming token-level output from a Gemini-backed graph.

- `5.Human_in_the_loop.ipynb`  
  Explores **manual approval and time travel**:
  - interrupting a graph before tool execution
  - inspecting and editing state
  - resuming or branching from past states using `get_state_history`.

- `6.Essay_Writer.ipynb`  
  A **multi-node essay-writing agent** that:
  - plans an outline
  - runs web research via Tavily
  - drafts, critiques, and revises essays in a loop
  - (optionally) exposes a small UI via a `helper.py` module (if present).

- `final_project.py`  
  A **policy‑research topic explorer** implemented as a LangGraph pipeline:
  - runs a ReAct search agent (Tavily + Wikipedia + Arxiv)
  - optionally augments results with Google Custom Search and Google Scholar (via SerpAPI)
  - synthesizes findings
  - extracts policy gaps
  - proposes academic topics
  - generates policy recommendations and ranking.

---

## Installation

### 1. Python environment

Use Python **3.10+** and a virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
```

### 2. Install dependencies

Install the core libraries used across the notebooks and `final_project.py`:

```bash
pip install \
  "langchain>=0.3.0" \
  langchain-openai \
  langchain-community \
  langchain-core \
  langchain-google-genai \
  langgraph \
  tavily-python \
  duckduckgo-search \
  google-generativeai \
  google-api-python-client \
  google-search-results \
  beautifulsoup4 \
  requests \
  python-dotenv \
  pygments
```

You may not need every package for every notebook, but this set should cover all examples in the repo.

---

## Environment variables

All secrets are loaded from environment variables (typically via a `.env` file and `python-dotenv`).

Create a **`.env` (not committed to Git)** with keys such as:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
GOOGLE_CSE_ID=...
SERPAPI_API_KEY=...
```

Where they are used:

- `OPENAI_API_KEY`  
  Used by `ChatOpenAI` in `final_project.py`, `5.Human_in_the_loop.ipynb`, and `6.Essay_Writer.ipynb`.

- `GOOGLE_API_KEY`  
  Used by Google Gemini clients (`google.generativeai`, `ChatGoogleGenerativeAI`) and for Google Programmable Search in `final_project.py`.

- `TAVILY_API_KEY`  
  Used by Tavily tools (`TavilySearchResults` and `TavilyClient`) in several notebooks and in `final_project.py`.

- `GOOGLE_CSE_ID`  
  Custom Search Engine ID for the Google Custom Search integration in `final_project.py`.

- `SERPAPI_API_KEY`  
  Used to access Google Scholar via SerpAPI in `final_project.py`.

> **Security tip**: keep your real keys only in `.env` or your shell environment.  
> Commit a `.env.example` file instead of a real `.env`, and add `.env` to `.gitignore`.

---

## Running the final project

The main entry point is `final_project.py`, which builds and runs a LangGraph pipeline.

### 1. Configure keys

Make sure at least these variables are set in your environment before running:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Optional but recommended for richer search:

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`
- `SERPAPI_API_KEY`

### 2. Run from the command line

From the project root:

```bash
python final_project.py
```

This will:

- construct the LangGraph
- run a sample query (currently hard-coded in `__main__`)
- print a dictionary containing:
  - the original query and region
  - ranked topics
  - synthesized summary text
  - policy recommendations.

### 3. Use as a module

You can also import and call the graph from your own Python code:

```python
from final_project import run

result = run(
    query="AI for health data governance",
    region="Pakistan",
    discipline="public policy",
    timeframe="last 18 months",
)

print(result["ranked_topics"])
```

---

## Working with the notebooks

1. Activate your virtual environment and install dependencies.
2. Launch Jupyter or VS Code notebooks, for example:

   ```bash
   pip install jupyter
   jupyter lab
   ```

3. Open any of the `*.ipynb` files and run the cells top‑to‑bottom.

Notes:

- Many notebooks rely on `GOOGLE_API_KEY` and/or `TAVILY_API_KEY` being set.
- `4.Persistence_and_Streaming.ipynb` and `5.Human_in_the_loop.ipynb` use `SqliteSaver` checkpointers; they can run fully in-memory, no database setup required.
- `6.Essay_Writer.ipynb` expects a `helper.py` module that provides `ewriter` and `writer_gui` utilities for the GUI demo. If you do not have `helper.py`, you can still run the core essay-writing graph cells up to the GUI section.

---

## Project ideas & extensions

Some natural next steps you can try on top of this codebase:

- Add your own tools (SQL, internal APIs, document search) to the LangGraph agents.
- Swap between OpenAI and Gemini models in the same graphs.
- Persist conversation state to a real SQLite file instead of `:memory:`.
- Deploy `final_project.py` behind a simple FastAPI or Gradio UI.

Contributions, issues, and suggestions are welcome.

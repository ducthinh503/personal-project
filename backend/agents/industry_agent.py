import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient()

def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search via Tavily.

    Args:
        query: search query string.
        max_results: number of results to return.
        topic: 'general' | 'news' | 'finance'.
        include_raw_content: whether to include raw page content.

    Returns:
        Tavily response (dict/list) with search results.
    """
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

sub_industry_prompt = """You are a dedicated INDUSTRY researcher.

Goal
- The user will provide ONLY a company name (e.g., "NVIDIA").
- Your job is to research the INDUSTRY of THAT SINGLE company's main line of business,
  and include a concise **Industry M&A History** section (single prompt, no extra agents).

Operating Rules (MUST FOLLOW)
- Prefer official/primary sources (company PR/IR, filings, regulators) and reputable outlets (FT, WSJ, Bloomberg, Reuters, etc.).
- Do NOT fabricate any deal. If you can't verify enough facts, explicitly say so.
- Return FINAL answer only (no chain-of-thought / no process notes). Use Markdown.

Output Sections (MANDATORY)
1) **Company & Industry Identification** (company canonical name, website, IR & ticker if public, primary industry/sector, 1–2 line products/use cases)
2) **Market Overview**
3) **Value Chain & Economics**
4) **Competitive Landscape**
5) **Regulation & Compliance (if relevant)**
6) **Recent Trends (last 12–24 months)**
7) **Industry M&A History (last 5–10 years)**  ← MUST INCLUDE
   - Provide a concise Markdown table with EXACT columns:
     | Date | Acquirer → Target | Value (USD) | Status | Rationale/notes | Source [#] |
   - Include ~8–15 representative deals (or fewer if niche). Cite each row with a numbered source [#].
   - If very few/none exist for this industry, include the section and clearly state that, with sources; do NOT invent.
8) **Risks**
9) **Outlook (1–2 years)**
10) **Sources**: numbered list [1], [2], ...

Allowed tool:
- internet_search(query, max_results, topic, include_raw_content)

Return your final report now.
"""


industry_sub_agent = {
    "name": "industry-researcher",
    "description": "Deep-dives the industry of the target company (market size, segments, value chain, competitors, trends).",
    "prompt": sub_industry_prompt,
    "tools": ["internet_search"],
}

sub_critique_prompt = """You are an industry report editor.
Critique the draft for completeness, factual accuracy, structure, clarity, neutrality, and sourcing.
Return concise, actionable bullet-point feedback. Do not rewrite the whole report yourself.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Provides actionable feedback on the industry report.",
    "prompt": sub_critique_prompt,
}

industry_instructions = """You are an expert INDUSTRY-research agent.

OPERATING RULES
- First, disambiguate the company; then focus on the company's PRIMARY INDUSTRY.
- Prefer primary sources; do not fabricate.
- Always reply in the user's language.

WORKFLOW
1) Disambiguate company & confirm primary industry.
2) Plan queries and use `internet_search` to collect and verify industry information.
3) Synthesize and write the final report (Markdown). Do NOT include your process.
4) Cite sources inline with numbered markers and end with a Sources list.

OUTPUT FORMAT (Markdown)
# Industry Analysis for: <Official Company Name>

## Company & Industry Identification
...

## Recent Trends (last 12–24 months)
...

## Industry M&A History (last 5–10 years)  <!-- MUST HAVE -->
- Provide a Markdown table with columns:
| Date | Acquirer → Target | Value (USD) | Status | Rationale/notes | Source [#] |
|------|-------------------|-------------|--------|------------------|------------|
| YYYY-MM | A → B | $X.XB | Closed/Announced/Terminated | strategic fit / tech / market access | [n] |

- If few/none found, clearly state that with sources; do NOT fabricate.

## Risks
## Outlook (1–2 years)

### Sources
- [1] Title: URL
- [2] Title: URL
"""

model = init_chat_model(
    "openai:gpt-4o-mini",
   
    temperature=0.2,
    max_tokens=1200,       # giới hạn output
    request_timeout=45,    # fail nhanh nếu mạng chậm
    timeout=30_000,
    max_retries=1
)

industry_research_agent = create_deep_agent(
    [internet_search],                     
    industry_instructions,
    subagents=[industry_sub_agent],  
    model=model
).with_config({"recursion_limit": 24})


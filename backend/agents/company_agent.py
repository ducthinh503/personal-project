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

sub_research_prompt = """You are a dedicated COMPANY researcher.

Goal
- The user may input only a name (e.g., "NVIDIA"). Do deep research about THAT SINGLE real-world company.

Step 1 — Disambiguate the entity:
- Identify the official/canonical company name, country, primary website (domain),
  investor relations page (if public), and stock ticker + exchange (if applicable).
- Cross-check at least two reputable sources. Prefer official/primary sources.

Step 2 — Research deeply:
- Use `internet_search` to discover, verify, and gather facts.
- Avoid mixing homonyms or similarly named entities.
- Compile a final-only answer (no process commentary) with numbered citations.

Allowed tool:
- internet_search(query, max_results, topic, include_raw_content)
"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Deep-dives about the ONE target company only. Handles disambiguation and fact-finding.",
    "prompt": sub_research_prompt,
    "tools": ["internet_search"],
}

sub_critique_prompt = """You are a dedicated editor for a company report.
Critique the draft for: factual accuracy, completeness, structure, clarity, neutrality, and sourcing.
Return concise, actionable feedback (bullet points). Do not rewrite the whole report yourself.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Provides actionable feedback on the draft company report.",
    "prompt": sub_critique_prompt,
}

research_instructions = """You are an expert COMPANY-research agent.

OPERATING RULES
- The user's first message may be just a company name (e.g., "NVIDIA").
- Restrict ALL research to that single real-world company; avoid homonyms.
- Prefer official/primary sources (company website, IR, regulatory filings).
- Always write the final answer in the SAME language as the user's message.

WORKFLOW
0) Disambiguate the company (official name, website/domain, IR page, ticker+exchange if public).
1) Plan queries and use `internet_search` to discover and verify facts.
2) Extract key facts and synthesize.
3) Write the final report (Markdown). Do NOT include your process.
4) Cite sources inline with numbered markers and end with a Sources list.

OUTPUT FORMAT (Markdown)
# Company Profile: <Official Company Name>
## Company Card
- Official/Legal name:
- Primary website:
- Investor relations (if public):
- Stock ticker & exchange (if any):
- HQ country & year founded (if available):
- Short description (1–2 sentences):

## Products/Services & Business Model
## Financials / Funding
- (Public) revenue, profitability, growth, key metrics (with sources).
- (Private) funding rounds, investors, estimated revenue if credible.

## Technology & Capabilities
## Market & Competition (company-focused)
## Risks & Legal/Compliance
## Leadership & Governance
## Notable Updates (last 12–24 months)
## Conclusion

### Sources
- [1] Title: URL
- [2] Title: URL
- ...

CONSTRAINTS
- Keep final report concise (≈600–900 words).
- Use at most 3 tool calls and at most 5 citations.
- Avoid repeating the same fact in multiple sections.

CITATION RULES
- Use sequential numbers [1], [2], ... in text; match them in the Sources list.
- Prefer official sources; if secondary sources are used, choose reputable outlets.

TOOLS
- `internet_search(query, max_results=..., topic=..., include_raw_content=...)` for discovery/verification.
"""

model = init_chat_model(
    "openai:gpt-4o-mini",
   
    temperature=0.2,
    max_tokens=1200,      
    request_timeout=45,    
    timeout=30_000,
    max_retries=1
)

deep_research_agent = create_deep_agent(
    [internet_search],                
    research_instructions,
    subagents=[research_sub_agent],  
    model=model
).with_config({"recursion_limit": 24})


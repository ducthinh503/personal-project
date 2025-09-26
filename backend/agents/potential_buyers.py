import os
import textwrap
from typing import Dict, Any, Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Tavily (tùy chọn)
try:
    from tavily import TavilyClient  # type: ignore
    _TAVILY_OK = True
except Exception:
    TavilyClient = None  # type: ignore
    _TAVILY_OK = False

load_dotenv()


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1)


def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    
    """Plain search helper; nếu thiếu Tavily key thì trả rỗng để không vỡ pipeline."""
    if not _TAVILY_OK or not os.getenv("TAVILY_API_KEY"):
        return {"results": []}
    
    from tavily import TavilyClient
    tavily = TavilyClient()
    try:
        return tavily.search(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as e:
        # degrade gracefully
        return {"results": [], "error": f"{type(e).__name__}: {e}"}

# ---------- Swarm các vi mô-agent ----------
def _gather_context(company: str) -> Dict[str, Any]:
    docs = internet_search(f"{company} competitors partners acquisitions strategy", max_results=5, topic="general")
    empty = not docs or not docs.get("results")
    return {"sources": docs, "no_sources": empty}


def _strategy_fit(company: str, sources: Dict[str, Any]) -> str:
    llm = _llm()
    prompt = textwrap.dedent(f"""
    ROLE: StrategyFit agent.
    Company: {company}
    Context sources (truncated JSON-ish):
    {str(sources)[:5000]}

    Task: Propose strategic acquirer profiles (3–6) that would gain product/customer/geographic synergies if acquiring {company}.
    Output bullet list: Buyer Name (or Archetype) — Why it fits (1–2 lines).
    """)
    return llm.invoke(prompt).content


def _capability_match(company: str, sources: Dict[str, Any]) -> str:
    llm = _llm()
    prompt = textwrap.dedent(f"""
    ROLE: CapabilityMatch agent.
    Company: {company}
    Context sources (truncated):
    {str(sources)[:5000]}

    Task: Suggest PE/financial buyers and adjacent-tech strategics who could scale {company}'s capabilities.
    Output bullet list with brief capability rationale & potential value-creation levers.
    """)
    return llm.invoke(prompt).content


def _deal_precedent(company: str, sources: Dict[str, Any]) -> str:
    llm = _llm()
    prompt = textwrap.dedent(f"""
    ROLE: DealPrecedent agent.
    Company: {company}
    Context (truncated):
    {str(sources)[:5000]}

    Task: List 3–5 recent M&A precedents in this industry (last ~3y), each with buyer—target—rationale.
    If uncertain, provide plausible archetypes + reasoning.
    """)
    return llm.invoke(prompt).content

def _aggregate(company, fit, cap, deals, feedback, *, no_sources=False, sources=None) -> str:
    llm = _llm()
    fb_txt = f"\nReviewer feedback to incorporate:\n{feedback}\n" if feedback else ""

    # bật/tắt chế độ yêu cầu nguồn
    require_sources = os.getenv("PB_REQUIRE_SOURCES", "true").lower() == "true"

    # (giữ phần build sources_snip nếu có)
    source_lines = []
    if isinstance(sources, dict):
        try:
            for i, item in enumerate(sources.get("results", [])[:8], 1):
                title = (item.get("title") or item.get("url") or "")[:120]
                url = item.get("url") or ""
                source_lines.append(f"[{i}] {title}: {url}")
        except Exception:
            pass
    sources_snip = "\n".join(source_lines)

    mode = "NO-SOURCE" if no_sources else "WITH-SOURCES"

    # nếu không có nguồn và đang bật strict mode -> in NOTE
    if no_sources and require_sources:
        return (
            "No external sources available (quota/disabled). "
            "Skipping market-validated buyers. Refer to the model-driven Buyer List section.\n\n"
            "Without access to external sources, I cannot provide a validated list."
        )

    # còn lại: sinh danh sách như bình thường (có/không có sources đều cho phép)
    prompt = f"""
You are the aggregator.
Mode: {mode}

Merge the three drafts below into a single ranked list of 6–10 potential buyers for {company}:
1) StrategyFit:
{fit}

2) CapabilityMatch:
{cap}

3) DealPrecedent:
{deals}
{fb_txt}

Requirements:
- Group by Strategic vs Financial (PE).
- Each item: Buyer name (or archetype), 1–2 line rationale, and a Fit Score 0–100.
- Attach citation markers [n] when derived from SOURCES below (if any).
- End with 'Assumptions & Caveats'.
Return FINAL MARKDOWN list only.

SOURCES
{sources_snip}
"""
    return llm.invoke(prompt).content


# ---------- Public API (được main.py gọi) ----------
def run_potential_buyers_swarm(company: str, feedback: Optional[str] = None) -> str:
    company = (company or "").strip() or "Unknown Company"

    ctx = _gather_context(company)
    fit = _strategy_fit(company, ctx)
    cap = _capability_match(company, ctx)
    deals = _deal_precedent(company, ctx)

    md = _aggregate(
        company,
        fit,
        cap,
        deals,
        feedback,
        no_sources=bool(ctx.get("no_sources")),
        sources=ctx.get("sources"),
    )
    return f"# Potential Buyers \n\n{md}"

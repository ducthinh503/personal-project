# agents/swarm_financial_model.py
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


# ---------- Helpers ----------
def _llm() -> ChatOpenAI:
    # có thể đổi OPENAI_MODEL trong .env nếu muốn
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """
    Web search helper. Nếu chưa cài/kết nối Tavily, trả về dict rỗng để pipeline vẫn chạy.
    """
    if not _TAVILY_OK or not os.getenv("TAVILY_API_KEY"):
        return {"results": []}
    tavily = TavilyClient()
    return tavily.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ---------- Micro-agents ----------
def _analyst_fetch(company: str) -> Dict[str, Any]:
    """Thu thập mẩu thông tin nền (company profile/IR/news)."""
    q = f"{company} revenue growth segments data center gaming IR site"
    docs = internet_search(q, max_results=5, topic="finance", include_raw_content=False)
    return {"sources": docs}


def _assumption_builder(company: str, sources: Dict[str, Any], feedback: Optional[str]) -> Dict[str, Any]:
    """
    Suy diễn Assumptions base/bull/bear 3 năm.
    Có thể dùng feedback (nếu QC yêu cầu sửa).
    """
    llm = _llm()
    fb_txt = f"\nReviewer feedback to incorporate:\n{feedback}\n" if feedback else ""
    prompt = textwrap.dedent(f"""
    You are a financial assumptions builder.
    Company: {company}
    You have noisy web snippets (structured JSON-ish) from a finance search:
    === SOURCES (truncated) ===
    {str(sources)[:6000]}
    {fb_txt}

    TASK:
    - Infer approximate base-year revenue (if unknown, state "unknown" but keep modeling).
    - Propose 3-year CAGR for Base/Bull/Bear, and EBIT margin range per scenario.
    - Output JSON ONLY with:
      {{
        "base_year_revenue": "<USD or 'unknown'>",
        "scenarios": {{
          "base": {{"cagr": <float>, "ebit_margin": <float>}},
          "bull": {{"cagr": <float>, "ebit_margin": <float>}},
          "bear": {{"cagr": <float>, "ebit_margin": <float>}}
        }},
        "notes": ["short bullet", ...]
      }}
    }}
    """)
    return {"assumptions_json": llm.invoke(prompt).content}


def _modeler(company: str, assumptions_json: str) -> str:
    """Dựng bảng dự phóng 3 năm (Base/Bull/Bear) ở Markdown."""
    llm = _llm()
    prompt = textwrap.dedent(f"""
    You are a financial modeler.
    Company: {company}
    ASSUMPTIONS (JSON):
    {assumptions_json}

    Build a compact 3-year projection table in Markdown:
    - Columns: Year0 (base), Year1, Year2, Year3
    - For each scenario (Base/Bull/Bear): Revenue, EBIT, EBIT Margin
    - If base-year revenue unknown, create a symbolic placeholder (e.g., "~USD X") and proceed.

    After the table, add a short paragraph explaining drivers.
    Return FINAL MARKDOWN (no JSON, no extra commentary).
    """)
    return llm.invoke(prompt).content


def _sanity_checker(company: str, model_md: str) -> str:
    """Kiểm tra hợp lý (nhẹ) và chỉnh wording nếu cần."""
    llm = _llm()
    prompt = textwrap.dedent(f"""
    You are a sanity checker & editor.
    Company: {company}
    MODEL MARKDOWN:
    {model_md}

    - Lightly check plausibility of growth & margins (qualitative).
    - Improve clarity only; do not change the numbers unless there is a clear arithmetic inconsistency.
    - Append a short 'Sanity notes' list at the end.

    Return FINAL MARKDOWN only.
    """)
    return llm.invoke(prompt).content


# ---------- Public API (được main.py gọi) ----------
def run_financial_swarm(company: str, feedback: Optional[str] = None) -> str:
    """
    Agent swarm (4 vi mô-agent): analyst_fetch → assumption_builder → modeler → sanity_checker.
    Tham số `feedback` là tùy chọn (từ QC), chèn vào khi build assumptions.
    Trả về Markdown để nhúng vào báo cáo.
    """
    company = (company or "").strip()
    if not company:
        company = "Unknown Company"

    blackboard: Dict[str, Any] = {}
    blackboard["fetch"] = _analyst_fetch(company)
    blackboard["assumptions"] = _assumption_builder(company, blackboard["fetch"], feedback)
    blackboard["model_md"] = _modeler(company, blackboard["assumptions"]["assumptions_json"])
    final_md = _sanity_checker(company, blackboard["model_md"])
    return f"# Financial Model (Swarm)\n\n{final_md}"

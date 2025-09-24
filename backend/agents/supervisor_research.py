# agents/supervisor_research.py
from typing import TypedDict, NotRequired, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage,  HumanMessage
from typing import List

# import 2 deep agents có sẵn
from agents.industry_research_company import industry_research_agent
from agents.deep_research_company import deep_research_agent  # đổi tên module/biến cho đúng repo của bạn

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ... trong State, thêm trường messages:
class State(TypedDict, total=False):
    input: NotRequired[str]
    company_query: NotRequired[str]
    company: NotRequired[str]
    industry: NotRequired[str]

    industry_notes: Annotated[list[str], operator.add]
    company_notes: Annotated[list[str], operator.add]
    report: NotRequired[str]

    # NEW: để FE nhận được chat assistant
    messages: Annotated[list[AnyMessage], operator.add]

def _as_text(out) -> str:
    """Chuẩn hoá output từ DeepAgents (str / dict / Message)."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for k in ("output", "content", "text"):
            if k in out and isinstance(out[k], str):
                return out[k]
    # langchain Message
    content = getattr(out, "content", None)
    if isinstance(content, str):
        return content
    return str(out)

def _run_deep(agent, query: str) -> str:
    out = agent.invoke({"messages": [("user", query)]})
    return _as_text(out)

def supervisor(state: State):
    raw = state.get("input") or state.get("company_query")
    if isinstance(raw, dict):
        raw = raw.get("input") or raw.get("company_query") or next(iter(raw.values()), None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Missing 'input' (or legacy 'company_query').")
    query = raw.strip()

    # QUAN TRỌNG: bơm human message để FE giữ được bubble của user
    return {
        "input": query,
        "messages": [HumanMessage(content=query)],
    }

def industry_node(state: State):
    q = state["input"]
    txt = _run_deep(industry_research_agent, q)
    return {"industry_notes": [txt]}

def company_node(state: State):
    q = state["input"]
    txt = _run_deep(deep_research_agent, q)
    return {"company_notes": [txt]}

def combine(state: State):
    industry_txt = "\n".join(state.get("industry_notes", []))
    company_txt = "\n".join(state.get("company_notes", []))

    prompt = f"""Write a polished investor memo using BOTH blocks below.

REQUIREMENTS (MUST HAVE):
- Clear headings.
- Include a dedicated section titled **Industry M&A History (last 5–10 years)**.
- If the industry block already includes a table, carry it over as-is.
- If the industry block lacks M&A deals, add the section with a short note:
  "No sufficiently verified M&A deals found for the last 5–10 years based on the sources." (Do NOT fabricate.)
- Keep citations that are present; do not invent new sources.

[INDUSTRY BLOCK]
{industry_txt}

[COMPANY BLOCK]
{company_txt}
"""

    msg = llm.invoke(prompt)
    return {"report": msg.content, "messages": [msg]}


def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("industry", industry_node)
    g.add_node("company", company_node)
    g.add_node("combine", combine)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "industry")
    g.add_edge("supervisor", "company")
    g.add_edge("industry", "combine")
    g.add_edge("company", "combine")
    g.add_edge("combine", END)
    return g.compile()

# export để runtime pick up
graph = build_graph()

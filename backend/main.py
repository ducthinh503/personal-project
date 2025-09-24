# main.py — single-file entry (gộp app.py)
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI

# === YOUR AGENTS / SWARMS ===
from agents.deep_research_company import deep_research_agent
from agents.industry_research_company import industry_research_agent
from agents.swarm_financial_model import run_financial_swarm   # run_financial_swarm(company: str, feedback: str = "")
from agents.swarm_potential_buyers import run_potential_buyers_swarm  # run_potential_buyers_swarm(company: str, feedback: str = "")

# ---------------- Config ----------------
QUALITY_THRESHOLD = 0.80
MAX_ROUNDS = 1  # demo cho nhanh
_QC_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------- State ----------------
class ChatState(TypedDict, total=False):
    # LangGraph Server có thể nhét input ban đầu ở key "input"
    input: Any

    # messages được cộng dồn (append), KHÔNG overwrite
    messages: Annotated[List[BaseMessage], add_messages]

    # input đã chuẩn hoá
    company_query: str
    round: int

    # outputs của các agent/swarms
    company_report: str
    industry_report: str
    financial_model: str
    buyers: str

    # QC JSON + feedback
    qc_json: Dict[str, Any]
    feedback_company: str
    feedback_industry: str
    feedback_financial: str
    feedback_buyers: str


# ---------------- Helpers ----------------
def _str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _run_agent(agent, prompt: str) -> str:
    """Call a DeepAgents graph (prebuilt agent) with a plain user prompt."""
    out = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    try:
        # DeepAgents thường trả về {"messages": [ ... ]}
        return _str(out["messages"][-1].content)
    except Exception:
        return _str(out)


def _make_revision_prompt(base_query: str, feedback: str) -> str:
    if feedback:
        return (
            f"{base_query}\n\n"
            "Revise using this feedback. Output ONLY the final Markdown report, numbered citations required:\n"
            f"{feedback}"
        )
    return base_query


def _parse_json_loose(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"error": "qc invalid json", "raw": text}
        return {"error": "qc not json", "raw": text}


def qc_score(company_md: str, industry_md: str, financial_md: str, buyers_md: str) -> Dict[str, Any]:
    """
    QC đơn giản bằng LLM (chấm 0..1 + flag cần sửa + feedback bullet).
    Nếu lỗi LLM, trả về điểm cao để không loop mãi.
    """
    schema = """Return PURE JSON only:
{
  "company_score": float,
  "industry_score": float,
  "financial_score": float,
  "buyers_score": float,
  "needs_rework_company": bool,
  "needs_rework_industry": bool,
  "needs_rework_financial": bool,
  "needs_rework_buyers": bool,
  "feedback_company": [string],
  "feedback_industry": [string],
  "feedback_financial": [string],
  "feedback_buyers": [string]
}"""
    user = f"""Company MD:
<<<
{company_md}
>>>

Industry MD:
<<<
{industry_md}
>>>

Financial MD:
<<<
{financial_md}
>>>

Buyers MD:
<<<
{buyers_md}
>>>"""

    try:
        msg = _QC_LLM.invoke([
            {"role": "system", "content": "You are a strict reviewer. " + schema},
            {"role": "user", "content": user},
        ])
        text = msg.content if isinstance(msg.content, str) else _str(msg.content)
        data = _parse_json_loose(text)
        # defaults to avoid KeyErrors
        return {
            "company_score": float(data.get("company_score", 1.0)),
            "industry_score": float(data.get("industry_score", 1.0)),
            "financial_score": float(data.get("financial_score", 1.0)),
            "buyers_score": float(data.get("buyers_score", 1.0)),
            "needs_rework_company": bool(data.get("needs_rework_company", False)),
            "needs_rework_industry": bool(data.get("needs_rework_industry", False)),
            "needs_rework_financial": bool(data.get("needs_rework_financial", False)),
            "needs_rework_buyers": bool(data.get("needs_rework_buyers", False)),
            "feedback_company": data.get("feedback_company", []),
            "feedback_industry": data.get("feedback_industry", []),
            "feedback_financial": data.get("feedback_financial", []),
            "feedback_buyers": data.get("feedback_buyers", []),
        }
    except Exception as e:
        return {
            "error": f"qc_error: {e}",
            "company_score": 1.0,
            "industry_score": 1.0,
            "financial_score": 1.0,
            "buyers_score": 1.0,
            "needs_rework_company": False,
            "needs_rework_industry": False,
            "needs_rework_financial": False,
            "needs_rework_buyers": False,
            "feedback_company": [],
            "feedback_industry": [],
            "feedback_financial": [],
            "feedback_buyers": [],
        }


# ---------------- Nodes ----------------
def n_parse_input(state: ChatState) -> Dict[str, Any]:
    """
    Lấy company name từ:
      - state["input"] (string hoặc dict có key 'input' / 'company_query')
      - hoặc từ messages human cuối cùng (nếu FE gửi messages).
    Đồng thời đảm bảo có HumanMessage trong `messages` (không duplicate).
    """
    # 1) Ưu tiên input trực tiếp
    q: str = ""
    raw = state.get("input")

    if isinstance(raw, str):
        q = raw.strip()
    elif isinstance(raw, dict):
        # chấp nhận cả {input: "..."} hoặc {company_query: "..."} hoặc format khác
        for k in ("input", "company_query", "query", "message"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                q = v.strip()
                break

    # 2) Nếu vẫn trống, thử lấy từ messages human
    if not q:
        for m in reversed(state.get("messages", []) or []):
            if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip():
                q = m.content.strip()
                break

    if not q:
        q = "NVIDIA"  # fallback an toàn

    # Đảm bảo có HumanMessage (không thêm trùng)
    need_append = True
    for m in reversed(state.get("messages", []) or []):
        if isinstance(m, HumanMessage) and m.content == q:
            need_append = False
            break

    msg_list: List[AnyMessage] = [HumanMessage(content=q)] if need_append else []

    return {
        "company_query": q,
        "round": 0,
        "messages": msg_list,
    }


def n_company(state: ChatState) -> Dict[str, Any]:
    q = state["company_query"]
    fb = state.get("feedback_company", "")
    prompt = _make_revision_prompt(q, fb)
    return {"company_report": _run_agent(deep_research_agent, prompt)}


def n_industry(state: ChatState) -> Dict[str, Any]:
    q = state["company_query"]
    fb = state.get("feedback_industry", "")
    prompt = _make_revision_prompt(q, fb)
    return {"industry_report": _run_agent(industry_research_agent, prompt)}


def n_financial(state: ChatState) -> Dict[str, Any]:
    q = state["company_query"]
    fb = state.get("feedback_financial", "")
    # dùng named arg để tránh TypeError (đã từng gặp)
    return {"financial_model": run_financial_swarm(q, feedback=fb)}


def n_buyers(state: ChatState) -> Dict[str, Any]:
    q = state["company_query"]
    fb = state.get("feedback_buyers", "")
    return {"buyers": run_potential_buyers_swarm(q, feedback=fb)}


def n_qc(state: ChatState) -> Dict[str, Any]:
    return {
        "qc_json": qc_score(
            _str(state.get("company_report", "")),
            _str(state.get("industry_report", "")),
            _str(state.get("financial_model", "")),
            _str(state.get("buyers", "")),
        )
    }


def n_set_feedback(state: ChatState) -> Dict[str, Any]:
    qc = state.get("qc_json") or {}
    return {
        "round": (state.get("round") or 0) + 1,
        "feedback_company": _str(qc.get("feedback_company", "")),
        "feedback_industry": _str(qc.get("feedback_industry", "")),
        "feedback_financial": _str(qc.get("feedback_financial", "")),
        "feedback_buyers": _str(qc.get("feedback_buyers", "")),
    }


def _need(qc: Dict[str, Any], flag: str, score_key: str) -> bool:
    try:
        return bool(qc.get(flag)) or (float(qc.get(score_key, 0.0)) < QUALITY_THRESHOLD)
    except Exception:
        return False


def router(state: ChatState) -> str:
    qc = state.get("qc_json") or {}
    if "error" in qc:
        return "end"

    redo = any([
        _need(qc, "needs_rework_company", "company_score"),
        _need(qc, "needs_rework_industry", "industry_score"),
        _need(qc, "needs_rework_financial", "financial_score"),
        _need(qc, "needs_rework_buyers", "buyers_score"),
    ])

    if redo and (state.get("round") or 0) < MAX_ROUNDS:
        return "redo"
    return "end"


def n_finalize(state: ChatState) -> Dict[str, Any]:
    comp = _str(state.get("company_report", ""))
    ind = _str(state.get("industry_report", ""))
    fin = _str(state.get("financial_model", ""))
    buy = _str(state.get("buyers", ""))
   
    body = (
        "## Company Report\n\n" + comp +
        "\n\n---\n\n## Industry Report\n\n" + ind +
        "\n\n---\n\n## Financial Model (Swarm)\n\n" + fin +
        "\n\n---\n\n## Potential Buyers (Swarm)\n\n" + buy
    )
    # APPEND vào messages (add_messages sẽ cộng dồn, không xóa Human)
    return {"messages": [AIMessage(content=body)]}


# ---------------- Build graph ----------------
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("parse_input", n_parse_input)
    g.add_node("company", n_company)
    g.add_node("industry", n_industry)
    g.add_node("financial_swarm", n_financial)
    g.add_node("buyers_swarm", n_buyers)
    g.add_node("supervisor_qc", n_qc)
    g.add_node("set_feedback", n_set_feedback)
    g.add_node("finalize", n_finalize)

    g.add_edge(START, "parse_input")
    g.add_edge("parse_input", "company")
    g.add_edge("company", "industry")
    g.add_edge("industry", "financial_swarm")
    g.add_edge("financial_swarm", "buyers_swarm")
    g.add_edge("buyers_swarm", "supervisor_qc")
    g.add_conditional_edges("supervisor_qc", router, {"redo": "set_feedback", "end": "finalize"})
    g.add_edge("set_feedback", "company")
    g.add_edge("finalize", END)
    return g.compile()


# === EXPORTS (langgraph.json trỏ vào "app:supervisor_graph") ===
supervisor_graph = build_graph()
app = supervisor_graph  # nếu đâu đó vẫn gọi "app:app"

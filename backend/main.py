# main.py
import json, time, uuid
from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
)

# ==== agents / swarms ====
from agents.company_agent import deep_research_agent
from agents.industry_agent import industry_research_agent
from agents.financial_model import run_financial_swarm
from agents.potential_buyers import run_potential_buyers_swarm
from agents.buyerlist import run_buyerlist

QUALITY_THRESHOLD = 0.80
MAX_ROUNDS = 1

def merge_dict(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
    # hợp nhất nông; nếu cần deep-merge có thể tự viết đệ quy
    return {**(a or {}), **(b or {})}

# ========================= STATE =========================
class ChatState(TypedDict, total=False):
    # NEW: nhận “input” thô từ payload /stream
    input: Any

    messages: Annotated[List[BaseMessage], add_messages]

    company_query: str
    round: int

    company_report: str
    industry_report: str
    financial_model: str
    # for buyerlist
    financial_assumptions: str
    
    potential_buyers: str
    buyerlist: str 
    qc_json: Dict[str, Any]

    feedback_company: str
    feedback_industry: str
    feedback_financial: str
    feedback_buyers: str

    # stream timing
    tool_ids: Dict[str, str]
    tool_started: Dict[str, float]
    
    kb: Annotated[Dict[str, Any], merge_dict]


# ======================== HELPERS ========================
def _coerce_str(x: Any) -> str:

    if isinstance(x, str):
        return x
    try:
        msgs = x.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    c = getattr(x, "content", None)
    if isinstance(c, str):
        return c
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def _run_agent(agent, prompt: str) -> str:

    last_err = None
    for attempt in range(4):
        try:
            out = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                               config={"recursion_limit": 100})
            return _coerce_str(out)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (2 ** attempt))
    return f"[tool_error] Upstream model error: {type(last_err).__name__}: {last_err}"

def _make_revision_prompt(base_query: str, feedback: str) -> str:
    if feedback:
        return (
            f"{base_query}\n\n"
            "Revise using this feedback. Output ONLY the final Markdown report, numbered citations required:\n"
            f"{feedback}"
        )
    return base_query


# ========================= NODES =========================

def n_parse_input(state: ChatState) -> Dict[str, Any]:
    TARGET_KEYS = {"input", "company_query", "query", "message", "text", "content"}

    def pick_str(x) -> str | None:
        if isinstance(x, str):
            s = x.strip()
            return s or None
        return None

    def deep_find(obj) -> str | None:
        from langchain_core.messages import HumanMessage
        if isinstance(obj, dict):
            for k in TARGET_KEYS:
                if k in obj:
                    s = deep_find(obj[k])
                    if s:
                        return s
            for v in obj.values():
                s = deep_find(v)
                if s:
                    return s
            return None
        if isinstance(obj, list):
            for v in reversed(obj):
                s = deep_find(v)
                if s:
                    return s
            return None
        if isinstance(obj, HumanMessage):
            return pick_str(obj.content)
        return pick_str(obj)

    # 1) từ state['input'] (có thể là dict lồng)
    q = deep_find(state.get("input"))

    # 2) từ messages (nếu FE đã bơm)
    if not q:
        q = deep_find(state.get("messages"))

    # 3) quét toàn bộ state phòng SDK đặt chỗ khác
    if not q:
        q = deep_find(state)

    if not q:
        raise ValueError("Missing company name. Please type a company name.")

    # đảm bảo FE có bubble user
    need_append = True
    for m in reversed(state.get("messages") or []):
        from langchain_core.messages import HumanMessage
        if isinstance(m, HumanMessage) and m.content == q:
            need_append = False
            break

    msg_list = []
    if need_append:
        from langchain_core.messages import HumanMessage
        msg_list.append(HumanMessage(content=q))

    return {"company_query": q, "round": 0, "messages": msg_list}


def n_announce_tools(state: ChatState) -> Dict[str, Any]:
    names = ["company_research", "industry_research", "financial_model", "potential_buyers", "buyerlist"]
    tool_ids = {n: uuid.uuid4().hex for n in names}
    tool_calls = [{"id": tool_ids[n], "type":"function", "function":{"name": n, "arguments": "{}"}} for n in names]
    ai = AIMessage(content="", additional_kwargs={"tool_calls": tool_calls})
    return {"messages": [ai], "tool_ids": tool_ids, "tool_started": {}}

def _tool_done(name: str, state: ChatState, content: str, *, elapsed_ms: int | None = None) -> Dict[str, Any]:
    tid = (state.get("tool_ids") or {}).get(name) or uuid.uuid4().hex
    if elapsed_ms is None:
        started = (state.get("tool_started") or {}).get(name, time.time())
        elapsed_ms = int((time.time() - started) * 1000)

    tool_msg = ToolMessage(
        content=_coerce_str(content),
        name=name,
        tool_call_id=tid,
        additional_kwargs={"elapsed_ms": int(elapsed_ms)},
    )
    return {"messages": [tool_msg]}

def n_company(state: ChatState) -> Dict[str, Any]:
    q  = state["company_query"]
    fb = state.get("feedback_company", "")
    prompt = _make_revision_prompt(q, fb)

    t0 = time.time()
    txt = _run_agent(deep_research_agent, prompt)
    dt = int((time.time() - t0) * 1000)

    done = _tool_done("company_research", state, txt, elapsed_ms=dt)
    return {
        "company_report": txt,
        # ghi vào kb để agent khác dùng lại
        "kb": {
            "company": {
                "query": q,
                "feedback": fb,
                "report_md": txt,   # có thể thay bằng JSON rút gọn nếu bạn đã trích xuất card
            }
        },
        **done,
    }

def n_industry(state: ChatState) -> Dict[str, Any]:
    q  = state["company_query"]
    fb = state.get("feedback_industry", "")
    prompt = _make_revision_prompt(q, fb)

    t0 = time.time()
    txt = _run_agent(industry_research_agent, prompt)
    dt = int((time.time() - t0) * 1000)

    done = _tool_done("industry_research", state, txt, elapsed_ms=dt)
    return {
        "industry_report": txt,
        "kb": {
            "industry": {
                "feedback": fb,
                "report_md": txt,   # nếu có bảng M&A riêng, bạn có thể lưu ma_table_md tại đây
            }
        },
        **done,
    }

def n_financial(state: ChatState) -> Dict[str, Any]:
    q  = state["company_query"]
    fb = _coerce_str(state.get("feedback_financial", ""))

    t0 = time.time()
    md, assumptions = "", ""
    try:
        out = run_financial_swarm(q, feedback=fb)
        if isinstance(out, dict):
            md = _coerce_str(out.get("markdown", ""))
            assumptions = _coerce_str(out.get("assumptions_json", ""))
        else:
            md = _coerce_str(out)
            assumptions = ""
    except Exception as e:
        md = f"[tool_error] {type(e).__name__}: {e}"
        assumptions = ""

    dt = int((time.time() - t0) * 1000)
    done = _tool_done("financial_model", state, md, elapsed_ms=dt)
    return {
        "financial_model": md,
        "financial_assumptions": assumptions,
        "kb": {
            "financial": {
                "feedback": fb,
                "model_md": md,
                "assumptions_json": assumptions,
            }
        },
        **done,
    }

def n_buyers(state: ChatState) -> Dict[str, Any]:
    q  = state["company_query"]
    fb = _coerce_str(state.get("feedback_buyers", ""))

    t0 = time.time()
    try:
        out = run_potential_buyers_swarm(q, feedback=fb)
        txt = _coerce_str(out)
    except Exception as e:
        txt = f"[tool_error] {type(e).__name__}: {e}"

    dt = int((time.time() - t0) * 1000)
    done = _tool_done("potential_buyers", state, txt, elapsed_ms=dt)
    return {
        "potential_buyers": txt,
        "kb": {
            "pb": {
                "feedback": fb,
                "summary_md": txt,
                # nếu bạn có mảng sources/citations, lưu vào đây: "citations": [...]
            }
        },
        **done,
    }

def n_buyerlist(state: ChatState) -> Dict[str, Any]:
    q   = state["company_query"]
    # ưu tiên lấy từ kb nếu đã có, fallback sang field state
    fm  = _coerce_str((state.get("kb", {}) or {}).get("financial", {}).get("model_md")
                      or state.get("financial_model", ""))
    ass = _coerce_str((state.get("kb", {}) or {}).get("financial", {}).get("assumptions_json")
                      or state.get("financial_assumptions", ""))
    fb  = _coerce_str(state.get("feedback_buyers", ""))

    t0 = time.time()
    try:
        txt = run_buyerlist(q, fm, ass, feedback=fb)
    except Exception as e:
        txt = f"[tool_error] {type(e).__name__}: {e}"

    dt = int((time.time() - t0) * 1000)
    done = _tool_done("buyerlist", state, txt, elapsed_ms=dt)
    return {
        "buyerlist": txt,
        "kb": {
            "buyerlist": {
                "feedback": fb,
                "summary_md": txt,
                "used_assumptions": bool(ass),
            }
        },
        **done,
    }


def qc_score(company: str, industry: str, financial: str, buyers: str) -> dict:

    def score(x: str) -> float:
        n = len(x or "")
        if n >= 2000: return 9.0
        if n >= 1000: return 8.5
        if n >= 600:  return 8.0
        if n >= 300:  return 7.5
        if n >= 150:  return 7.0
        return 6.5
    return {
        "company_score":   score(company),
        "industry_score":  score(industry),
        "financial_score": score(financial),
        "buyers_score":    score(buyers),
        "needs_rework_company":   False,
        "needs_rework_industry":  False,
        "needs_rework_financial": False,
        "needs_rework_buyers":    False,
        "feedback_company":   "",
        "feedback_industry":  "",
        "feedback_financial": "",
        "feedback_buyers":    "",
    }

def n_qc(state: ChatState) -> Dict[str, Any]:
    return {"qc_json": qc_score(
        _coerce_str(state.get("company_report", "")),
        _coerce_str(state.get("industry_report", "")),
        _coerce_str(state.get("financial_model", "")),
        _coerce_str(state.get("potential_buyers", "")),
    )}

def n_set_feedback(state: ChatState) -> Dict[str, Any]:
    qc = state.get("qc_json") or {}
    return {
        "round": (state.get("round") or 0) + 1,
        "feedback_company": _coerce_str(qc.get("feedback_company", "")),
        "feedback_industry": _coerce_str(qc.get("feedback_industry", "")),
        "feedback_financial": _coerce_str(qc.get("feedback_financial", "")),
        "feedback_buyers": _coerce_str(qc.get("feedback_buyers", "")),
    }

def router(state: ChatState) -> str:
    qc = state.get("qc_json") or {}
    def need(flag: str, score_key: str) -> bool:
        return bool(qc.get(flag)) or (float(qc.get(score_key, 0.0)) < QUALITY_THRESHOLD)
    redo = any([
        need("needs_rework_company", "company_score"),
        need("needs_rework_industry", "industry_score"),
        need("needs_rework_financial", "financial_score"),
        need("needs_rework_buyers", "buyers_score"),
    ])
    if redo and (state.get("round") or 0) < MAX_ROUNDS:
        return "redo"
    return "end"

def n_finalize(state: ChatState) -> Dict[str, Any]:
    comp = _coerce_str(state.get("company_report", ""))
    ind  = _coerce_str(state.get("industry_report", ""))
    fin  = _coerce_str(state.get("financial_model", ""))
    bl   = _coerce_str(state.get("buyerlist", ""))  
    buy  = _coerce_str(state.get("potential_buyers", ""))
    
    body = (
        "## Company Report\n\n" + comp +
        "\n\n---\n\n## Industry Report\n\n" + ind +
        "\n\n---\n\n## Financial Model \n\n" + fin +
        "\n\n---\n\n## Buyer List (uses Financial Model)\n\n" + bl +    
        "\n\n---\n\n## Potential Buyers \n\n" + buy
    )
    
    return {"messages": [AIMessage(content=body)]}

# ======================= BUILD GRAPH ======================

# Helper router: quyết định có chạy potential_buyers hay bỏ qua
def decide_pbuyers(state: ChatState) -> Dict[str, Any]:
    import os
    has_sources = bool(os.getenv("TAVILY_API_KEY"))
    # nếu cần tinh vi hơn, bạn có thể set cờ khác trong state rồi đọc ở đây
    return {"route": "potential" if has_sources else "skip"}


def build_graph():
    g = StateGraph(ChatState)

    # === Nodes ===
    g.add_node("parse_input", n_parse_input)
    g.add_node("announce_tools", n_announce_tools)
    g.add_node("company", n_company)
    g.add_node("industry", n_industry)
    g.add_node("financial_model", n_financial)
    g.add_node("buyerlist", n_buyerlist)
    g.add_node("decide_pbuyers", decide_pbuyers)   # NEW router
    g.add_node("potential_buyers", n_buyers)
    g.add_node("supervisor_qc", n_qc)
    g.add_node("set_feedback", n_set_feedback)
    g.add_node("finalize", n_finalize)

    # === Edges ===
    g.add_edge(START, "parse_input")
    g.add_edge("parse_input", "announce_tools")

    # chạy song song từ announce_tools
    g.add_edge("announce_tools", "company")
    g.add_edge("announce_tools", "industry")
    g.add_edge("announce_tools", "financial_model")   # song song với industry

    # chuỗi tài chính -> buyerlist
    g.add_edge("financial_model", "buyerlist")

    # quyết định có chạy potential_buyers không
    g.add_edge("buyerlist", "decide_pbuyers")
    g.add_conditional_edges(
        "decide_pbuyers",
        lambda s: s.get("route", "skip"),
        {"potential": "potential_buyers", "skip": "supervisor_qc"},
    )

    # nếu có chạy potential_buyers thì đi QC, còn không thì đã rẽ sang QC ở trên
    g.add_edge("potential_buyers", "supervisor_qc")

    # QC -> set_feedback (redo) hoặc finalize
    g.add_conditional_edges(
        "supervisor_qc",
        router,  # hàm router hiện có của bạn
        {"redo": "set_feedback", "end": "finalize"},
    )
    g.add_edge("set_feedback", "company")

    g.add_edge("finalize", END)

    return g.compile()


supervisor_graph = build_graph()
app = supervisor_graph
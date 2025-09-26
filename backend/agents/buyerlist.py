# agents/buyerlist.py
import os, textwrap
from typing import Optional
from langchain_openai import ChatOpenAI

def _llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)

def run_buyerlist(company: str, financial_model_md: str, assumptions_json: str, feedback: Optional[str]=None) -> str:
    prompt = f"""
            You are a BuyerList agent.

            Inputs:
            - Company: {company}
            - ASSUMPTIONS_JSON:
            {assumptions_json}
            - FINANCIAL_MODEL_MD:
            {financial_model_md}
            {f"Reviewer feedback:\n{feedback}" if feedback else ""}

TASK:
1) Parse ASSUMPTIONS_JSON to get base_year_revenue (if any), CAGR for base/bull/bear, and EBIT margins.
2) Derive a target revenue band & profitability band for an acquirer.
3) Propose 8–12 buyers grouped by Strategic vs Financial that *fit those bands*.
4) Compute FitScore = 0.5*GrowthFit + 0.3*MarginFit + 0.2*Adjacency (0–100). Show the three sub-scores.
5) Add "Assumptions & Caveats". No web search. No citations. Return Markdown only.
"""
    return _llm().invoke(prompt).content


# app.py
import os
import json
from typing import TypedDict, List, Dict, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tools import make_agent_tools

load_dotenv()

MAX_RESEARCH_CALLS = 2
MAX_REVISIONS = 2
PASS_THRESHOLD = 70

EVAL_CONFIG = {
    "spam_check": "You are a deliverability reviewer. Flag spam-trigger words, excessive punctuation/caps, and sketchy links.",
    "personalization_check": "You are a personalization reviewer. Judge whether this email uses the specific research signals or CRM notes given, vs reading like a template.",
    "tone_check": "You are an editor. Judge whether the email is concise (80-150 words), sounds human, and has exactly one clear CTA.",
}


class LeadsRequest(BaseModel):
    csv_path: str
    sender_company: str
    sender_role: str


class SalesState(TypedDict):
    lead: Dict[str, str]
    sender_company: str
    sender_role: str
    research_notes: str
    draft_subject: str
    draft_body: str
    revision_count: int
    feedback_history: List[str]
    checks: List[Dict]
    status: str


class SalesOutreachAgent:
    def __init__(self):
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.4,
        )
        self.groq_llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
        )
        self.graph = self._build_graph()

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, list):
            return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content or "")

    @staticmethod
    def _parse_email(content: str) -> Tuple[str, str]:
        if "SUBJECT:" in content and "BODY:" in content:
            after = content.split("SUBJECT:", 1)[1]
            subj, body = after.split("BODY:", 1)
            return subj.strip().splitlines()[0].strip(), body.strip()
        return "", content.strip()

    @staticmethod
    def _parse_score(content: str) -> Tuple[int, str]:
        score, feedback = 0, content.strip()
        for line in content.splitlines():
            u = line.strip().upper()
            if u.startswith("SCORE:"):
                digits = "".join(c for c in line.split(":", 1)[1] if c.isdigit())
                score = int(digits) if digits else 0
            elif u.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()
        return max(0, min(100, score)), feedback

    # ---- research: looks up program/career outcome signals, not company info ----
    def _research(self, state: SalesState) -> Dict:
        lead = state["lead"]
        tools = make_agent_tools()
        llm = self.gemini_llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        messages = [
            SystemMessage(content=(
                f"You are a research assistant for a {state['sender_role']} at "
                f"{state['sender_company']}. You have a web_search tool. Find "
                "1-3 recent, specific, credible facts about career or salary "
                "outcomes for people who complete the program this lead is "
                "interested in (e.g. average salary hike, typical roles after "
                "completion, industry demand). Search at least once. Reply "
                "with a short bulleted list, under 100 words. If nothing "
                "useful turns up, say so explicitly."
            )),
            HumanMessage(content=(
                f"Program of interest: {lead.get('program_of_interest','')}\n"
                f"Lead's behavior signal: {lead.get('comments') or 'none'}"
            )),
        ]

        for _ in range(MAX_RESEARCH_CALLS):
            response = llm.invoke(messages)
            messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                break
            for call in calls:
                fn = tool_map.get(call["name"])
                try:
                    result = fn.invoke(call["args"]) if fn else f"Unknown tool {call['name']}"
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append(HumanMessage(content=f"[{call['name']} result] {result}"))

        notes = self._text(messages[-1].content).strip()
        if not notes:
            notes = "No strong outcome data found; drafting from behavior signal only."
        return {"research_notes": notes}

    # ---- copywriter ----
    def _copywriter(self, state: SalesState) -> Dict:
        lead = state["lead"]
        feedback_block = ""
        if state["feedback_history"]:
            feedback_block = "Address this feedback from the last draft:\n- " + "\n- ".join(state["feedback_history"])

        prompt = f"""You are a {state['sender_role']} at {state['sender_company']} writing a personalized outreach email.

Lead: {lead['contact_name']}
Program of interest: {lead.get('program_of_interest','')}
Behavior signal: {lead.get('comments') or 'none'}

Career outcome research:
{state['research_notes']}

{feedback_block}

Write a short, personalized email nudging them toward enrolling or booking a
call. First name only, one clear CTA, 80-150 words, no "I hope this finds
you well". Reference their specific behavior signal (e.g. what page they
visited, what they downloaded) and the program's value.

Respond in exactly this format:
SUBJECT: <subject line>
BODY:
<email body>"""
        response = self.gemini_llm.invoke(prompt)
        subject, body = self._parse_email(self._text(response.content))
        return {"draft_subject": subject, "draft_body": body}

    # ---- all 3 checks in one node ----
    def _run_checks(self, state: SalesState) -> Dict:
        results = []
        for key, criteria in EVAL_CONFIG.items():
            prompt = f"""{criteria}

Subject: {state['draft_subject']}
Body:
{state['draft_body']}

Respond in exactly this format:
SCORE: <integer 0-100>
FEEDBACK: <one or two specific, actionable sentences>"""
            response = self.groq_llm.invoke(prompt)
            score, feedback = self._parse_score(self._text(response.content))
            results.append({"check": key, "score": score, "passed": score >= PASS_THRESHOLD, "feedback": feedback})
        return {"checks": results}

    def _score_gate(self, state: SalesState) -> Dict:
        checks = state["checks"]
        if all(c["passed"] for c in checks):
            return {"status": "approved"}
        feedback = [f"{c['check']}: {c['feedback']}" for c in checks if not c["passed"]]
        if state["revision_count"] < MAX_REVISIONS:
            return {"revision_count": state["revision_count"] + 1, "feedback_history": feedback, "status": "pending"}
        return {"status": "flagged"}

    def _route(self, state: SalesState) -> str:
        return {"approved": "approved", "flagged": "flagged"}.get(state["status"], "revise")

    def _build_graph(self):
        g = StateGraph(SalesState)
        g.add_node("research", self._research)
        g.add_node("copywriter", self._copywriter)
        g.add_node("checks", self._run_checks)
        g.add_node("score_gate", self._score_gate)
        g.add_node("approved", lambda s: {})
        g.add_node("flagged", lambda s: {})

        g.set_entry_point("research")
        g.add_edge("research", "copywriter")
        g.add_edge("copywriter", "checks")
        g.add_edge("checks", "score_gate")
        g.add_conditional_edges("score_gate", self._route, {"revise": "copywriter", "approved": "approved", "flagged": "flagged"})
        g.add_edge("approved", END)
        g.add_edge("flagged", END)
        return g.compile()


def _initial_state(lead: Dict, sender_company: str, sender_role: str) -> Dict:
    return {
        "lead": lead, "sender_company": sender_company, "sender_role": sender_role,
        "research_notes": "", "draft_subject": "", "draft_body": "",
        "revision_count": 0, "feedback_history": [], "checks": [], "status": "pending",
    }


agent = SalesOutreachAgent()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])
OUTBOX: List[Dict] = []


@app.post("/generate-emails")
def generate_emails(request: LeadsRequest):
    if not os.path.isfile(request.csv_path):
        return {"error": f"'{request.csv_path}' is not a valid file."}

    leads = pd.read_csv(request.csv_path).fillna("").to_dict(orient="records")
    outbox, flagged = [], []

    for lead in leads:
        result = agent.graph.invoke(_initial_state(lead, request.sender_company, request.sender_role))
        record = {
            "contact_name": lead.get("contact_name", ""), "email": lead.get("email", ""),
            "subject": result["draft_subject"], "body": result["draft_body"],
            "research_notes": result["research_notes"], "checks": result["checks"], "revisions": result["revision_count"],
        }
        (outbox if result["status"] == "approved" else flagged).append(record)

    OUTBOX.extend(outbox)
    return {"total_leads": len(leads), "approved": len(outbox), "flagged": len(flagged), "outbox": outbox, "flagged_leads": flagged}


@app.post("/generate-emails/stream")
async def generate_emails_stream(request: LeadsRequest):
    if not os.path.isfile(request.csv_path):
        async def err():
            yield f"event: error\ndata: {json.dumps({'error': 'invalid csv_path'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    leads = pd.read_csv(request.csv_path).fillna("").to_dict(orient="records")
    STAGE_LABELS = {"research": "researching", "copywriter": "drafting", "checks": "reviewing", "score_gate": "scoring"}

    async def gen():
        for lead in leads:
            yield f"event: lead_start\ndata: {json.dumps({'contact_name': lead.get('contact_name','')})}\n\n"
            status = "pending"
            async for update in agent.graph.astream(_initial_state(lead, request.sender_company, request.sender_role), stream_mode="updates"):
                for node, delta in update.items():
                    label = STAGE_LABELS.get(node, node)
                    yield f"data: {json.dumps({'stage': label})}\n\n"
                    if delta and "status" in delta:
                        status = delta["status"]
            yield f"event: lead_done\ndata: {json.dumps({'contact_name': lead.get('contact_name',''), 'status': status})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/outbox")
def get_outbox():
    return {"count": len(OUTBOX), "emails": OUTBOX}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
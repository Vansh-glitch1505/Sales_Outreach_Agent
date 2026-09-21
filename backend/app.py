import os
import json
import io
from typing import TypedDict, List, Dict, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    "spam_check": (
        "You are a deliverability reviewer. Flag spam-trigger words, "
        "excessive punctuation/caps, and sketchy links."
    ),
    "personalization_check": (
        "You are a personalization reviewer. Judge whether this email "
        "uses the specific research signals or CRM notes given, vs "
        "reading like a template."
    ),
    "tone_check": (
        "You are an editor. Judge whether the email is concise "
        "(80-150 words), sounds human, and has exactly one clear CTA."
    ),
}


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
            model="gemini-3.6-flash",
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
            return "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )

        return str(content or "")

    @staticmethod
    def _parse_email(content: str) -> Tuple[str, str]:

        if "SUBJECT:" in content and "BODY:" in content:

            after = content.split("SUBJECT:", 1)[1]

            subj, body = after.split("BODY:", 1)

            return (
                subj.strip().splitlines()[0].strip(),
                body.strip(),
            )

        return "", content.strip()

    @staticmethod
    def _parse_score(content: str) -> Tuple[int, str]:

        score, feedback = 0, content.strip()

        for line in content.splitlines():

            u = line.strip().upper()

            if u.startswith("SCORE:"):

                digits = "".join(
                    c for c in line.split(":", 1)[1]
                    if c.isdigit()
                )

                score = int(digits) if digits else 0

            elif u.startswith("FEEDBACK:"):

                feedback = line.split(":", 1)[1].strip()

        return max(0, min(100, score)), feedback

    # ---------------------------------------------------------
    # RESEARCH
    # ---------------------------------------------------------

    def _research(self, state: SalesState) -> Dict:

        lead = state["lead"]

        tools = make_agent_tools()

        llm = self.gemini_llm.bind_tools(tools)

        tool_map = {
            t.name: t
            for t in tools
        }

        messages = [

            SystemMessage(
                content=(
                    f"You are a research assistant for a "
                    f"{state['sender_role']} at "
                    f"{state['sender_company']}. "
                    "You have a web_search tool. Find "
                    "1-3 recent, specific, credible facts about "
                    "career or salary outcomes for people who "
                    "complete the program this lead is interested "
                    "in (e.g. average salary hike, typical roles "
                    "after completion, industry demand). Search "
                    "at least once. Reply with a short bulleted "
                    "list, under 100 words. If nothing useful "
                    "turns up, say so explicitly."
                )
            ),

            HumanMessage(
                content=(
                    f"Program of interest: "
                    f"{lead.get('program_of_interest', '')}\n"
                    f"Lead's behavior signal: "
                    f"{lead.get('comments') or 'none'}"
                )
            ),
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

                    result = (
                        fn.invoke(call["args"])
                        if fn
                        else f"Unknown tool {call['name']}"
                    )

                except Exception as e:

                    result = f"Tool error: {e}"

                messages.append(
                    HumanMessage(
                        content=(
                            f"[{call['name']} result] "
                            f"{result}"
                        )
                    )
                )

        notes = self._text(
            messages[-1].content
        ).strip()

        if not notes:

            notes = (
                "No strong outcome data found; "
                "drafting from behavior signal only."
            )

        return {
            "research_notes": notes
        }

    # ---------------------------------------------------------
    # COPYWRITER
    # ---------------------------------------------------------

    def _copywriter(self, state: SalesState) -> Dict:

        lead = state["lead"]

        feedback_block = ""

        if state["feedback_history"]:

            feedback_block = (
                "Address this feedback from the last draft:\n- "
                + "\n- ".join(
                    state["feedback_history"]
                )
            )

        prompt = f"""
You are a {state['sender_role']} at
{state['sender_company']} writing a personalized outreach email.

Lead: {lead['contact_name']}

Program of interest:
{lead.get('program_of_interest', '')}

Behavior signal:
{lead.get('comments') or 'none'}

Career outcome research:
{state['research_notes']}

{feedback_block}

Write a short, personalized email nudging them toward
enrolling or booking a call.

First name only.
One clear CTA.
80-150 words.
No "I hope this finds you well".

Reference their specific behavior signal
(e.g. what page they visited, what they downloaded)
and the program's value.

Respond in exactly this format:

SUBJECT: <subject line>
BODY:

<email body>
"""

        response = self.gemini_llm.invoke(prompt)

        subject, body = self._parse_email(
            self._text(response.content)
        )

        return {
            "draft_subject": subject,
            "draft_body": body,
        }

    # ---------------------------------------------------------
    # QUALITY CHECKS
    # ---------------------------------------------------------

    def _run_checks(self, state: SalesState) -> Dict:

        results = []

        for key, criteria in EVAL_CONFIG.items():

            prompt = f"""
{criteria}

Subject:
{state['draft_subject']}

Body:
{state['draft_body']}

Respond in exactly this format:

SCORE: <integer 0-100>
FEEDBACK: <one or two specific, actionable sentences>
"""

            response = self.groq_llm.invoke(prompt)

            score, feedback = self._parse_score(
                self._text(response.content)
            )

            results.append(
                {
                    "check": key,
                    "score": score,
                    "passed": score >= PASS_THRESHOLD,
                    "feedback": feedback,
                }
            )

        return {
            "checks": results
        }

    # ---------------------------------------------------------
    # SCORE GATE
    # ---------------------------------------------------------

    def _score_gate(self, state: SalesState) -> Dict:

        checks = state["checks"]

        if all(
            c["passed"]
            for c in checks
        ):

            return {
                "status": "approved"
            }

        feedback = [
            f"{c['check']}: {c['feedback']}"
            for c in checks
            if not c["passed"]
        ]

        if state["revision_count"] < MAX_REVISIONS:

            return {
                "revision_count": (
                    state["revision_count"] + 1
                ),
                "feedback_history": feedback,
                "status": "pending",
            }

        return {
            "status": "flagged"
        }

    # ---------------------------------------------------------
    # ROUTING
    # ---------------------------------------------------------

    def _route(self, state: SalesState) -> str:

        return {
            "approved": "approved",
            "flagged": "flagged",
        }.get(
            state["status"],
            "revise",
        )

    # ---------------------------------------------------------
    # LANGGRAPH
    # ---------------------------------------------------------

    def _build_graph(self):

        g = StateGraph(SalesState)

        g.add_node(
            "research",
            self._research
        )

        g.add_node(
            "copywriter",
            self._copywriter
        )

        g.add_node(
            "checks",
            self._run_checks
        )

        g.add_node(
            "score_gate",
            self._score_gate
        )

        g.add_node(
            "approved",
            lambda s: {}
        )

        g.add_node(
            "flagged",
            lambda s: {}
        )

        g.set_entry_point("research")

        g.add_edge(
            "research",
            "copywriter"
        )

        g.add_edge(
            "copywriter",
            "checks"
        )

        g.add_edge(
            "checks",
            "score_gate"
        )

        g.add_conditional_edges(
            "score_gate",
            self._route,
            {
                "revise": "copywriter",
                "approved": "approved",
                "flagged": "flagged",
            },
        )

        g.add_edge(
            "approved",
            END
        )

        g.add_edge(
            "flagged",
            END
        )

        return g.compile()


# -------------------------------------------------------------
# INITIAL STATE
# -------------------------------------------------------------

def _initial_state(
    lead: Dict,
    sender_company: str,
    sender_role: str
) -> Dict:

    return {
        "lead": lead,
        "sender_company": sender_company,
        "sender_role": sender_role,
        "research_notes": "",
        "draft_subject": "",
        "draft_body": "",
        "revision_count": 0,
        "feedback_history": [],
        "checks": [],
        "status": "pending",
    }


# -------------------------------------------------------------
# FASTAPI
# -------------------------------------------------------------

agent = SalesOutreachAgent()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTBOX: List[Dict] = []


# -------------------------------------------------------------
# GENERATE EMAILS
# -------------------------------------------------------------

@app.post("/generate-emails")
async def generate_emails(
    file: UploadFile = File(...),
    sender_company: str = Form(...),
    sender_role: str = Form(...)
):

    try:

        contents = await file.read()

        leads = pd.read_csv(
            io.BytesIO(contents),
            encoding="cp1252"
        ).fillna("").to_dict(
            orient="records"
        )

    except Exception as e:

        return {
            "error": f"Could not read CSV: {e}"
        }

    outbox = []
    flagged = []

    for lead in leads:

        result = agent.graph.invoke(
            _initial_state(
                lead,
                sender_company,
                sender_role
            )
        )

        record = {
            "contact_name": lead.get(
                "contact_name",
                ""
            ),
            "email": lead.get(
                "email",
                ""
            ),
            "subject": result[
                "draft_subject"
            ],
            "body": result[
                "draft_body"
            ],
            "research_notes": result[
                "research_notes"
            ],
            "checks": result[
                "checks"
            ],
            "revisions": result[
                "revision_count"
            ],
        }

        if result["status"] == "approved":

            outbox.append(record)

        else:

            flagged.append(record)

    OUTBOX.extend(outbox)

    return {
        "total_leads": len(leads),
        "approved": len(outbox),
        "flagged": len(flagged),
        "outbox": outbox,
        "flagged_leads": flagged,
    }


# -------------------------------------------------------------
# STREAMING ENDPOINT
# -------------------------------------------------------------
# Keep this endpoint for now.
# We are using /generate-emails from the Gradio frontend.

@app.post("/generate-emails/stream")
async def generate_emails_stream(
    request
):

    async def err():

        yield (
            "event: error\n"
            'data: {"error": "Streaming endpoint '
            'not configured for file uploads yet"}\n\n'
        )

    return StreamingResponse(
        err(),
        media_type="text/event-stream"
    )


# -------------------------------------------------------------
# OUTBOX
# -------------------------------------------------------------

@app.get("/outbox")
def get_outbox():

    return {
        "count": len(OUTBOX),
        "emails": OUTBOX,
    }


# -------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# -------------------------------------------------------------
# LOCAL DEVELOPMENT
# -------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        ),
    )
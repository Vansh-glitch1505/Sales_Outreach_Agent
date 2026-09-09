# 📧 Sales Outreach Agent

An agentic AI system that turns a raw leads CSV into personalized, research-backed cold outreach emails — automatically researched, drafted, critiqued, and revised until they pass quality gates.

Built with **LangGraph**, **FastAPI**, and **Gradio**.

---

## ✨ What it does

Give it a CSV of leads and your company/role, and for **each lead** the agent:

1. **Researches** — searches the web for recent, credible career/salary outcome signals relevant to the lead's program of interest
2. **Drafts** — writes a short, personalized email referencing the lead's specific behavior (e.g. "downloaded the brochure") and the research findings
3. **Reviews** — runs the draft through 3 independent AI evaluators (spam check, personalization check, tone check)
4. **Revises or ships** — if any check fails, feedback is sent back to the copywriter for another pass (up to 2 revisions); if it still fails, the lead is flagged for manual review instead of sending a weak email

The result: an **outbox** of approved, ready-to-send emails and a **flagged** list of leads that need a human touch.

---

## 🧠 Architecture

```
        ┌───────────┐
        │  research │  ← Gemini + web_search / fetch_website tools
        └─────┬─────┘
              ▼
        ┌───────────┐
   ┌───▶│ copywriter │  ← Gemini drafts SUBJECT + BODY
   │    └─────┬─────┘
   │          ▼
   │    ┌───────────┐
   │    │   checks   │  ← Groq: spam / personalization / tone (0–100 each)
   │    └─────┬─────┘
   │          ▼
   │    ┌───────────┐
   └────│ score_gate │──▶ approved / flagged
        └───────────┘
```

Implemented as a **LangGraph `StateGraph`** with a revise-loop: failed checks route feedback straight back into the `copywriter` node, capped at `MAX_REVISIONS` before the lead is flagged instead of looping forever.

### Why two LLMs?
- **Gemini 2.5 Flash** — research (tool-calling) and copywriting, where quality and tool support matter most
- **Groq (`openai/gpt-oss-20b`)** — the 3 evaluation checks, where speed matters and the task is simple scoring/classification

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLMs | Google Gemini 2.5 Flash, Groq (OSS-20B) |
| Web research | DuckDuckGo Search (`ddgs`) + BeautifulSoup |
| Backend API | FastAPI (with SSE streaming support) |
| Frontend | Gradio |
| Data | Pandas |

---

## 📁 Project Structure

```
.
├── app.py              # LangGraph agent + FastAPI backend
├── app_ui.py            # Gradio frontend
├── tools.py              # web_search & fetch_company_website tools
├── synthetic_leads.csv   # sample leads dataset
└── .env                  # API keys (not committed)
```

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/Vansh-glitch1505/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Set up environment variables
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

### 3. Run the backend
```bash
python app.py
# FastAPI server running on http://localhost:8000
```

### 4. Run the frontend
```bash
python app_ui.py
# Gradio UI running on http://localhost:7860
```

### 5. Use it
- Open the Gradio UI
- Upload a leads CSV (see format below)
- Enter your company name and role
- Click **Generate emails**
- Get a summary + downloadable outbox and flagged tables

---

## 📄 Leads CSV Format

| Column | Description |
|---|---|
| `contact_name` | Lead's name |
| `email` | Lead's email |
| `program_of_interest` | What they're interested in |
| `country` / `city` | Location |
| `comments` | Behavior signal (e.g. "downloaded brochure", "viewed pricing") |

A `synthetic_leads.csv` sample is included to try the pipeline immediately.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate-emails` | POST | Run the full pipeline synchronously, returns outbox + flagged leads |
| `/generate-emails/stream` | POST | Same pipeline, streamed via Server-Sent Events (live per-lead stage updates) |
| `/outbox` | GET | Retrieve all approved emails generated so far |
| `/health` | GET | Health check |

---

## 🎯 Quality Gates

Every email must score ≥ **70/100** on all three checks before approval:

- **Spam check** — flags spam-trigger words, excessive punctuation/caps, sketchy links
- **Personalization check** — verifies the email actually uses the research/CRM signal, not generic template language
- **Tone check** — enforces 80–150 words, human tone, exactly one clear CTA

Emails that don't pass after 2 revision attempts are routed to a **flagged** queue for manual review — never sent unchecked.

---

## 🔮 Possible Extensions

- Real CRM integration (HubSpot / Salesforce) instead of CSV upload
- Send emails directly via SMTP/Gmail API with human-in-the-loop approval
- A/B testing across subject line variants
- Persistent storage (currently in-memory `OUTBOX`)

---

## 👤 Author

**Vansh** — AI & Data Science Engineering, TSEC Mumbai
[GitHub](https://github.com/Vansh-glitch1505)

# Orbital AI — CLAUDE.md

Portfolio demo deployed at **ortaeir.com**. Target audience: recruiters.

---

## Quick Start

```bash
pip install -r requirements.txt
python flask_app.py          # runs on http://localhost:5001
```

> `app.py` is a separate Streamlit redirect page — NOT the Flask app. Do not confuse the two.

---

## Project Structure

```
AI AGENT/
├── flask_app.py             # Flask app factory + entry point (port 5001)
├── flask_routes/
│   ├── auth.py              # Login, /demo auto-login, session management
│   ├── dashboard.py         # /dashboard + /api/kpis + /api/charts
│   ├── chat.py              # /chat + /api/chat
│   ├── prediction.py        # /prediction + /api/prediction/*
│   ├── consultant.py        # /consultant + /api/consultant/* (agent: Zyon)
│   └── utils.py             # login_required decorator
├── agents/
│   ├── Manager.py           # Orchestrator — routes all requests
│   ├── Sales_Analyst.py     # Revenue, countries, trends
│   ├── Product_Analyst.py   # Product performance, returns
│   ├── Customer_Analyst.py  # Segmentation, top customers
│   ├── Prediction_Analyst.py# ML forecasting, slow movers, growth
│   ├── Code_Executor.py     # Persistent Python sandbox (subprocess + matplotlib)
│   └── Data_Agent.py        # Startup data loading only
├── flask_agents.py          # Singleton manager instance (get_manager())
├── flask_templates/         # Jinja2 HTML templates
├── flask_static/            # CSS, JS, avatars
└── data/
    └── mixed_online_retail.csv   # The single source of truth dataset
```

---

## Pages

| Route | Blueprint | Purpose |
|---|---|---|
| `/` | index (flask_app.py) | Landing page |
| `/login` | auth_bp | Login form (rate-limited: 5 attempts, 10-min lockout) |
| `/demo` | auth_bp | Auto-login as Demo viewer (no password) |
| `/dashboard` | dashboard_bp | KPI cards + Plotly.js trend charts |
| `/chat` | chat_bp | Conversational AI with all agents |
| `/prediction` | prediction_bp | ML forecast + product trend charts |
| `/consultant` | consultant_bp | Business strategy advisor (agent: Zyon) |

Auth: two hardcoded admin users (`Or`, `Taeir`). Sessions last 30 minutes.

---

## Agent Architecture

```
User message
  → flask_routes/{chat,consultant}.py
  → flask_agents.get_manager()
  → Manager._route_to_agent()     ← gpt-4o classifier
  → ReAct agent (LangGraph)       ← gpt-4o
       SalesAnalyst | ProductAnalyst | CustomerAnalyst | PredictionAnalyst | Zyon
  → tools: execute_python + domain-specific pre-built tools
  → Manager yields {"type": "result", "content": ..., "agent_label": ...}
```

- All agents use **gpt-4o**
- All agents share one `CodeExecutor` sandbox per session
- `DataAgent` runs at startup only — it is NOT invoked per request
- `Manager.handle_request()` → chat agents; `Manager.handle_consultant_request()` → Zyon
- Get a manager instance: `from flask_agents import get_manager; manager = get_manager()`

---

## Data

- Single source: `data/mixed_online_retail.csv`
- Loaded once at startup as a pandas DataFrame, injected into all analyst classes
- Pre-computations (aggregations, ML models) run in each analyst's `__init__` — changing column names breaks everything downstream
- The DataFrame variable name in agent code contexts is always `df`

---

## Chart Pipeline

Two completely separate chart systems:

### 1. Dashboard & Prediction pages — Plotly.js
- Flask routes (`dashboard.py`, `prediction.py`) return pre-aggregated JSON via `/api/charts` and `/api/prediction/charts`
- Browser renders interactive charts using **Plotly.js 2.27.0**
- No images, no matplotlib involved

### 2. Agent-generated charts — matplotlib → base64 PNG

| Step | Location | Detail |
|---|---|---|
| Setup | `Code_Executor.py:18-20` | `matplotlib.use("Agg")` — non-interactive backend |
| Styling | `Code_Executor.py:104-131` | `_apply_chart_style()` — seaborn-v0_8-whitegrid, DPI 120, fig size (10,5) |
| Execution | `Code_Executor.py:138-218` | `_subprocess_worker()` — runs agent matplotlib code in child process |
| Capture | `Code_Executor.py:205-218` | Detects figures via `plt.get_fignums()`, saves to `BytesIO` as PNG at DPI 150, base64-encodes |
| Buffer | `Code_Executor.py:265, 367-374` | `self._charts: list[str]` + `get_pending_charts()` clears and returns the list |
| Manager wrapper | `Manager.py:888-893` | `manager.get_pending_charts()` delegates to CodeExecutor |

**Known gap:** `flask_routes/chat.py:api_chat()` (line 40) never calls `get_pending_charts()` — charts generated during chat are silently discarded. This is a planned fix.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | All agent LLM calls |
| `SECRET_KEY` / `FLASK_SECRET_KEY` | No | Flask session signing (has fallback) |

---

## Tech Stack

**Backend:** Flask, LangChain + LangGraph, OpenAI SDK (gpt-4o), pandas, numpy, matplotlib (Agg), seaborn, python-dotenv

**Frontend:** Plotly.js 2.27.0 (loaded in base.html), marked.js (markdown in chat), vanilla JS, CSS custom properties for dark/light theme

**No tests. No linting tools.**

---

## Planned Features (not yet built)

- **Charts in chat:** Wire `get_pending_charts()` into `flask_routes/chat.py:api_chat()` and embed base64 images in the frontend chat bubble
- **Custom data upload:** User uploads their own CSV; AI auto-identifies and maps columns to the expected schema

---

## Response Style Preference

Keep responses **short and direct**. No summaries at the end of turns.

# Email Assistant — a deployable Deep Agent

A stateful email assistant for **Morgan** built with [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/). It manages an inbox over time: it triages incoming email (via a delegated **triage subagent**), drafts replies, and remembers senders and Liam's preferences across conversations using long-term memory.

It's built to run locally with `langgraph dev` and to deploy to **LangSmith Deployments**.

## What's inside

| File | Purpose |
|------|---------|
| `agent.py` | The deep agent — model setup, email tools, a triage subagent, and a `CompositeBackend` that persists `/memories/*` |
| `langgraph.json` | Graph registration for `langgraph dev` / LangSmith |
| `pyproject.toml` | Dependencies |
| `.env.example` | Template for the API keys you need |

The email tools (`write_email`, `schedule_meeting`, `check_calendar_availability`) are placeholders that return strings — swap in real integrations to make it live.

## Setup

```bash
# 1. Install dependencies (using uv)
uv sync

# 2. Add your keys
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY (and LANGSMITH_API_KEY for tracing)
```

## Run locally

```bash
uv run langgraph dev
```

This opens LangGraph Studio with the **Email Assistant** graph. The platform provisions the store and checkpointer automatically, so the agent's `/memories/*` persist across conversations — view them with the **"memory"** button in Studio.

Try messages like:
- *"Who do I need to respond to today?"*
- *"A new email from Jordan in Sales arrived asking for a demo next week — triage it and draft a reply."*
- *"Remember that I prefer Wednesday meetings."*

## Deploy to LangSmith

1. Push this repo to GitHub (e.g. `github.com/langchain-ai/email-agent-demo`).
2. In [LangSmith](https://smith.langchain.com) → **Deployments** → **New Deployment**, connect this GitHub repo.
3. Set the environment variables (`ANTHROPIC_API_KEY`, and LangSmith vars are managed by the platform).
4. Deploy. LangSmith reads `langgraph.json`, builds the graph, and provisions a persistent store automatically.

> **Never commit `.env`** — it's gitignored. Provide real keys via the deployment's environment settings.

"""Stateful Email Assistant — a deployable Deep Agent.

A long-running executive assistant for Liam built with DeepAgents. It demonstrates:
- Email tools (write_email, schedule_meeting, check_calendar_availability) —
  placeholders that return strings, no real sending.
- A triage subagent the assistant delegates to via the built-in task() tool.
- Long-term memory via CompositeBackend: /memories/* is routed to a StoreBackend so
  sender notes, the inbox, drafts, and Liam's preferences persist across conversations;
  everything else stays in ephemeral state.

When running via `langgraph dev` (or a LangSmith Deployment), the store and
checkpointer are provisioned automatically by the platform — so we do NOT pass
them here. The persistent `/memories/` files are visible in Studio via the
"memory" button.
"""

from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=True)

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


# --- Model ---

# Default: Anthropic (needs ANTHROPIC_API_KEY)
model = init_chat_model("anthropic:claude-haiku-4-5")

# Or use OpenAI instead (needs OPENAI_API_KEY):
# model = init_chat_model("openai:gpt-4.1-mini")


# --- Tools (placeholders that return strings) ---


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: datetime,
    start_time: int,
) -> str:
    """Schedule a calendar meeting."""
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return (
        f"Meeting '{subject}' scheduled on {date_str} at {start_time} "
        f"for {duration_minutes} minutes with {len(attendees)} attendees"
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"


# --- Triage subagent ---

# The assistant delegates classification to this specialist via the task() tool.
# It has no custom tools: it inherits the deep-agent filesystem tools and shares
# /memories/, so it reads the email and writes its verdict straight back.
triage_subagent = {
    "name": "triage-agent",
    "description": (
        "Classify ONE inbox email as respond, notify, or ignore and record the verdict. "
        "Pass the inbox file id (e.g. 003) to triage."
    ),
    "system_prompt": """You are an email triage specialist for Liam, a software engineer at LangChain.

Read the email from /memories/inbox/<id>.md (only touch files under /memories/inbox/ and /memories/senders/).
Classify it into exactly one of:
- RESPOND  - needs a direct reply (meeting requests, direct questions, requests from management)
- NOTIFY   - important FYI, no reply needed (maintenance notices, deployments, announcements, FYI threads)
- IGNORE   - spam, marketing newsletters, automated noise

Then update that inbox file's `**Status:**` line to `<classification> | pending` (e.g. `notify | pending`),
and return the classification with one sentence of reasoning.""",
}


# --- Backend: persist /memories/* across conversations ---

# Default (scratch) lives in ephemeral state; /memories/* is routed to the
# platform-provided store so it survives across threads, sessions, and restarts.
def memory_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime, namespace=lambda ctx: ("memories",))
        },
    )


# --- System prompt: the assistant's persona + memory layout ---

EMAIL_ASSISTANT_INSTRUCTIONS = f"""
< Role >
You are Liam's long-running executive assistant. Liam is a software engineer at LangChain
(liam@langchain.dev). You manage his inbox over time and hold a continuing conversation
with him about it. Today's date is {datetime.now().strftime('%Y-%m-%d')}.
</ Role >

< Your memory >
You have a persistent filesystem under /memories/ that survives across conversations.
ALWAYS consult it before answering, and keep it up to date:
  /memories/user_profile.md     - Liam's standing preferences (tone, scheduling, what to ignore)
  /memories/senders/<name>.md   - what you know about each person who emails Liam
  /memories/inbox/<id>.md       - each email and its triage status (respond | notify | ignore, pending | done)
  /memories/drafts/<id>.md      - reply drafts awaiting Liam's approval
Use ls and read_file to recall, and write_file/edit_file to remember. Anything NOT under
/memories/ is scratch and will be lost.
</ Your memory >

< Tools >
You have email tools:
  write_email(to, subject, content)         - send an email
  schedule_meeting(attendees, subject, ...)  - book a meeting
  check_calendar_availability(day)           - check open slots
And a subagent you can delegate to with the task() tool:
  triage-agent - classifies a single inbox email (respond / notify / ignore)
</ Tools >

< How to work >
- When a new email arrives, first save it to /memories/inbox/<id>.md with status `pending`, then
  delegate classification to the triage-agent subagent via task() (give it the inbox file id).
  Don't classify emails yourself — that's the subagent's job. Update the sender's file with anything new.
- When Liam asks who he needs to respond to, read /memories/inbox/ and list the ones marked respond + pending.
- When asked to draft a reply, read the relevant inbox email, the sender's file, and user_profile.md,
  then write the draft to /memories/drafts/<id>.md and show it to him. Do NOT send unless he approves.
- When Liam tells you a preference, save it to /memories/user_profile.md.
Keep replies in Liam's preferred tone. When referencing file paths, use backticks like `path/file.md`.
"""


# --- Agent ---

# No checkpointer/store passed here: the platform provisions them automatically.
agent = create_deep_agent(
    model=model,
    tools=[write_email, schedule_meeting, check_calendar_availability],
    system_prompt=EMAIL_ASSISTANT_INSTRUCTIONS,
    subagents=[triage_subagent],
    backend=memory_backend,
)

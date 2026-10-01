"""Incident investigation workflow.

When deployed:
  1. Reads the Slack alert message by link (via Slack connector)
  2. Summarises it with Mistral
  3. DMs the summary back to the triggering user

Input: a Slack message link (e.g. https://mistralai.slack.com/archives/C.../p...)
"""

from __future__ import annotations

import re

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)
from pydantic import BaseModel

slack_connector = connector("slack")

MY_SLACK_USER_ID = "U0A6AS9V407"


def _parse_slack_link(message_link: str) -> tuple[str, float]:
    match = re.search(r"/archives/([A-Z0-9]+)/p(\d+)", message_link)
    if not match:
        raise ValueError(f"Cannot parse Slack message link: {message_link!r}")
    channel_id = match.group(1)
    raw_ts = match.group(2)
    ts = float(f"{raw_ts[:-6]}.{raw_ts[-6:]}")
    return channel_id, ts


def _unwrap_text(response: object) -> str:
    if isinstance(response, dict):
        content = response.get("content") or []
    else:
        content = getattr(response, "content", None) or []
    if not content:
        return ""
    first = content[0]
    return first.get("text") if isinstance(first, dict) else getattr(first, "text", "") or ""


def _extract_llm_text(response: object) -> str:
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            (item.text if hasattr(item, "text") else item.get("text", ""))
            for item in content
        )
    return str(content or "")


class IncidentInput(BaseModel):
    message_link: str


@workflows.activity()
async def fetch_alert_message(
    message_link: str,
    slack: ToolCallClient = Depends(slack_connector),
) -> str:
    """Fetch the Slack alert message text by link."""
    channel_id, ts = _parse_slack_link(message_link)
    response = await slack.call_tool(
        tool_name="slack_read_channel",
        arguments={"channel_id": channel_id, "limit": 10, "oldest": str(ts - 5), "latest": str(ts + 60)},
    )
    return _unwrap_text(response)


@workflows.activity()
async def summarise_alert(alert_message: str) -> str:
    """Summarise the alert in a few concise lines."""
    prompt = f"""You are an on-call SRE. Summarise this Slack alert in 3–5 bullet points.
Be concise and focus on: what is firing, which service/cluster, and what to check first.

Alert:
{alert_message}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return _extract_llm_text(response)


@workflows.activity()
async def send_dm(
    text: str,
    slack: ToolCallClient = Depends(slack_connector),
) -> None:
    """Send a DM to the on-call engineer."""
    await slack.call_tool(
        tool_name="slack_send_message",
        arguments={"channel_id": "C0BRAB3A7LH", "message": text},
    )


@workflows.workflow.define(
    name="incident-investigate",
    workflow_display_name="Incident Investigation",
    workflow_description="Reads a Slack alert by link, summarises it, and DMs the summary back.",
    on_behalf_of=True,
)
@uses_connectors(slack_connector)
class IncidentInvestigateWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: IncidentInput) -> str:
        alert_message = await fetch_alert_message(input.message_link)
        summary = await summarise_alert(alert_message)
        await send_dm(summary)
        return summary

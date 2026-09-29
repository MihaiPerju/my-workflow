"""Workflow that reads recent messages from a Slack alert channel and summarizes them."""

from __future__ import annotations

import json

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

CHANNEL = "eng-alerts-apps"
MESSAGE_LIMIT = 50


class AlertSummaryInput(BaseModel):
    limit: int = MESSAGE_LIMIT


def _unwrap(response: object) -> dict | list:
    if isinstance(response, dict):
        content = response.get("content") or []
    else:
        content = getattr(response, "content", None) or []
    if not content:
        return {}
    first = content[0]
    text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
    if text is None:
        return {}
    return json.loads(text)


@workflows.activity()
async def fetch_channel_messages(
    limit: int,
    slack: ToolCallClient = Depends(slack_connector),
) -> list[dict]:
    """Fetch the most recent messages from #eng-alerts-apps."""
    payload = _unwrap(
        await slack.call_tool(
            tool_name="read_channel",
            arguments={"channel_name": CHANNEL, "limit": limit},
        )
    )
    messages = payload.get("messages", payload if isinstance(payload, list) else [])
    return [
        {
            "user": m.get("user") or m.get("username", "unknown"),
            "text": m.get("text", ""),
            "ts": m.get("ts", ""),
        }
        for m in messages
        if m.get("text")
    ]


@workflows.activity()
async def summarize_alerts(messages: list[dict]) -> str:
    """Use Mistral to summarize the alerts from the fetched messages."""
    if not messages:
        return "No messages found in #eng-alerts-apps."

    messages_text = "\n".join(
        f"[{m['ts']}] {m['user']}: {m['text']}" for m in messages
    )

    prompt = f"""You are an on-call engineer reading a Slack alert channel.
Analyze the following messages from #{CHANNEL} and provide a concise summary:

- Which alerts are currently firing or were recently triggered
- Which alerts have resolved
- Any patterns or recurring issues
- Priority of attention needed

Messages:
{messages_text}

Provide a short, actionable summary."""

    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-medium-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)

    try:
        return response.choices[0].message.content or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response from mistralai_chat_complete: {response!r}"
        ) from exc


@workflows.workflow.define(
    name="slack-alert-summary",
    workflow_display_name="Slack Alert Summary",
    workflow_description=f"Reads recent messages from #{CHANNEL} and summarizes firing alerts.",
    on_behalf_of=True,
)
@uses_connectors(slack_connector)
class SlackAlertSummaryWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: AlertSummaryInput) -> str:
        messages = await fetch_channel_messages(input.limit)
        return await summarize_alerts(messages)

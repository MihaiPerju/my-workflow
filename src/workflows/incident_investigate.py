"""Incident investigation workflow.

Pulls data from three sources in parallel:
  1. Slack  — recent messages from the alerts channel
  2. Grafana — currently firing alert rules
  3. Loki   — recent error logs

Then feeds everything to an LLM that produces a structured incident report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

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
grafana_connector = connector("grafana")

SLACK_CHANNEL_ID = "C0BRAB3A7LH"
SLACK_CHANNEL_NAME = "eng-alerts-apps"


class IncidentInput(BaseModel):
    time_window_minutes: int = 30
    service: str | None = None
    log_query: str | None = None


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
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}


def _unwrap_text(response: object) -> str:
    if isinstance(response, dict):
        content = response.get("content") or []
    else:
        content = getattr(response, "content", None) or []
    if not content:
        return ""
    first = content[0]
    return first.get("text") if isinstance(first, dict) else getattr(first, "text", "") or ""


@workflows.activity()
async def fetch_slack_alerts(
    slack: ToolCallClient = Depends(slack_connector),
) -> str:
    """Fetch recent messages from the Slack alert channel."""
    response = await slack.call_tool(
        tool_name="slack_read_channel",
        arguments={"channel_id": SLACK_CHANNEL_ID, "limit": 20},
    )
    return _unwrap_text(response)


@workflows.activity()
async def fetch_grafana_alerts(
    grafana: ToolCallClient = Depends(grafana_connector),
) -> str:
    """Fetch currently firing Grafana alert rules."""
    response = await grafana.call_tool(
        tool_name="grafana_list_alert_rules",
        arguments={"state": "firing"},
    )
    return _unwrap_text(response)


@workflows.activity()
async def fetch_loki_logs(
    time_window_minutes: int,
    service: str | None,
    log_query: str | None,
    grafana: ToolCallClient = Depends(grafana_connector),
) -> str:
    """Query Loki for recent errors within the time window."""
    now = datetime.now(timezone.utc)
    start_ns = int((now.timestamp() - time_window_minutes * 60) * 1e9)
    end_ns = int(now.timestamp() * 1e9)

    if log_query:
        query = log_query
    elif service:
        query = f'{{service="{service}"}} |= "error" | logfmt'
    else:
        query = '{job=~".+"} |= "error" | logfmt'

    response = await grafana.call_tool(
        tool_name="grafana_query_loki",
        arguments={
            "query": query,
            "start": str(start_ns),
            "end": str(end_ns),
            "limit": 50,
        },
    )
    return _unwrap_text(response)


@workflows.activity()
async def synthesize_incident(
    slack_messages: str,
    grafana_alerts: str,
    loki_logs: str,
    time_window_minutes: int,
    service: str | None,
) -> str:
    """Synthesize all data into a structured incident report."""
    service_context = f"focusing on service: {service}" if service else "across all services"

    prompt = f"""You are an on-call engineer investigating an incident.
You have {time_window_minutes} minutes of data {service_context}.

Analyze the following data sources and produce a structured incident report.

---
## Slack Alert Channel (#eng-alerts-apps)
{slack_messages or "No messages available."}

---
## Grafana Firing Alerts
{grafana_alerts or "No firing alerts."}

---
## Loki Error Logs (last {time_window_minutes} minutes)
{loki_logs or "No logs available."}

---

Produce a concise incident report with these sections:

### Current Status
One sentence: is there an active incident?

### What's Wrong
- Which services/components are affected
- What the symptoms are
- When it started (if determinable from the data)

### Likely Root Cause
Your best hypothesis based on the evidence.

### What Else to Investigate
Concrete next steps — specific metrics to check, queries to run, services to inspect.

### Severity
LOW / MEDIUM / HIGH / CRITICAL with a one-line justification.
"""

    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-large-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)

    try:
        return response.choices[0].message.content or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response: {response!r}") from exc


MY_SLACK_USER_ID = "U0A6AS9V407"


@workflows.activity()
async def send_dm_report(
    report: str,
    slack: ToolCallClient = Depends(slack_connector),
) -> None:
    """Send the incident report as a Slack DM."""
    await slack.call_tool(
        tool_name="slack_send_message",
        arguments={
            "channel_id": MY_SLACK_USER_ID,
            "text": report,
        },
    )


@workflows.workflow.define(
    name="incident-investigate",
    workflow_display_name="Incident Investigation",
    workflow_description="Pulls Slack alerts, Grafana firing rules, and Loki logs, then synthesizes a full incident report.",
    on_behalf_of=True,
)
@uses_connectors(slack_connector, grafana_connector)
class IncidentInvestigateWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: IncidentInput) -> str:
        slack_messages, grafana_alerts, loki_logs = await workflows.gather(
            fetch_slack_alerts(),
            fetch_grafana_alerts(),
            fetch_loki_logs(
                input.time_window_minutes,
                input.service,
                input.log_query,
            ),
        )

        report = await synthesize_incident(
            slack_messages,
            grafana_alerts,
            loki_logs,
            input.time_window_minutes,
            input.service,
        )

        await send_dm_report(report)
        return report

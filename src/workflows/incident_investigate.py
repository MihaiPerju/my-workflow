"""Incident investigation workflow.

When deployed (k8s):
  1. Reads the Slack alert message by link (via Slack connector)
  2. Summarises it with Mistral
  3. DMs the summary back via Slack

Locally:
  Pass the alert text directly as `alert_text` — Slack is not used.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.parse
from datetime import UTC

import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)
from pydantic import BaseModel

load_dotenv()

slack_connector = connector("slack")
grafana_connector = connector("grafana")

MY_SLACK_CHANNEL_ID = "C0C5URF1H99"
IS_LOCAL = not os.getenv("KUBERNETES_SERVICE_HOST")

GRAFANA_URL = os.getenv("GRAFANA_URL", "https://grafana-infra.cheetah-koi.ts.net")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN", "")
LOKI_DATASOURCE_UID = os.getenv("LOKI_DATASOURCE_UID", "")


def _grafana_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if GRAFANA_TOKEN:
        headers["Authorization"] = f"Bearer {GRAFANA_TOKEN}"
    return headers


async def _find_prometheus_uid() -> str | None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{GRAFANA_URL}/api/datasources", headers=_grafana_headers()
            )
            resp.raise_for_status()
            for ds in resp.json():
                if ds.get("type") in ("prometheus", "victoriametrics-datasource"):
                    return ds.get("uid")
    except Exception:
        return None
    return None


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
    return (
        first.get("text")
        if isinstance(first, dict)
        else getattr(first, "text", "") or ""
    )


def _extract_llm_text(response: object) -> str:
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            (item.text if hasattr(item, "text") else item.get("text", ""))
            for item in content
        )
    return str(content or "")


class IncidentInput(BaseModel):
    message_link: str = ""  # Required when deployed; unused locally
    alert_text: str = ""  # Pass directly when running locally


@workflows.activity()
async def fetch_alert_message(
    message_link: str,
    slack: ToolCallClient = Depends(slack_connector),
) -> str:
    """Fetch the Slack alert message text by link."""
    channel_id, ts = _parse_slack_link(message_link)
    response = await slack.call_tool(
        tool_name="slack_read_channel",
        arguments={
            "channel_id": channel_id,
            "limit": 10,
            "oldest": str(ts - 5),
            "latest": str(ts + 60),
        },
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
    """Send the summary to Slack."""
    await slack.call_tool(
        tool_name="slack_send_message",
        arguments={"channel_id": MY_SLACK_CHANNEL_ID, "message": text},
    )


@workflows.activity()
async def extract_service_name(alert_text: str) -> str:
    """Use LLM to extract the service/component name from the alert."""
    prompt = f"""Extract the service or component name from this alert.
Return only the service name as a short lowercase slug (e.g. "spaces-deploy", "billing-api").
If you cannot determine it, return "unknown".

Alert:
{alert_text}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return _extract_llm_text(response).strip().strip('"').lower()


@workflows.activity()
async def discover_metrics(service_name: str) -> dict:
    """Query Prometheus to find metric names and label names available for the service."""
    import httpx

    prom_uid = await _find_prometheus_uid()
    if not prom_uid:
        return {
            "error": "Prometheus datasource not found",
            "metric_names": [],
            "label_names": [],
        }

    base = f"{GRAFANA_URL}/api/datasources/proxy/uid/{prom_uid}/api/v1"
    headers = _grafana_headers()

    # 1. Metric names for this service
    metric_names: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{base}/label/__name__/values",
                headers=headers,
                params={"match[]": f'{{service="{service_name}"}}'},
            )
            if resp.status_code == 200:
                metric_names = resp.json().get("data", [])
    except Exception as e:
        metric_names = [f"error: {e}"]

    # 2. Label names for this service (so LLM knows what to filter on)
    label_names: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{base}/labels",
                headers=headers,
                params={"match[]": f'{{service="{service_name}"}}'},
            )
            if resp.status_code == 200:
                label_names = resp.json().get("data", [])
    except Exception as e:
        label_names = [f"error: {e}"]

    return {
        "service": service_name,
        "prom_uid": prom_uid or "",
        "metric_names": metric_names,
        "label_names": label_names,
    }


def _loki_explore_url(
    loki_uid: str, expr: str, alert_ts: float, now_ts: float, time_range: str = "now-1h"
) -> str:
    duration_s = _time_range_to_seconds(time_range)
    from_ms = int((alert_ts - duration_s) * 1000)
    to_ms = int(min(alert_ts + 900, now_ts) * 1000)
    payload = json.dumps(
        {
            "datasource": loki_uid,
            "queries": [{"refId": "A", "expr": expr}],
            "range": {"from": str(from_ms), "to": str(to_ms)},
        },
        separators=(",", ":"),
    )
    return f"{GRAFANA_URL}/explore?orgId=1&left={urllib.parse.quote(payload)}"


def _grafana_explore_url(
    prom_uid: str, expr: str, alert_ts: float, now_ts: float, time_range: str = "now-1h"
) -> str:
    duration_s = _time_range_to_seconds(time_range)
    from_ms = int((alert_ts - duration_s) * 1000)
    to_ms = int(min(alert_ts + 900, now_ts) * 1000)
    payload = json.dumps(
        {
            "datasource": prom_uid,
            "queries": [{"refId": "A", "expr": expr, "instant": False, "range": True}],
            "range": {"from": str(from_ms), "to": str(to_ms)},
        },
        separators=(",", ":"),
    )
    return f"{GRAFANA_URL}/explore?orgId=1&left={urllib.parse.quote(payload)}"


@workflows.activity()
async def generate_promql_queries(
    alert_text: str,
    service_name: str,
    cluster: str,
    label_names: list[str],
    metric_names: list[str],
) -> list[dict]:
    """Use LLM to produce targeted PromQL queries for investigating the incident."""
    labels_str = ", ".join(label_names[:40])
    metrics_str = ", ".join(metric_names[:50]) if metric_names else "none found"
    cluster_filter = f'cluster="{cluster}", ' if cluster else ""

    prompt = f"""You are an SRE investigating a production incident. Generate 6-8 targeted PromQL queries to investigate this alert.

Alert:
{alert_text}

Service slug: {service_name}
Available Prometheus label names: {labels_str or "(discovery failed — use best-guess labels)"}
Known metric names for this service: {metrics_str}

## Label and metric conventions for this stack

**Primary filter**: always use `namespace="{service_name}"` (NOT `service="{service_name}"`).
Also include `{cluster_filter}` in every query.

**HTTP metrics** (FastAPI services):
- `fastapi_responses_total{{{cluster_filter}namespace="{service_name}", service=~"{service_name}.*", status_code=~"5.."}}` — response counts by status code
- `fastapi_requests_duration_milliseconds_bucket{{{cluster_filter}namespace="{service_name}", service=~"{service_name}.*"}}` — latency histogram (use histogram_quantile)
- Status label is `status_code`, values like `"200"`, `"500"`, use `status_code=~"5.."` for 5xx

**Kubernetes availability**:
- `kube_deployment_status_replicas_available{{{cluster_filter}namespace="{service_name}", deployment="{service_name}-app"}}` — available replicas
- `kube_pod_container_status_restarts_total{{{cluster_filter}namespace="{service_name}", pod=~"{service_name}-app-.+"}}` — pod restarts

**Container resources**:
- `container_memory_working_set_bytes{{{cluster_filter}namespace="{service_name}", pod=~"{service_name}-app-.+", container!=""}}` — memory
- `container_cpu_usage_seconds_total{{{cluster_filter}namespace="{service_name}", pod=~"{service_name}-app-.+", container!=""}}` — CPU

**Service-specific custom metrics** (if known metric names include them):
- Prefer exact metric names from "Known metric names" above when available.

For each query return a JSON object with:
- "title": short name (e.g. "5xx Error Rate by Endpoint")
- "explanation": 1-2 sentences on what this shows and why it matters for this incident
- "expr": valid PromQL expression using the conventions above
- "time_range": one of "now-15m", "now-1h", "now-3h" — pick the window that best surfaces this signal

Cover: 5xx error rate, latency percentiles, traffic volume, pod availability, pod restarts, upstream dependency health.
Return ONLY a JSON array, no markdown fences or commentary.
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return _parse_json_array(_extract_llm_text(response))


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, tolerating preambles and fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Try to pull out the first [...] block (handles "Sure, here are: [...]")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def _time_range_to_seconds(time_range: str) -> int:
    match = re.match(r"now-(\d+)([mhd])", time_range)
    if not match:
        return 3600
    val, unit = int(match.group(1)), match.group(2)
    return val * {"m": 60, "h": 3600, "d": 86400}[unit]


@workflows.activity()
async def get_current_time() -> float:
    return time.time()


def _extract_cluster(alert_text: str) -> str:
    """Extract cluster name from alert text (e.g. 'prod-swedencentral-1')."""
    m = re.search(
        r'cluster[=\s:]+["\']?([a-z0-9][a-z0-9-]+)["\']?', alert_text, re.IGNORECASE
    )
    return m.group(1) if m else ""


def _extract_alert_time(alert_text: str, now_ts: float) -> float:
    """Best-effort: parse a Unix/ISO timestamp from the alert text; fall back to now."""
    # Unix timestamp embedded (e.g. Slack p-link style)
    m = re.search(r"\b(17\d{8})\b", alert_text)
    if m:
        return float(m.group(1))
    # ISO-ish datetime
    m = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)", alert_text)
    if m:
        from datetime import datetime

        try:
            dt = datetime.fromisoformat(m.group(1).replace(" ", "T"))
            return dt.replace(tzinfo=UTC).timestamp()
        except ValueError:
            pass
    return now_ts


@workflows.activity()
async def execute_and_summarise_query(
    query: dict, prom_uid: str, alert_ts: float
) -> dict:
    """Run one PromQL query against Prometheus and summarise the result with LLM."""
    import httpx

    expr = query.get("expr", "")
    time_range = query.get("time_range", "now-1h")
    duration_s = _time_range_to_seconds(time_range)
    # Centre the window on the alert: look back duration_s before it, forward 15 min after
    end = min(alert_ts + 900, time.time())
    start = alert_ts - duration_s
    step = max(15, duration_s // 100)

    base = f"{GRAFANA_URL}/api/datasources/proxy/uid/{prom_uid}/api/v1"
    headers = _grafana_headers()

    prom_result = None
    fetch_error = None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{base}/query_range",
                headers=headers,
                params={
                    "query": expr,
                    "start": str(int(start)),
                    "end": str(int(end)),
                    "step": str(step),
                },
            )
            if resp.status_code == 200:
                prom_result = resp.json().get("data", {})
            else:
                fetch_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        fetch_error = str(e)

    has_data = bool(prom_result and prom_result.get("result"))
    if not has_data:
        return {**query, "result_summary": "No data."}

    result_str = json.dumps(prom_result, separators=(",", ":"))
    if len(result_str) > 4000:
        result_str = result_str[:4000] + "... [truncated]"

    prompt = f"""You are an SRE investigating a production incident. You have the Prometheus result below.
In 1-2 sentences: what was observed, what it points to, and what to investigate next.
Do not describe the data format. Focus on the signal.

Query: {query.get("title", "")}
PromQL: {expr}

Result:
{result_str}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return {**query, "result_summary": _extract_llm_text(response).strip()}


@workflows.activity()
async def generate_loki_queries(
    alert_text: str,
    service_name: str,
    cluster: str,
    loki_uid: str,
) -> list[dict]:
    """Use LLM to generate targeted LogQL queries for investigating the incident."""
    cluster_sel = f', cluster="{cluster}"' if cluster else ""

    prompt = f"""You are an SRE investigating a production incident. Generate 4-6 targeted LogQL queries to investigate this alert.

Alert:
{alert_text}

Service: {service_name}
Cluster: {cluster or "(unknown)"}

## Label and log conventions for this Loki setup

**Stream selector**: always use `{{namespace="{service_name}"{cluster_sel}}}`.
Optionally add `container="main"` to narrow to the app container.

**Log format**: logs are JSON objects. Use `| json` to parse them, then filter on fields.
- HTTP status codes are in the JSON body as `status_code` (e.g. `500`, `"500"`).
  Filter with `|= "\\"status_code\\":5"` (NOT as a stream label `status_code=~"5.."`).
- Events are under `event` field.
- Logger/level under `level` or `detected_level` stream label (values: `error`, `warning`, `info`, `debug`).

**Useful line filter patterns**:
- 5xx errors: `|= "\\"status_code\\":5"`
- Errors by level: `{{namespace="{service_name}"{cluster_sel}, detected_level="error"}}`
- Upstream failures: `|= "Failed"` or `|= "exception"`
- OOM: `|= "OOM"` or `|= "killed"`

For each query return a JSON object with:
- "title": short name (e.g. "5xx HTTP Errors")
- "explanation": 1-2 sentences on what this shows and why it matters for this incident
- "expr": valid LogQL expression using the conventions above
- "time_range": one of "now-15m", "now-1h", "now-3h"

Cover: 5xx HTTP errors (use JSON body filter), error-level logs, upstream/remote failures, pod crashes/OOM.
Return ONLY a JSON array, no markdown fences or commentary.
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    raw = _extract_llm_text(response)
    result = _parse_json_array(raw)
    if not result:
        # Fallback: generate minimal hardcoded queries so the log section is never empty
        result = [
            {
                "title": "5xx HTTP Errors",
                "explanation": "Filters log lines where status_code is 5xx using JSON body match.",
                "expr": f'{{namespace="{service_name}"{cluster_sel}}} |= "\\"status_code\\":5"',
                "time_range": "now-1h",
            },
            {
                "title": "Error-Level Logs",
                "explanation": "Streams with detected_level=error to find exceptions and failures.",
                "expr": f'{{namespace="{service_name}"{cluster_sel}, detected_level="error"}}',
                "time_range": "now-1h",
            },
            {
                "title": "Upstream / Remote Failures",
                "explanation": "Lines containing 'Failed' to catch upstream dependency errors.",
                "expr": f'{{namespace="{service_name}"{cluster_sel}}} |= "Failed"',
                "time_range": "now-1h",
            },
        ]
    return result


@workflows.activity()
async def execute_and_summarise_log_query(
    query: dict, loki_uid: str, alert_ts: float
) -> dict:
    """Run one LogQL query against Loki and summarise the result with LLM."""
    import httpx

    expr = query.get("expr", "")
    time_range = query.get("time_range", "now-1h")
    duration_s = _time_range_to_seconds(time_range)
    end = int(min(alert_ts + 900, time.time()) * 1_000_000_000)
    start = int((alert_ts - duration_s) * 1_000_000_000)

    base = f"{GRAFANA_URL}/api/datasources/proxy/uid/{loki_uid}/loki/api/v1"
    headers = _grafana_headers()

    log_lines: list[str] = []
    fetch_error: str | None = None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{base}/query_range",
                headers=headers,
                params={
                    "query": expr,
                    "start": str(start),
                    "end": str(end),
                    "limit": 100,
                    "direction": "backward",
                },
            )
            if resp.status_code == 200:
                for stream in resp.json().get("data", {}).get("result", []):
                    for _ts, line in stream.get("values", []):
                        log_lines.append(line)
            else:
                fetch_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        fetch_error = str(e)

    if not log_lines:
        return {**query, "result_summary": "No data."}

    sample = "\n".join(log_lines[:60])
    if len(sample) > 4000:
        sample = sample[:4000] + "\n... [truncated]"

    prompt = f"""You are an SRE investigating a production incident. You have the log lines below.
In 1-2 sentences: what was observed, what it points to, and what to investigate next.
Do not describe the log format. Focus on the signal.

Query: {query.get("title", "")}
LogQL: {expr}

Logs ({len(log_lines)} lines):
{sample}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return {**query, "result_summary": _extract_llm_text(response).strip()}


def _parse_datasources(raw: str) -> tuple[str, str]:
    """Return (prom_uid, loki_uid) from a Grafana list_datasources response.

    Handles:
    - JSON array of datasource objects
    - JSON object with a "datasources" or "data" key wrapping the array
    - Newline-separated text with uid/type fields
    """
    prom_uid = ""
    loki_uid = ""

    if not raw:
        return prom_uid, loki_uid

    # Try JSON
    try:
        parsed = json.loads(raw)
        # Could be a list directly or wrapped in a key
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("datasources") or parsed.get("data") or []
            # Some implementations return a single datasource object
            if not items and "uid" in parsed:
                items = [parsed]
        else:
            items = []

        for ds in items:
            if not isinstance(ds, dict):
                continue
            t = ds.get("type", "")
            uid = ds.get("uid", "") or ds.get("UID", "")
            if t in ("prometheus", "victoriametrics-datasource") and not prom_uid:
                prom_uid = uid
            if t == "loki" and not loki_uid:
                loki_uid = uid
        return prom_uid, loki_uid
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: text parsing — look for uid/type pairs
    # e.g. "uid: abc123\ntype: prometheus"
    current: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip().lstrip("- ")
        if ":" in line:
            k, _, v = line.partition(":")
            current[k.strip().lower()] = v.strip().strip('"')
        if not line and current:
            t = current.get("type", "")
            uid = current.get("uid", "")
            if t in ("prometheus", "victoriametrics-datasource") and not prom_uid:
                prom_uid = uid
            if t == "loki" and not loki_uid:
                loki_uid = uid
            current = {}
    # flush last entry
    t = current.get("type", "")
    uid = current.get("uid", "")
    if t in ("prometheus", "victoriametrics-datasource") and not prom_uid:
        prom_uid = uid
    if t == "loki" and not loki_uid:
        loki_uid = uid

    return prom_uid, loki_uid


@workflows.activity()
async def discover_metrics_via_connector(
    service_name: str,
    grafana: ToolCallClient = Depends(grafana_connector),
) -> dict:
    """Discover Prometheus metrics and labels using the Grafana connector."""
    ds_resp = await grafana.call_tool("list_datasources", {})
    raw = _unwrap_text(ds_resp)
    prom_uid, loki_uid_discovered = _parse_datasources(raw)

    if not prom_uid:
        return {
            "service": service_name,
            "prom_uid": "",
            "loki_uid": loki_uid_discovered,
            "metric_names": [],
            "label_names": [],
            "_raw_datasources": raw[:500],
        }

    return {
        "service": service_name,
        "prom_uid": prom_uid,
        "loki_uid": loki_uid_discovered,
        "metric_names": [],
        "label_names": [],
    }


@workflows.activity()
async def execute_and_summarise_query_via_connector(
    query: dict,
    prom_uid: str,
    alert_ts: float,
    now_ts: float,
    grafana: ToolCallClient = Depends(grafana_connector),
) -> dict:
    """Execute a PromQL query via the Grafana connector and summarise with LLM."""
    from datetime import datetime

    expr = query.get("expr", "")
    time_range = query.get("time_range", "now-1h")
    duration_s = _time_range_to_seconds(time_range)
    start_ts = alert_ts - duration_s
    end_ts = min(alert_ts + 900, now_ts)
    step = max(15, duration_s // 100)

    def _ts_to_rfc3339(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    has_data = False
    result_str = ""
    try:
        resp = await grafana.call_tool(
            "query_metrics_range",
            {
                "datasource_uid": prom_uid,
                "expr": expr,
                "start_rfc3339": _ts_to_rfc3339(start_ts),
                "end_rfc3339": _ts_to_rfc3339(end_ts),
                "step_seconds": step,
            },
        )
        raw = _unwrap_text(resp)
        data = json.loads(raw) if raw else {}
        result = data.get("result", data.get("data", {}).get("result", []))
        has_data = bool(result)
        result_str = json.dumps(result, separators=(",", ":"))[:4000]
    except Exception as e:
        result_str = f"error: {e}"

    if not has_data:
        return {**query, "result_summary": "No data."}

    prompt = f"""You are an SRE investigating a production incident. You have the Prometheus result below.
In 1-2 sentences: what was observed, what it points to, and what to investigate next.
Do not describe the data format. Focus on the signal.

Query: {query.get("title", "")}
PromQL: {expr}

Result:
{result_str}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return {**query, "result_summary": _extract_llm_text(response).strip()}


@workflows.activity()
async def execute_and_summarise_log_query_via_connector(
    query: dict,
    loki_uid: str,
    alert_ts: float,
    now_ts: float,
    grafana: ToolCallClient = Depends(grafana_connector),
) -> dict:
    """Execute a LogQL query via the Grafana connector and summarise with LLM."""
    from datetime import datetime

    expr = query.get("expr", "")
    time_range = query.get("time_range", "now-1h")
    duration_s = _time_range_to_seconds(time_range)
    start_ts = alert_ts - duration_s
    end_ts = min(alert_ts + 900, now_ts)

    def _ts_to_rfc3339(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_lines: list[str] = []
    try:
        resp = await grafana.call_tool(
            "query_logs_range",
            {
                "datasource_uid": loki_uid,
                "query": expr,
                "start_rfc3339": _ts_to_rfc3339(start_ts),
                "end_rfc3339": _ts_to_rfc3339(end_ts),
                "limit": 100,
                "direction": "backward",
            },
        )
        raw = _unwrap_text(resp)
        data = json.loads(raw) if raw else {}
        for stream in data.get("result", data.get("data", {}).get("result", [])):
            for _ts, line in stream.get("values", []):
                log_lines.append(line)
    except Exception:
        pass

    if not log_lines:
        return {**query, "result_summary": "No data."}

    sample = "\n".join(log_lines[:60])[:4000]
    prompt = f"""You are an SRE investigating a production incident. You have the log lines below.
In 1-2 sentences: what was observed, what it points to, and what to investigate next.
Do not describe the log format. Focus on the signal.

Query: {query.get("title", "")}
LogQL: {expr}

Logs ({len(log_lines)} lines):
{sample}
"""
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    return {**query, "result_summary": _extract_llm_text(response).strip()}


@workflows.workflow.define(
    name="incident-investigate",
    workflow_display_name="Incident Investigation",
    workflow_description="Reads a Slack alert by link, summarises it, and DMs the summary back.",
    on_behalf_of=True,
)
@uses_connectors(slack_connector, grafana_connector)
class IncidentInvestigateWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: IncidentInput) -> str:
        use_slack = bool(input.message_link)
        if use_slack:
            alert_message = await fetch_alert_message(input.message_link)
        else:
            alert_message = input.alert_text

        summary = await summarise_alert(alert_message)
        service_name = await extract_service_name(alert_message)
        metrics = await (
            discover_metrics_via_connector(service_name)
            if use_slack
            else discover_metrics(service_name)
        )

        prom_uid = metrics.get("prom_uid", "")
        loki_uid = metrics.get("loki_uid", "") if use_slack else LOKI_DATASOURCE_UID
        now_ts = await get_current_time()
        # Prefer the timestamp embedded in the Slack message link (reliable);
        # fall back to parsing the alert text or now.
        if use_slack:
            _, alert_ts = _parse_slack_link(input.message_link)
        else:
            alert_ts = _extract_alert_time(alert_message, now_ts)

        cluster = _extract_cluster(alert_message)

        # Generate PromQL and LogQL query lists — run sequentially to avoid
        # asyncio.gather + Temporal activity edge cases
        promql_raw = await generate_promql_queries(
            alert_text=alert_message,
            service_name=service_name,
            cluster=cluster,
            label_names=metrics.get("label_names", []),
            metric_names=metrics.get("metric_names", []),
        )
        if loki_uid:
            try:
                logql_raw = await generate_loki_queries(
                    alert_text=alert_message,
                    service_name=service_name,
                    cluster=cluster,
                    loki_uid=loki_uid,
                )
            except Exception:
                logql_raw = []
        else:
            logql_raw = []

        promql_with_links = [
            {
                "title": q.get("title", ""),
                "explanation": q.get("explanation", ""),
                "expr": q.get("expr", ""),
                "time_range": q.get("time_range", "now-1h"),
                "grafana_url": _grafana_explore_url(
                    prom_uid, q["expr"], alert_ts, now_ts, q.get("time_range", "now-1h")
                )
                if prom_uid
                else "",
            }
            for q in promql_raw
        ]
        logql_with_links = [
            {
                "title": q.get("title", ""),
                "explanation": q.get("explanation", ""),
                "expr": q.get("expr", ""),
                "time_range": q.get("time_range", "now-1h"),
                "grafana_url": _loki_explore_url(
                    loki_uid, q["expr"], alert_ts, now_ts, q.get("time_range", "now-1h")
                )
                if loki_uid
                else "",
            }
            for q in logql_raw
        ]

        # Execute and summarise all queries in parallel
        if use_slack and prom_uid:
            prom_tasks = [
                execute_and_summarise_query_via_connector(q, prom_uid, alert_ts, now_ts)
                for q in promql_with_links
            ]
        elif prom_uid:
            prom_tasks = [
                execute_and_summarise_query(q, prom_uid, alert_ts)
                for q in promql_with_links
            ]
        else:
            prom_tasks = []

        if use_slack and loki_uid:
            loki_tasks = [
                execute_and_summarise_log_query_via_connector(
                    q, loki_uid, alert_ts, now_ts
                )
                for q in logql_with_links
            ]
        elif loki_uid:
            loki_tasks = [
                execute_and_summarise_log_query(q, loki_uid, alert_ts)
                for q in logql_with_links
            ]
        else:
            loki_tasks = []
        all_tasks = prom_tasks + loki_tasks
        if all_tasks:
            results = list(await asyncio.gather(*all_tasks))
            queries = results[: len(prom_tasks)] if prom_tasks else promql_with_links
            log_queries = results[len(prom_tasks) :] if loki_tasks else logql_with_links
        else:
            queries = promql_with_links
            log_queries = logql_with_links

        if use_slack:
            # --- chunk 1: header + summary ---
            summary_lines = [f"*Incident: {service_name}*\n", "*Summary*"]
            for b in summary.split("\n"):
                b = b.strip().lstrip("•- ")
                if b:
                    summary_lines.append(f"• {b}")
            await send_dm("\n".join(summary_lines))

            # --- chunk 2: metrics ---
            metric_lines = ["*Metrics*"]
            for q in queries:
                finding = q.get("result_summary", "No data.")
                url = q.get("grafana_url", "")
                link = f" (<{url}|graph>)" if url else ""
                metric_lines.append(f"*{q['title']}*{link}\n{finding}")
            await send_dm("\n".join(metric_lines))

            # --- chunk 3: logs ---
            log_lines = ["*Logs*"]
            for q in log_queries:
                finding = q.get("result_summary", "No data.")
                url = q.get("grafana_url", "")
                link = f" (<{url}|logs>)" if url else ""
                log_lines.append(f"*{q['title']}*{link}\n{finding}")
            await send_dm("\n".join(log_lines))

        return json.dumps(
            {
                "summary": summary,
                "service": service_name,
                "metrics": metrics,
                "queries": queries,
                "log_queries": log_queries,
            },
            indent=2,
        )

# src/agent_monitor.py
# CLI:
#   uv run python -m src.agent_monitor \
#     --metrics data/metrics_history.jsonl \
#     --drift data/drift_latest.json \
#     --out artifacts/agent_plan.yaml 

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from langgraph.graph import StateGraph, END

# Vertex AI (Gemini on Vertex)
from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel


# -------------------------- Utilities --------------------------

def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                pass
    return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _latest_k(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return rows[-k:] if len(rows) > k else rows


def _time_sort(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(r):
        ts = r.get("timestamp") or r.get("time") or r.get("ts")
        dt = _parse_ts(ts) if ts else None
        return dt or datetime.min
    return sorted(rows, key=_key)


def _init_vertex(project: Optional[str], location: Optional[str], default_region: str = "us-central1"):
    vertexai_init(
        project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=location or os.getenv("GOOGLE_CLOUD_REGION") or default_region,
    )


# ---------------- Structured output helpers ----------------

def _gen_json(model: GenerativeModel, prompt: str, schema: Optional[dict] = None) -> dict:
    """
    Ask Vertex Gemini to return strict JSON (no markdown fences).
    If available, pass response_schema to enforce shape (newer SDKs).
    """
    gen_cfg = {"response_mime_type": "application/json"}
    if schema:
        gen_cfg["response_schema"] = schema  # if unsupported by SDK, it may be ignored/raise
    resp = model.generate_content(prompt, generation_config=gen_cfg)

    text = getattr(resp, "text", None)
    if not text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") for p in parts)
    s = (text or "").strip()
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON.\n---RAW---\n{s}") from e


# JSON Schemas for sub-agents
METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "roc_auc_median_7d": {"type": "number"},
        "pr_auc_median_7d": {"type": "number"},
        "roc_auc_latest": {"type": "number"},
        "pr_auc_latest": {"type": "number"},
        "roc_auc_drop_pct": {"type": "number"},
        "pr_auc_drop_pct": {"type": "number"},
        "latency_recent_two_over_400": {"type": "boolean"},
        "latest_latency_p95_ms": {"type": ["integer", "null"]},
    },
    "required": [
        "roc_auc_median_7d", "pr_auc_median_7d",
        "roc_auc_latest", "pr_auc_latest",
        "roc_auc_drop_pct", "pr_auc_drop_pct",
        "latency_recent_two_over_400", "latest_latency_p95_ms"
    ],
    "additionalProperties": False,
}

DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_drift": {"type": "boolean"},
        "top_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "psi": {"type": "number"},
                },
                "required": ["name", "psi"],
                "additionalProperties": False,
            },
            "maxItems": 5
        },
    },
    "required": ["overall_drift", "top_features"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["healthy", "warn", "critical"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 1,
                "additionalProperties": {"type": ["number", "boolean", "integer"]}
            }
        },
        "actions": {
            "type": "array",
            "items": {"type": "string", "enum": [
                "open_incident", "trigger_retraining", "roll_back_model",
                "raise_thresholds", "page_oncall=false", "do_nothing"
            ]},
            "minItems": 1
        },
        "rationale": {"type": "string"},
    },
    "required": ["status", "findings", "actions", "rationale"],
    "additionalProperties": False,
}


# -------------------------- Agent State --------------------------

@dataclass
class AgentState:
    metrics_path: str
    drift_path: str
    out_path: str
    project: Optional[str] = None
    location: Optional[str] = None
    model_name: str = "gemini-1.5-flash-002"

    # Inputs
    history: List[Dict[str, Any]] = field(default_factory=list)
    drift: Dict[str, Any] = field(default_factory=dict)

    # LLM Sub-agent outputs
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    drift_summary: Dict[str, Any] = field(default_factory=dict)
    decision_json: Dict[str, Any] = field(default_factory=dict)

    # Final output
    yaml_text: str = ""


# -------------------------- Nodes --------------------------

def observe(state: AgentState) -> AgentState:
    history = _read_jsonl(state.metrics_path)
    if not history:
        raise RuntimeError(f"No metrics found in {state.metrics_path}")
    state.history = _time_sort(history)
    state.drift = _read_json(state.drift_path)
    return state


def metrics_analyst(state: AgentState) -> AgentState:
    """
    LLM sub-agent: compute derived metrics from last <=10 rows.
    """
    _init_vertex(state.project, state.location)
    model = GenerativeModel(state.model_name)
    metrics_block = json.dumps(_latest_k(state.history, 10), indent=2)

    prompt = f"""
            You are a metrics analysis agent. Input is an array of JSON metric points
            with fields: timestamp/time/ts, roc_auc, pr_auc, latency_p95_ms (some may be missing).

            Compute from the LAST <=10 entries:
            - roc_auc_median_7d, pr_auc_median_7d: if daily timestamps missing, median over provided window.
            - roc_auc_latest, pr_auc_latest
            - roc_auc_drop_pct, pr_auc_drop_pct: percent drop of latest vs those medians (one decimal)
            - latency_recent_two_over_400: true iff last TWO latency_p95_ms > 400
            - latest_latency_p95_ms: latest latency value (integer), or null if not present

            Return STRICT JSON exactly matching the provided schema.

            METRICS (JSON array):
            {metrics_block}
            """.strip()

    state.metrics_summary = _gen_json(model, prompt, METRICS_SCHEMA)
    return state


def drift_analyst(state: AgentState) -> AgentState:
    """
        LLM sub-agent: summarize drift (overall flag + top features by PSI).
    """
    _init_vertex(state.project, state.location)
    model = GenerativeModel(state.model_name)
    drift_block = json.dumps(state.drift, indent=2)

    prompt = f"""
        You are a drift analysis agent. Input is a drift report JSON.

        Produce strict JSON:
        - overall_drift: copy boolean from input.
        - top_features: up to 5 items with largest PSI values (descending).
        Each item is {{"name": <string>, "psi": <number rounded to two decimals>}}.

        Return STRICT JSON exactly matching the provided schema.

        DRIFT REPORT:
        {drift_block}
        """.strip()

    state.drift_summary = _gen_json(model, prompt, DRIFT_SCHEMA)
    return state


def decision_maker(state: AgentState) -> AgentState:
    """
    LLM sub-agent: apply heuristics and produce decision JSON
    (status, findings, actions, rationale).
    """
    _init_vertex(state.project, state.location)
    model = GenerativeModel(state.model_name)

    metrics_json = json.dumps(state.metrics_summary, indent=2)
    drift_json = json.dumps(state.drift_summary, indent=2)

    prompt = f"""
            You are a decision agent. Apply these heuristics:

            - warn if ROC-AUC drops ≥ 3% vs 7-day median OR p95 latency > 400ms for 2 consecutive points.
            - critical if ROC-AUC drops ≥ 6% OR (overall_drift == true AND PR-AUC down ≥ 5% vs 7-day median).
            - healthy otherwise.

            Allowed actions: ["open_incident","trigger_retraining","roll_back_model","raise_thresholds","page_oncall=false","do_nothing"].

            INPUT METRICS_SUMMARY:
            {metrics_json}

            INPUT DRIFT_SUMMARY:
            {drift_json}

            Return STRICT JSON with keys: status, findings, actions, rationale, matching the schema.
            Rules:
            - findings includes ONLY triggered numeric signals plus drift_overall (always).
            * Include {{"roc_auc_drop_pct": <one decimal>}} ONLY if relevant.
            * Include {{"latency_p95_ms": <integer>}} ONLY if two most recent p95 > 400ms.
            * ALWAYS include {{"drift_overall": <true|false>}}.
            - actions consistent with status.
            - rationale: 2–4 sentences.

            Return JSON only.
            """.strip()

    state.decision_json = _gen_json(model, prompt, DECISION_SCHEMA)

    # Minimal local checks
    if state.decision_json["status"] not in ("healthy", "warn", "critical"):
        raise RuntimeError("Invalid status from decision_maker.")
    if not isinstance(state.decision_json["findings"], list):
        raise RuntimeError("decision_maker findings must be a list")
    if not any(isinstance(x, dict) and "drift_overall" in x for x in state.decision_json["findings"]):
        raise RuntimeError("decision_maker findings must include drift_overall")
    return state


def yaml_writer(state: AgentState) -> AgentState:
    """
    Deterministically convert decision JSON to the exact YAML required.
    """
    d = state.decision_json
    lines = []
    lines.append(f"status: {d['status']}")
    lines.append("findings:")
    for item in d["findings"]:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        k, v = next(iter(item.items()))
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        else:
            v_str = f"{v}"
        lines.append(f"  - {k}: {v_str}")
    lines.append("actions:")
    for a in d["actions"]:
        lines.append(f"  - {a}")
    lines.append("rationale: >")
    rationale = str(d["rationale"]).strip().replace("\n", "\n  ")
    lines.append(f"  {rationale}")
    state.yaml_text = "\n".join(lines) + "\n"
    return state


def write_to_disk(state: AgentState) -> AgentState:
    os.makedirs(os.path.dirname(state.out_path) or ".", exist_ok=True)
    with open(state.out_path, "w") as f:
        f.write(state.yaml_text)
    print(state.yaml_text.strip())
    return state


# -------------------------- Orchestration --------------------------

def build_runner():
    graph = StateGraph(AgentState)
    graph.add_node("observe", observe)
    graph.add_node("metrics_analyst", metrics_analyst)
    graph.add_node("drift_analyst", drift_analyst)
    graph.add_node("decision_maker", decision_maker)
    graph.add_node("yaml_writer", yaml_writer)
    graph.add_node("write", write_to_disk)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "metrics_analyst")
    graph.add_edge("metrics_analyst", "drift_analyst")
    graph.add_edge("drift_analyst", "decision_maker")
    graph.add_edge("decision_maker", "yaml_writer")
    graph.add_edge("yaml_writer", "write")
    graph.add_edge("write", END)

    app = graph.compile()

    def _runner(state: AgentState) -> AgentState:
        return app.invoke(state)

    return _runner


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics_history.jsonl")
    ap.add_argument("--drift", required=True, help="drift_latest.json")
    ap.add_argument("--out", required=True, help="Output YAML path")
    ap.add_argument("--project", help="GCP Project ID (falls back to ADC project)")
    ap.add_argument("--location", default="us-central1", help="GCP Region (e.g., us-central1)")
    ap.add_argument("--model", default="gemini-1.5-flash-002", help="Vertex Gemini model name")
    args = ap.parse_args()

    state = AgentState(
        metrics_path=args.metrics,
        drift_path=args.drift,
        out_path=args.out,
        project=args.project,
        location=args.location,
        model_name=args.model,
    )

    runner = build_runner()
    final_state = runner(state)

    # Non-zero exit for critical to trip CI alarms (optional)
    try:
        parsed = yaml.safe_load(final_state.yaml_text) or {}
        if parsed.get("status") == "critical":
            sys.exit(2)
    except Exception:
        pass


if __name__ == "__main__":
    main()

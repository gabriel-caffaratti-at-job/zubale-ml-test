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
from vertexai.generative_models import GenerativeModel, GenerationConfig


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

def _gen_json(model: GenerativeModel, prompt: str) -> dict:
    """
    Ask Vertex Gemini to return strict JSON (no markdown fences) by MIME type.
    NOTE: We intentionally DO NOT pass response_schema due to SDK enum/type issues.
    """
    gen_cfg = GenerationConfig(response_mime_type="application/json")
    resp = model.generate_content(prompt, generation_config=gen_cfg)

    text = getattr(resp, "text", None)
    if not text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") for p in parts)
    s = (text or "").strip()
    try:
        return json.loads(s)
    except Exception as e:
        # Helpful during iteration; won’t break CI if stderr is ignored
        print("DEBUG: RAW_JSON_FROM_MODEL:\n", s, file=sys.stderr)
        raise RuntimeError("Model did not return valid JSON.") from e


def _normalize_decision(decision: Dict[str, Any], drift_overall: bool) -> Dict[str, Any]:
    """
    Coerce model output to required shape:
    - findings: list of 1-key dicts, always includes drift_overall once
    - actions: list of strings
    - status: in {"healthy","warn","critical"}
    - rationale: string
    """
    d = dict(decision or {})

    # findings -> list[dict[str, Any]]
    f = d.get("findings", [])
    if f is None:
        f = []
    elif isinstance(f, dict):
        # Convert {"a":1,"b":2} -> [{"a":1},{"b":2}]
        f = [{k: v} for k, v in f.items()]
    elif not isinstance(f, list):
        f = [f]

    fixed: List[Dict[str, Any]] = []
    for item in f:
        if isinstance(item, dict) and len(item) == 1:
            fixed.append(item)
        elif isinstance(item, dict) and len(item) > 1:
            # Split multi-key dicts
            fixed.extend([{k: v} for k, v in item.items()])
        else:
            # Skip non-dict junk
            continue

    # Ensure drift_overall is present exactly once and boolean-typed
    saw = False
    for i, item in enumerate(fixed):
        if "drift_overall" in item:
            fixed[i] = {"drift_overall": bool(item["drift_overall"])}
            saw = True
            break
    if not saw:
        fixed.insert(0, {"drift_overall": bool(drift_overall)})

    # actions -> list[str]
    actions = d.get("actions", [])
    if actions is None:
        actions = []
    elif isinstance(actions, str):
        actions = [actions]
    elif not isinstance(actions, list):
        actions = [str(actions)]
    actions = [str(a) for a in actions if a]

    # status -> allowed set
    status = (d.get("status") or "").strip().lower()
    if status not in ("healthy", "warn", "critical"):
        status = "healthy"

    # rationale -> string
    rationale = str(d.get("rationale") or "").strip()

    # Fallback action defaults
    if not actions:
        actions = ["do_nothing"] if status == "healthy" else ["page_oncall=false"]

    return {
        "status": status,
        "findings": fixed,
        "actions": actions,
        "rationale": rationale or "No rationale provided by the model; applied defaults.",
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

            Return STRICT JSON ONLY with keys exactly:
            {{
            "roc_auc_median_7d": <number>,
            "pr_auc_median_7d": <number>,
            "roc_auc_latest": <number>,
            "pr_auc_latest": <number>,
            "roc_auc_drop_pct": <number>,
            "pr_auc_drop_pct": <number>,
            "latency_recent_two_over_400": <true|false>,
            "latest_latency_p95_ms": <integer|null>
            }}

            METRICS (JSON array):
            {metrics_block}
            """.strip()

    state.metrics_summary = _gen_json(model, prompt)
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

            Produce STRICT JSON ONLY:
            {{
            "overall_drift": <true|false>,
            "top_features": [ {{"name": <string>, "psi": <number>}}, ... up to 5, descending by psi ]
            }}

            - overall_drift: copy boolean from input if present; otherwise infer.
            - top_features: pick up to 5 with largest PSI values; round to two decimals.

            DRIFT REPORT:
            {drift_block}
            """.strip()

    state.drift_summary = _gen_json(model, prompt)
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

                Return STRICT JSON ONLY with keys: status, findings, actions, rationale.
                Rules:
                - findings MUST be an ARRAY (use [] if none).
                - findings includes ONLY triggered numeric signals plus drift_overall (always).
                * Include {{"roc_auc_drop_pct": <one decimal>}} ONLY if relevant.
                * Include {{"latency_p95_ms": <integer>}} ONLY if two most recent p95 > 400ms.
                * ALWAYS include {{"drift_overall": <true|false>}}.
                - actions must be an ARRAY of allowed strings and consistent with status.
                - rationale: 2–4 sentences.

                Valid example (shape only):
                {{
                "status": "warn",
                "findings": [{{"drift_overall": true}}, {{"roc_auc_drop_pct": 4.3}}],
                "actions": ["trigger_retraining","page_oncall=false"],
                "rationale": "..."
                }}
                """.strip()

    raw = _gen_json(model, prompt)
    state.decision_json = _normalize_decision(
        raw, drift_overall=bool(state.drift_summary.get("overall_drift", False))
    )

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

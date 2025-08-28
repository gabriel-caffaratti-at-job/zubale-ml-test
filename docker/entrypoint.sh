#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-serve}"

case "$cmd" in
  train)
    shift || true
    exec uv run --frozen python -m src.train \
      --data "${DATA_PATH:-data/customer_churn_synth.csv}" \
      --outdir "${ARTIFACT_DIR:-artifacts}" \
      "$@"
    ;;
  serve)
    shift || true
    exec uv run --frozen python -m uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  drift)
    shift || true
    exec uv run --frozen python -m src.drift \
      --ref "${REF_PATH:-data/churn_ref_sample.csv}" \
      --new "${NEW_PATH:-data/churn_shifted_sample.csv}" \
      "$@"
    ;;
  agent)
    shift || true
    exec uv run --frozen python -m src.agent_monitor \
      --metrics "${METRICS_PATH:-data/metrics_history.jsonl}" \
      --drift "${DRIFT_PATH:-data/drift_latest.json}" \
      --out "${AGENT_OUT:-artifacts/agent_plan.yaml}" \
      "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

# GCP Design (Vertex AI Candidate)


## 1) Split the Repo into Purpose-Built Images

To keep builds small and roles clear, publish three images (CI builds all on push):

1. **Preprocess**  
   - Contains only `src/features.py` + deps.  
   - Fits and persists `preprocessor.pkl` from a reference dataset (BQ or GCS).

2. **Train**  
   - Contains `src/train.py`, `src/metrics.py`, and `features`.  
   - Saves `model.keras`/SavedModel, `metrics.json`, and importances/SHAP.

3. **Inference**  
   - Contains `src/app.py`, `src/io_schemas.py`, and `features`.  
   - Loads artifacts for serving; used either for Vertex Endpoint (as custom container) or for smoke-testing.

This split also helps **Pipelines** (each step pulls a small, focused image).

## 2) Vertex AI Pipeline Design

Create a Kubeflow/Vertex **Pipeline** with components:

1. **Load/Validate Data (A)**  
   - Input: BigQuery table (or GCS CSV).  
   - Validate schema & basic stats; optionally snapshot to GCS for reproducibility.

2. **Fit Preprocessor (B)**  
   - Runs the **preprocess image**.  
   - Output: `preprocessor.pkl` (saved to GCS).

3. **Train (C)**  
   - Runs the **training image**.  
   - Output: `model.keras`/SavedModel, `metrics.json`, `feature_importances.csv`.

4. **Evaluate & Gate (D)**  
   - Loads `metrics.json`.  
   - Enforces acceptance criteria (ROC-AUC ≥ 0.83, accuracy floor).  
   - Emits an approval flag and eval summary.

5. **Register Model (E)**  
   - If passed, registers to **Model Registry** with metrics + version tags.

6. **(Optional) Canary Deploy (F)**  
   - Deploy new version to a **staging Vertex Endpoint** (e.g., 20% traffic).  
   - Optionally promote to **prod** after canary passes.

7. **Post-Deploy Checks (G)**  
   - Update **Cloud Monitoring dashboards** and alerting (latency, errors).  
   - Publish pipeline run summary (metrics, URIs) to team channels.

8. **Drift Mini-Check (H, scheduled)**  
   - Reads reference + new window (BQ or GCS).  
   - Computes PSI/KS with `src.drift`.  
   - Writes `drift_latest.json` + summary row to BQ.  
   - Alerts if `overall_drift=true` or PSI exceeds thresholds.

9. **Agentic Monitor (I, scheduled)**  
   - Runs `src.agent_monitor` (Gemini on Vertex) to produce `agent_plan.yaml` from `metrics_history.jsonl` + `drift_latest.json`.  
   - Triggers follow-ups (open incident, retrain, rollback) via webhook when status is `warn`/`critical`.

> The **same pipeline** can be triggered manually, by scheduler, or on **data drift signals** (e.g., Pub/Sub messages from Drift component).

## 3) Observability on Vertex

- **Endpoint Monitoring**: Enable request logging, track latency (p50/p95/p99), QPS, and 4xx/5xx.  
- **Model Performance (delayed)**: Once labels arrive, compute AUC/PR-AUC on recent predictions and append to **metrics history**.  
- **Artifact Lineage**: Use Vertex **ML Metadata** (pipeline run → components → artifacts) for traceability.

## 4) Security & Config

- Use **Artifact Registry** for container images.  
- Use **GCS** for artifacts with per-env buckets (`-stg`, `-prod`).  
- Use **Workload Identity** for service accounts (avoid keys).  
- Store secrets in **Secret Manager**, mount only when needed.

## 5) Cost & Scale

- **Training**: Start with small CPUs (e.g., n1-standard-4) that hit the 0.83 ROC-AUC; scale if needed.  
- **Endpoints**: Start with one small CPU node; autoscale based on SLOs.  
- **Pipelines**: Leverage caching to skip re-runs for unchanged inputs (enable_caching=True).

## 6) Why a Pipeline?

- One-click, reproducible runs: **data → preprocess → train → gate → register → deploy**.  
- Governance: all versions traceable; rollback = flip a version in Registry.  
- Easy to attach **scheduled drift checks** and **LLM-based Agent Monitor** as downstream tasks.

---

### Notes

- Put **preprocess**, **training**, and **inference** in separate containers.  
- Orchestrate them with a **Vertex AI Pipeline** that: validates data → fits preprocessor → trains → evaluates/gates → registers → (optionally) deploys.  
- Use **Vertex AI Endpoints** for serving, **Cloud Monitoring** for ops metrics, a scheduled **drift job**, and your **agent monitor** to turn metrics+drift into concrete actions.

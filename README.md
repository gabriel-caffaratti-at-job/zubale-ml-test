# Mini-Prod ML Challenge (Starter)

# UV Quickstart — Install & Run (train / drift / test / serve)

This short guide shows how to **install [uv]** and how to use it to run your project’s common workflows: **train**, **drift**, **test**, and **serve**.

> Works on macOS, Linux, and Windows. Commands are shell-friendly; on PowerShell, swap `\` line continuations for `` ` ``.

---

## 1) Install `uv`

### macOS / Linux (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# After install, restart your shell or add uv to PATH as the installer suggests.
uv --version
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

> Alternative: via **pipx**
```bash
pipx install uv
uv --version
```

---


## 2) Run common workflows with `uv run`

`uv run` executes commands inside the project’s environment **without manually activating the venv**.

### 2.1 Train
```bash
uv run python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
```
### 2.2 Drift (data/model drift checks)
```bash
uv run python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
```

### 2.3 Test (pytest)
```bash
uv run pytest -q
```

### 2.4 Serve (local API)
```bash
# FastAPI via uvicorn
uv run python -m uvicorn src.app:app --port 8000
```

### 2.4 Agent (local API)
```bash
uv run python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml
```
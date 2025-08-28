# tests/test_training.py
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn_synth.csv"
ART  = ROOT / "artifacts"

def test_training_produces_artifacts_and_meets_auc_threshold():
    # Preconditions
    assert DATA.exists(), f"Missing dataset: {DATA}"

    # Run training as a module so imports like `from src.*` work
    cmd = [sys.executable, "-m", "src.train", "--data", str(DATA), "--outdir", str(ART)]
    
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

    # Artifacts
    model_path = ART / "model_keras.keras"
    pp_path    = ART / "preprocessor.pkl"
    metrics_fp = ART / "metrics.json"
    assert model_path.exists(), f"Model not saved at {model_path}"
    assert pp_path.exists(),    f"Preprocessor not saved at {pp_path}"
    assert metrics_fp.exists(), f"Metrics not found at {metrics_fp}"

    # Metrics
    with open(metrics_fp) as f:
        payload = json.load(f)
    assert "metrics" in payload, f"metrics.json missing 'metrics' key: {payload}"
    m = payload["metrics"]
    assert "roc_auc" in m and "pr_auc" in m and "accuracy" in m, f"Incomplete metrics: {m}"

    roc = float(m["roc_auc"])
    pr  = float(m["pr_auc"])
    acc = float(m["accuracy"])

    # Acceptance: ROC-AUC ≥ 0.83
    assert roc >= 0.83, (
        f"ROC-AUC below threshold: {roc:.4f} < 0.83 "
        f"(pr_auc={pr:.4f}, accuracy={acc:.4f}). "
        f"Check data split or training configuration."
    )

    fi_csv = ART / "feature_importances.csv"
    if fi_csv.exists():
        # lightweight sanity: non-empty file
        assert fi_csv.stat().st_size > 0, "feature_importances.csv is empty"

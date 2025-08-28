# src/drift.py
# CLI: uv run python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
import argparse, json, os
import numpy as np
import pandas as pd

DEFAULT_CATS = ["plan_type","contract_type","autopay","is_promo_user"]
DEFAULT_NUMS = [
    "add_on_count","tenure_months","monthly_usage_gb","avg_latency_ms",
    "support_tickets_30d","discount_pct","payment_failures_90d","downtime_hours_30d"
]

def _safe_quantiles(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if edges.size < 2:
        e = edges[0]
        return np.array([e - 1e-6, e + 1e-6])
    return edges

def psi_numeric(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    eps = 1e-10
    q = np.linspace(0, 1, bins + 1)
    edges = _safe_quantiles(ref, q)
    ref_hist, _ = np.histogram(ref, bins=edges)
    new_hist, _ = np.histogram(new, bins=edges)
    ref_pct = ref_hist / max(1, ref.size) + eps
    new_pct = new_hist / max(1, new.size) + eps
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

def psi_categorical(ref: pd.Series, new: pd.Series) -> float:
    eps = 1e-10
    cats = sorted(set(ref.dropna().unique()).union(set(new.dropna().unique())))
    ref_counts = ref.value_counts(normalize=True).reindex(cats).fillna(0.0).values + eps
    new_counts = new.value_counts(normalize=True).reindex(cats).fillna(0.0).values + eps
    return float(np.sum((new_counts - ref_counts) * np.log(new_counts / ref_counts)))

def split_columns(df: pd.DataFrame):
    cats = [c for c in DEFAULT_CATS if c in df.columns]
    nums = [c for c in DEFAULT_NUMS if c in df.columns]
    if cats or nums:
        return cats, nums
    # fallback
    cats = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for lbl in ("churned","target","label"):
        if lbl in cats: cats.remove(lbl)
        if lbl in nums: nums.remove(lbl)
    return cats, nums

def compute_drift(ref, new, cat_cols, num_cols, psi_bins=10, psi_threshold=0.20):
    features = {}
    overall = False

    for c in cat_cols + num_cols:
        if c not in ref.columns or c not in new.columns:
            continue
        if c in num_cols:
            ref_col = pd.to_numeric(ref[c], errors="coerce").to_numpy()
            new_col = pd.to_numeric(new[c], errors="coerce").to_numpy()
            psi = psi_numeric(ref_col, new_col, bins=psi_bins)
        else:
            psi = psi_categorical(ref[c], new[c])
        psi = round(psi, 2)
        features[c] = psi
        if psi >= psi_threshold:
            overall = True

    return {
        "overall_drift": overall,
        "threshold": psi_threshold,
        "features": features
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--psi_bins", type=int, default=10)
    ap.add_argument("--psi_threshold", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ref = pd.read_csv(args.ref)
    new = pd.read_csv(args.new)
    cats, nums = split_columns(ref)

    report = compute_drift(ref, new, cats, nums, psi_bins=args.psi_bins, psi_threshold=args.psi_threshold)

    out_path = os.path.join(args.outdir, "drift_latest.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

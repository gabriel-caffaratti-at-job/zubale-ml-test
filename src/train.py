# CLI: python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
# uv run python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
# src/train.py
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.features import TabPreprocess
from src.models import build_keras_model, to_keras_feed
from src.metrics import compute_metrics

CAT = ["plan_type","contract_type","autopay","is_promo_user"]
NUM = ["add_on_count","tenure_months","monthly_usage_gb","avg_latency_ms",
       "support_tickets_30d","discount_pct","payment_failures_90d","downtime_hours_30d"]
TARGET = "churned"

def permutation_importance(model, X_val, y_val, cat_cols, num_cols, base_probs, metric_fn):
    base_score = metric_fn(y_val, base_probs)["roc_auc"]
    results = []
    for col in cat_cols + num_cols:
        X_temp = X_val.copy()
        X_temp[col] = np.random.permutation(X_temp[col].values)
        probs = model.predict(to_keras_feed(X_temp, cat_cols, num_cols), verbose=0).ravel()
        score = metric_fn(y_val, probs)["roc_auc"]
        results.append({"feature": col, "auc_drop": float(base_score - score)})
    return pd.DataFrame(results).sort_values("auc_drop", ascending=False)

def main():
    #arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load data
    df = pd.read_csv(args.data)
    X = df[CAT + NUM]
    y = df[TARGET].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2,
                                              random_state=args.seed, stratify=y)

    # preprocess
    pre = TabPreprocess(CAT, NUM, artifacts_dir=args.outdir)
    Xtr_m = pre.fit_transform(X_tr)
    Xva_m = pre.transform(X_va)

    # model definition
    model = build_keras_model(Xtr_m, CAT, NUM)

    # early stopping callback 
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max",
                                           patience=5, restore_best_weights=True)]

    model.fit(
        to_keras_feed(Xtr_m, CAT, NUM), y_tr,
        validation_data=(to_keras_feed(Xva_m, CAT, NUM), y_va),
        epochs=50, batch_size=512, verbose=1, callbacks=cb
    )

    probs = model.predict(to_keras_feed(Xva_m, CAT, NUM), verbose=0).ravel()
    metrics = compute_metrics(y_va, probs)

    # permutation feature importance
    fi_df = permutation_importance(model, Xva_m, y_va, CAT, NUM, probs, compute_metrics)
    fi_path = os.path.join(args.outdir, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)

    #save model
    model.save(os.path.join(args.outdir, "model_keras.keras"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

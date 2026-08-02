"""
train.py
========
End-to-end offline training entry point. Run this ONCE (locally or in a
notebook/Colab) to produce the artifacts the Streamlit app loads at runtime:

    artifacts/model.h5              (best checkpointed Keras model)
    artifacts/feature_scaler.joblib (MinMaxScaler fit on training features)
    artifacts/target_scaler.joblib  (MinMaxScaler fit on training target)
    artifacts/metrics.json          (held-out test metrics)
    artifacts/history.json          (loss curves, for the "Model Insights" tab)

Usage:
    python train.py --data data/powerconsumption.csv
    python train.py --data data/powerconsumption.csv --epochs 150 --batch-size 64

If --data is omitted and no file exists at the default path, a synthetic
dataset is generated automatically so the whole pipeline is runnable
out-of-the-box (see utils/generate_sample_data.py docstring for why, and how
to swap in the real Kaggle CSV).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import run_full_pipeline, FEATURE_COLS, TARGET_COL, target_distribution_summary
from src.model import train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train the energy load forecasting ANN.")
    parser.add_argument("--data", default="data/powerconsumption.csv", help="Path to the raw Kaggle CSV.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[train.py] No dataset found at {args.data}.")
        print("[train.py] Generating a synthetic stand-in dataset so you can test the pipeline now.")
        print("[train.py] >>> Replace this file with the real Kaggle CSV before deploying to production. <<<")
        from utils.generate_sample_data import generate
        generate(out_path=args.data)

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("[train.py] Running Phase 1 pipeline: clean -> engineer features -> scale -> chronological split ...")
    engineered_df, split, feature_scaler, target_scaler = run_full_pipeline(args.data)

    print("[train.py] Target distribution diagnostics:")
    try:
        print(json.dumps(target_distribution_summary(engineered_df), indent=2))
    except ImportError:
        print("  (scipy not installed -- skipping skewness/kurtosis diagnostic, non-fatal)")

    # Carve a validation split out of the training partition (chronological,
    # last 15% of train) so EarlyStopping/Checkpoint never see the test set.
    n_train = len(split.X_train)
    val_cut = int(n_train * 0.85)
    X_tr, X_val = split.X_train[:val_cut], split.X_train[val_cut:]
    y_tr, y_val = split.y_train[:val_cut], split.y_train[val_cut:]

    print(f"[train.py] Train={len(X_tr)}  Val={len(X_val)}  Test={len(split.X_test)}  Features={FEATURE_COLS}")

    print("[train.py] Running Phase 2: building & training the pure ANN (MLP) ...")
    model, history = train_model(
        X_tr, y_tr, X_val, y_val,
        checkpoint_path=os.path.join(args.artifacts_dir, "model.h5"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )

    print("[train.py] Evaluating on the held-out chronological test set ...")
    metrics = evaluate_model(model, split.X_test, split.y_test, target_scaler)
    print(json.dumps(metrics, indent=2))

    # --- Persist all artifacts ---
    joblib.dump(feature_scaler, os.path.join(args.artifacts_dir, "feature_scaler.joblib"))
    joblib.dump(target_scaler, os.path.join(args.artifacts_dir, "target_scaler.joblib"))
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "feature_columns.json"), "w") as f:
        json.dump({"features": FEATURE_COLS, "target": TARGET_COL}, f, indent=2)

    print(f"[train.py] Done. Artifacts saved to '{args.artifacts_dir}/'.")
    print("[train.py] You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()

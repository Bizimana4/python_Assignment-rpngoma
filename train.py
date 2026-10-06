"""Train and compare FNN experiments, then save the best model.

Usage:
    python train.py                      # uses data/Crop_recommendation.csv
    python train.py --data path/to.csv   # use another CSV
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_prep import (
    ROOT, BASE_FEATURES, clean_data, encode_labels, load_config, load_data,
    quality_report, scale_features, split_indices,
)
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_history
from src.features import build_feature_frame
from src.model import build_fnn


def run(config_path=None, data_override=None, verbose=0):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    results_dir = ROOT / "results"
    models_dir = ROOT / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # ---- 1. acquire + inspect + clean --------------------------------
    df = load_data(data_override or cfg["data_path"])
    report = quality_report(df)
    df, removed = clean_data(df)
    report["rows_removed_by_cleaning"] = removed
    with open(results_dir / "data_quality.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Data quality:", json.dumps(report, indent=2))

    y, encoder = encode_labels(df)
    class_names = list(encoder.classes_)
    train_idx, val_idx, test_idx = split_indices(
        y, cfg["test_size"], cfg["val_size"], seed
    )
    print(f"Split -> train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # ---- 2. run the experiments --------------------------------------
    rows, best = [], None
    for exp in cfg["experiments"]:
        X = build_feature_frame(df, exp["use_engineered_features"])
        scaler, Xtr, Xva, Xte = scale_features(
            X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
        )
        model = build_fnn(
            n_features=Xtr.shape[1], n_classes=len(class_names),
            hidden_units=exp["hidden_units"], dropout=exp["dropout"],
            learning_rate=exp["learning_rate"], seed=seed,
        )
        stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
        )
        t0 = time.time()
        history = model.fit(
            Xtr, y[train_idx], validation_data=(Xva, y[val_idx]),
            epochs=cfg["max_epochs"], batch_size=cfg["batch_size"],
            callbacks=[stop], verbose=verbose,
        )
        secs = time.time() - t0
        val_metrics, _, _, _ = evaluate_model(model, Xva, y[val_idx], class_names)
        tr_acc = float(model.evaluate(Xtr, y[train_idx], verbose=0)[1])
        val_loss = float(model.evaluate(Xva, y[val_idx], verbose=0)[0])
        row = {
            "experiment": exp["name"],
            "features": Xtr.shape[1],
            "hidden_units": str(exp["hidden_units"]),
            "dropout": exp["dropout"],
            "learning_rate": exp["learning_rate"],
            "epochs_run": len(history.history["loss"]),
            "train_accuracy": round(tr_acc, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "val_loss": round(val_loss, 4),
            "train_seconds": round(secs, 1),
        }
        rows.append(row)
        print(row)
        # best = highest val accuracy; ties broken by val F1, then lower val loss
        key = (row["val_accuracy"], row["val_f1_macro"], -row["val_loss"])
        if best is None or key > best["key"]:
            best = dict(key=key, row=row, exp=exp, model=model, scaler=scaler,
                        history=history, Xte=Xte)

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv(results_dir / "experiments.csv", index=False)
    print("\nExperiment comparison (chosen on VALIDATION accuracy):")
    print(exp_df.to_string(index=False))

    # ---- 3. final evaluation of the chosen model on the TEST set -----
    exp = best["exp"]
    metrics, text_report, cm, _ = evaluate_model(
        best["model"], best["Xte"], y[test_idx], class_names
    )
    print(f"\nBest experiment: {exp['name']}")
    print("TEST metrics:", json.dumps(metrics, indent=2))
    print(text_report)

    (results_dir / "classification_report.txt").write_text(text_report)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"best_experiment": exp["name"], "test_metrics": metrics}, f, indent=2)
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix.png")
    plot_history(best["history"], results_dir / "training_curves.png",
                 title=f"Training curves - {exp['name']}")

    # ---- 4. save everything needed to reproduce / serve the model ----
    best["model"].save(models_dir / "crop_fnn.keras")
    joblib.dump(best["scaler"], models_dir / "scaler.joblib")
    joblib.dump(encoder, models_dir / "label_encoder.joblib")
    meta = {
        "experiment": exp,
        "feature_names_raw": BASE_FEATURES,
        "use_engineered_features": exp["use_engineered_features"],
        "class_names": class_names,
        "raw_feature_ranges": {
            c: [float(df[c].min()), float(df[c].max())] for c in BASE_FEATURES
        },
        "test_metrics": metrics,
        "seed": seed,
    }
    with open(models_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved model, scaler, label encoder and metadata to models/")
    return meta, exp_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to a CSV (optional)")
    ap.add_argument("--config", default=None, help="path to a config.json")
    ap.add_argument("--verbose", type=int, default=0, help="Keras verbosity 0/1/2")
    args = ap.parse_args()
    run(args.config, args.data, args.verbose)

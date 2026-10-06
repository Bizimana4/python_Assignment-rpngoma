"""Gradio demo - Crop Recommendation (FNN).

Run:  python app.py      then open the local URL shown in the terminal.
Train first (python train.py) so the models/ folder exists.
"""
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import ROOT, BASE_FEATURES
from src.features import build_feature_frame

MODELS = ROOT / "models"
DISCLAIMER = (
    "Advisory tool only. It was trained on data from India, so results may "
    "not suit Rwandan soils and climate. Confirm with a local agronomist "
    "(e.g. RAB extension officer) before planting."
)

_model = _scaler = _meta = None


def _load():
    global _model, _scaler, _meta
    if _model is None:
        if not (MODELS / "crop_fnn.keras").exists():
            raise FileNotFoundError(
                "No trained model found in the models/ folder. First put "
                "Crop_recommendation.csv in data/, then run:  python train.py"
            )
        from tensorflow import keras
        try:
            _model = keras.models.load_model(MODELS / "crop_fnn.keras")
        except Exception as err:  # e.g. model saved by a different TensorFlow version
            print(f"Could not load the saved model ({type(err).__name__}: {err}).")
            print("Re-training with the TensorFlow installed on this computer ...")
            import train
            train.run()
            _model = keras.models.load_model(MODELS / "crop_fnn.keras")
        _scaler = joblib.load(MODELS / "scaler.joblib")
        with open(MODELS / "model_meta.json", encoding="utf-8") as f:
            _meta = json.load(f)
    return _model, _scaler, _meta


def _validate_inputs(values):
    missing = [name for name, value in values.items() if value is None]
    if missing:
        return "Please fill in all seven values."

    min_values = {
        "N": 0,
        "P": 0,
        "K": 0,
        "temperature": -20,
        "humidity": 0,
        "ph": 0,
        "rainfall": 0,
    }
    for name, minimum in min_values.items():
        if values[name] < minimum:
            return f"{name} cannot be lower than {minimum}."

    max_values = {
        "N": 500,
        "P": 300,
        "K": 300,
        "temperature": 60,
        "humidity": 100,
        "ph": 14,
        "rainfall": 500,
    }
    for name, maximum in max_values.items():
        if values[name] > maximum:
            return f"{name} cannot be higher than {maximum}."

    return None


def _make_result_chart(top3):
    labels = list(top3.keys())
    values = list(top3.values())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color=["#3AAE7F", "#7AC8A4", "#B9E1C6"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Confidence")
    ax.set_title("Top 3 crop predictions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(ax.patches, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def _write_export_file(top3, notes):
    export_text = "Crop Recommendation Result\n\n" + notes + "\n\nTop 3 crops:\n"
    for idx, (crop, score) in enumerate(top3.items(), start=1):
        export_text += f"{idx}. {crop}: {score:.1%}\n"

    temp_dir = Path(tempfile.gettempdir())
    file_path = temp_dir / "crop_recommendation_result.txt"
    file_path.write_text(export_text, encoding="utf-8")
    return str(file_path)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Return (top-3 probabilities dict, notes text). Errors are shown in Notes."""
    try:
        return _predict(N, P, K, temperature, humidity, ph, rainfall)
    except FileNotFoundError as e:
        return None, f"SETUP PROBLEM: {e}"
    except Exception as e:  # keep the demo alive and tell the user what happened
        import traceback
        traceback.print_exc()
        return None, f"Something went wrong ({type(e).__name__}): {e}"


def predict_crop_full(N, P, K, temperature, humidity, ph, rainfall):
    try:
        top3, notes = _predict(N, P, K, temperature, humidity, ph, rainfall)
        if top3 is None:
            return None, notes, None, None
        chart = _make_result_chart(top3)
        export_path = _write_export_file(top3, notes)
        return top3, notes, chart, export_path
    except FileNotFoundError as e:
        return None, f"SETUP PROBLEM: {e}", None, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Something went wrong ({type(e).__name__}): {e}", None, None


def _predict(N, P, K, temperature, humidity, ph, rainfall):
    model, scaler, meta = _load()
    values = dict(zip(BASE_FEATURES, [N, P, K, temperature, humidity, ph, rainfall]))
    validation_error = _validate_inputs(values)
    if validation_error:
        return None, validation_error

    # Guardrail: warn when an input is outside the range seen in training.
    warnings = []
    for name, (lo, hi) in meta["raw_feature_ranges"].items():
        if not lo <= values[name] <= hi:
            warnings.append(f"{name}={values[name]} is outside the training range "
                            f"[{lo:g}, {hi:g}] - treat the result with caution.")

    row = pd.DataFrame([values])
    X = build_feature_frame(row, meta["use_engineered_features"])
    probs = model.predict(scaler.transform(X), verbose=0)[0]
    scores = {c: float(p) for c, p in zip(meta["class_names"], probs)}
    top3 = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3])
    best_crop, best_p = next(iter(top3.items()))
    notes = f"Recommended crop: {best_crop} (confidence {best_p:.1%}).\n"
    if best_p < 0.6:
        notes += "Low confidence - several crops fit these conditions.\n"
    if warnings:
        notes += "\n".join("WARNING: " + w for w in warnings) + "\n"
    notes += "\n" + DISCLAIMER
    return top3, notes


def build_ui():
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Crop Recommendation Assistant
            Enter the soil and weather values to get the three most suitable crops for the farm.
            """
        )
        gr.HTML(
            '<div style="background:#f4faf6; border:1px solid #dfeee2; border-radius:10px; padding:12px 14px; color:#1d3a2d; margin-bottom:10px;">'
            + DISCLAIMER +
            '</div>'
        )

        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    n = gr.Number(label="Nitrogen (N)", value=90)
                    p = gr.Number(label="Phosphorus (P)", value=42)
                    k = gr.Number(label="Potassium (K)", value=43)
                    temperature = gr.Number(label="Temperature (°C)", value=21)
                    humidity = gr.Number(label="Humidity (%)", value=82)
                    ph = gr.Number(label="Soil pH", value=6.5)
                    rainfall = gr.Number(label="Rainfall (mm)", value=203)
                    predict_btn = gr.Button("Predict crop", variant="primary")

                with gr.Column(scale=1):
                    top3_output = gr.Label(num_top_classes=3, label="Top 3 recommended crops")
                    notes_output = gr.Textbox(label="Prediction notes", lines=8)
                    chart_output = gr.Plot(label="Confidence chart")
                    export_output = gr.File(label="Download result")

            examples = [
                [90, 42, 43, 21, 82, 6.5, 203],
                [20, 130, 200, 23, 83, 6.0, 70],
                [100, 18, 50, 27, 80, 6.0, 25],
            ]
            gr.Examples(
                examples=examples,
                inputs=[n, p, k, temperature, humidity, ph, rainfall],
                outputs=[top3_output, notes_output, chart_output, export_output],
                fn=predict_crop_full,
                label="Try a sample input",
                cache_examples=False,
            )

            predict_btn.click(
                fn=predict_crop_full,
                inputs=[n, p, k, temperature, humidity, ph, rainfall],
                outputs=[top3_output, notes_output, chart_output, export_output],
            )

        with gr.Tab("How to read the result"):
            gr.Markdown(
                """
                ### How the result is interpreted
                - The model gives the top 3 crop predictions ranked by confidence.
                - The first crop is the best recommendation for the entered conditions.
                - If the input is outside the training range, the app shows a warning.
                - If confidence is low, the result is less certain and should be checked carefully.

                ### Example output
                For a typical maize-like soil profile, the app may return something like:
                1. Jute — 52.2%
                2. Rice — 47.8%
                3. Coffee — 0.06%

                This means the system believes that Jute is the most suitable crop, while the other two crops are possible but less likely.
                """
            )
            gr.HTML(
                """
                <div style="background:#f9fbf9; border:1px solid #dfeee2; border-radius:10px; padding:18px; max-width:700px;">
                  <strong>Key idea:</strong> the app does not show only one crop. It shows the most likely crops with confidence values so the farmer can compare options.
                </div>
                """
            )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7862")),
    )

"""Step 2 - Feature engineering.

Adds agronomy-inspired features on top of the 7 raw measurements.
The same function is used in training and in the Gradio app so the model
always receives features built in exactly the same way.
"""
BASE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
ENGINEERED_FEATURES = [
    "npk_total",
    "n_to_p",
    "n_to_k",
    "p_to_k",
    "temp_humidity_index",
    "ph_deviation",
]


def add_engineered_features(df):
    """Return a copy of df with the engineered columns appended."""
    out = df.copy()
    out["npk_total"] = out["N"] + out["P"] + out["K"]
    # +1 avoids division by zero when a nutrient is 0
    out["n_to_p"] = out["N"] / (out["P"] + 1.0)
    out["n_to_k"] = out["N"] / (out["K"] + 1.0)
    out["p_to_k"] = out["P"] / (out["K"] + 1.0)
    out["temp_humidity_index"] = out["temperature"] * out["humidity"] / 100.0
    # distance from a neutral soil pH (6.5): very acidic/alkaline soils differ
    out["ph_deviation"] = (out["ph"] - 6.5).abs()
    return out


def get_feature_names(use_engineered):
    return BASE_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])


def build_feature_frame(df, use_engineered):
    """Select the columns the model will see, in a fixed order."""
    if use_engineered:
        df = add_engineered_features(df)
    return df[get_feature_names(use_engineered)].astype("float32")

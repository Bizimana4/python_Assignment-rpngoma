"""TEST ONLY - creates a fake dataset with the same columns as the Kaggle one.

It lets you check that the pipeline runs end-to-end when the real CSV is not
available. Do NOT use its results for your CAT: use the real Kaggle data.

    python tools/make_synthetic_data.py      -> data/synthetic_test.csv
    python train.py --data data/synthetic_test.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
crops = ["rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
         "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
         "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
         "coconut", "cotton", "jute", "coffee"]
rows = []
for crop in crops:
    centre = np.array([rng.uniform(0, 140), rng.uniform(5, 145), rng.uniform(5, 200),
                       rng.uniform(9, 43), rng.uniform(15, 95), rng.uniform(4, 9),
                       rng.uniform(20, 290)])
    spread = np.array([12, 12, 12, 2.5, 6, 0.4, 20])
    for _ in range(100):
        v = centre + rng.normal(0, 1, 7) * spread
        rows.append([*v, crop])
df = pd.DataFrame(rows, columns=["N", "P", "K", "temperature", "humidity",
                                 "ph", "rainfall", "label"])
df[["N", "P", "K"]] = df[["N", "P", "K"]].clip(lower=0).round().astype(int)
df["humidity"] = df["humidity"].clip(14, 100)
df["ph"] = df["ph"].clip(3.5, 9.9)
df["rainfall"] = df["rainfall"].clip(lower=20)
out = Path(__file__).resolve().parent.parent / "data" / "synthetic_test.csv"
df.sample(frac=1, random_state=0).to_csv(out, index=False)
print("wrote", out, df.shape)

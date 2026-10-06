"""Step 3 - Model definition (feed-forward neural network, Keras)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_fnn(n_features, n_classes, hidden_units, dropout, learning_rate, seed=42):
    """Dense network: [Dense -> ReLU -> Dropout] x N  ->  Softmax(n_classes)."""
    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential(name="crop_fnn")
    model.add(layers.Input(shape=(n_features,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

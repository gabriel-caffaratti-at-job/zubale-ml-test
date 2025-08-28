# src/model.py
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_keras_model(mapped_train_df, cat_cols, num_cols):
    """
    Keras Model for churn:
    - Categorical inputs (embeddings) + numerical inputs
    - Full connected layers with dropout
    """

    inputs = {
        **{c: layers.Input(shape=(1,), name=c, dtype="int64") for c in cat_cols},
        **{c: layers.Input(shape=(1,), name=c, dtype="float32") for c in num_cols},
    }

    # Categorical embeddings
    emb_flat = []
    for c in cat_cols:
        vocab = int(mapped_train_df[c].max())
        emb_dim = max(2, min(50, int(math.sqrt(max(2, vocab)) * 2)))
        emb = layers.Embedding(input_dim=vocab+1, output_dim=emb_dim, name=f"emb_{c}")(inputs[c])
        emb_flat.append(layers.Flatten()(emb))

    pieces = emb_flat[:]
    if num_cols:
        nums = layers.Concatenate(name="numerics")([inputs[c] for c in num_cols])
        nums = layers.Flatten()(nums)
        pieces.append(nums)

    feats = layers.Concatenate(name="features")(pieces) if len(pieces) > 1 else pieces[0]

    x = layers.Dense(128, activation="relu")(feats)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="roc_auc"), "accuracy"])
    return model

def to_keras_feed(df, cat_cols, num_cols):
    """Devuelve dict {input_name: array} para .fit/.predict"""
    feed = {}
    for c in cat_cols + num_cols:
        feed[c] = df[c].values
    return feed

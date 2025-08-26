import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

def build_lstm(input_dim, cfg):
    model = models.Sequential([
        layers.Input(shape=(1, input_dim)),
        layers.LSTM(cfg["HIDDEN"], return_sequences=False, dropout=cfg["DROPOUT"]),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(cfg["LR"]), loss="mse")
    return model

def get_callbacks(cfg):
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=cfg["PATIENCE"], restore_best_weights=True)
    ]

# models/dqn_network.py
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model

class DuelingDQN_LSTM(Model):
    """
    Dueling DQN with 2 LSTM layers over a sliding window of features.
    Input shape: (batch, seq_len, n_features)
    Output: Q-values per action in order [short, flat, long]
    """
    def __init__(
        self,
        n_features: int,
        n_actions: int = 3,
        lstm1_units: int = 64,
        lstm2_units: int = 32,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_actions = n_actions

        self.lstm1 = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")
        self.lstm2 = layers.LSTM(lstm2_units, return_sequences=False, name="lstm2")
        self.dropout = layers.Dropout(dropout)
        self.act = layers.LeakyReLU(0.1)

        # Value stream
        self.val_fc1 = layers.Dense(lstm2_units, activation=self.act, name="val_fc1")
        self.val_out = layers.Dense(1, name="V")

        # Advantage stream
        self.adv_fc1 = layers.Dense(lstm2_units, activation=self.act, name="adv_fc1")
        self.adv_out = layers.Dense(n_actions, name="A")

        # Build once (helps summary() users)
        dummy = tf.zeros((1, 16, n_features), dtype=tf.float32)
        _ = self.call(dummy)

    def call(self, x, training=False):
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout(x, training=training)

        V = self.val_out(self.val_fc1(x, training=training))              # (B, 1)
        A = self.adv_out(self.adv_fc1(x, training=training))              # (B, nA)
        A_mean = tf.reduce_mean(A, axis=1, keepdims=True)                 # (B, 1)
        Q = V + (A - A_mean)                                              # (B, nA)
        return Q

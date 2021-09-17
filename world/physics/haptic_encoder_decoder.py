import numpy as np
import tensorflow as tf

from world.physics.utils import create_bidir_lstm_layer


class HapticEncoderDecoder(tf.keras.Model):
    def __init__(self, batch_size: int, dropout: float):
        super().__init__()
        self.batch_size = batch_size
        self.dropout = dropout

        # encodes input states in to the feature form
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2D(32, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2D(64, 5, 2, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2D(128, 5, 5, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout)
        ])

        self.state_encoder_flatten = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128)
        ])

        # aggregate data from timesteps
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(64, 5, 5, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(32, 5, 2, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(16, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(1, 5, 4, padding="same")
        ])

        # aggregate data from timesteps
        self.aggregator = create_bidir_lstm_layer(self.batch_size,
                                                  lstm_units=128,
                                                  dropout=self.dropout,
                                                  return_sequences=False,
                                                  stateful=True)

    def __call__(self, inputs, training=None, mask=None):
        depth_before, action = inputs
        features = self.encoder(depth_before, training=training)
        features_flat = self.state_encoder_flatten(features, training=training)

        feed = tf.concat([features_flat, action], -1)
        feed = tf.expand_dims(tf.cast(feed, tf.float32), 0)
        lstm_outputs = self.aggregator(feed, training=training)

        num_units = np.prod(tf.shape(features).numpy())
        features_flat_action = tf.keras.Sequential([
            tf.keras.layers.Dense(num_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout)
        ])(lstm_outputs, training=training)

        features_flat_action_reshaped = tf.reshape(features_flat_action, shape=tf.shape(features))
        depth_after_hat = self.decoder(features_flat_action_reshaped, training=training)
        return depth_after_hat

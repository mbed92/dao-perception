import tensorflow as tf


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
            tf.keras.layers.Conv2D(64, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2D(128, 5, 4, padding="same"),
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
            tf.keras.layers.Conv2DTranspose(128, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(64, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(32, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2DTranspose(16, 5, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Conv2D(1, 5, 4, padding="same"),
        ])

    def __call__(self, inputs, training=None, mask=None):
        depth_before, action = inputs
        features = self.encoder(depth_before, training=training)
        features_flat = self.state_encoder_flatten(features, training=training)

        feed = tf.concat([features_flat, action], -1)
        feed = tf.cast(feed, tf.float32)

        depth_after_hat = self.decoder(feed, training=training)
        return depth_after_hat

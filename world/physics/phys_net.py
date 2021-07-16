import tensorflow as tf

from world.physics.utils import create_bidir_lstm_layer


class HapticRegressor(tf.keras.Model):
    def __init__(self, batch_size: int, num_outputs: int, action_kernel_size: int, dropout: float, lstm_units: int,
                 stateful_lstm: bool):
        super().__init__()
        self.batch_size = batch_size
        self.num_outputs = num_outputs
        self.action_kernel_size = action_kernel_size
        self.dropout = dropout
        self.lstm_units = lstm_units
        self.stateful_lstm = stateful_lstm

        # encodes input states in to the feature form
        self.state_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(self.dropout)
        ])

        # aggregate data from timesteps
        self.aggregator = create_bidir_lstm_layer(self.batch_size,
                                                  self.lstm_units,
                                                  dropout=self.dropout,
                                                  return_sequences=False,
                                                  stateful=self.stateful_lstm)
        self.state = None

        # final layer
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.num_outputs)
        ])

    def call(self, inputs, training=None, mask=None):
        observations, action = inputs
        state_features = self.state_encoder(observations, training=training)

        feed = tf.concat([state_features, action], -1)
        feed = tf.cast(feed, tf.float32)

        lstm_outputs = self.aggregator(feed, training=training)
        predictions = self.regressor(lstm_outputs, training=training)

        return predictions

    def reset_states(self):
        self.aggregator.reset_states()

import tensorflow as tf

import nn


class CosseratNet(tf.keras.Model):

    def __init__(self, batch_size, num_pts, seq_length, momentum=0.99, activation='relu'):
        super().__init__(name='cosserat_net')
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.seq_length = seq_length
        self.dim = 3

        self.extractor = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, kernel_size=1, padding="same"),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(256, kernel_size=1, padding="same"),
            tf.keras.layers.BatchNormalization(momentum=momentum)
        ])

        # create bilstm modules
        self.bilstm = tf.keras.Sequential([
            nn.layers.recurrent.create_bidir_lstm_layer(128, True)
        ])

        self.predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_pts * self.dim),
        ])

    def call(self, inputs, training=None):  # noqa
        sequence, target, params = inputs
        b, s = tf.shape(sequence)[:2]

        # mix sequence, target pose and params into one feature bag
        target = tf.tile(target, [1, s, 1, 1])
        seq_t = tf.concat([sequence, target], axis=2)
        seq_t = tf.reshape(seq_t, shape=[b, s, -1])
        params = tf.tile(tf.expand_dims(params, 1), [1, s, 1])
        seq_t_p = tf.concat([seq_t, params], -1)

        features = self.extractor(seq_t_p, training=training)
        recurrent = self.bilstm(features, training=training)
        predictions = self.predictor(recurrent, training=training)
        predictions = tf.reshape(predictions, shape=[b, s, self.dim, self.num_pts])
        return predictions

    def warmup(self):
        sequence = tf.zeros([2, self.seq_length, self.dim, self.num_pts])
        target = tf.zeros([2, 1, self.dim, self.num_pts])
        params = tf.zeros([2, 4])
        self([sequence, target, params], training=tf.constant(False))

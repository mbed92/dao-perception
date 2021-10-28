import tensorflow as tf


def create_bidir_lstm_layer(lstm_units, return_sequences=True, dropout=0.0, stateful=False):
    forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dtype=tf.float32,
                                         dropout=dropout, stateful=stateful)
    backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                          dropout=dropout, dtype=tf.float32, stateful=stateful)
    return tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                         input_shape=(None, int(2 * lstm_units)))

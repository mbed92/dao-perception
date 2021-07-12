import tensorflow as tf


def create_bidir_lstm_layer(batch_size, lstm_units, dropout=0.0, stateful=True, return_states=True):
    forward_layer = tf.keras.layers.LSTM(lstm_units, dtype=tf.float32, dropout=dropout,
                                         stateful=stateful, return_state=return_states)
    backward_layer = tf.keras.layers.LSTM(lstm_units, go_backwards=True, dropout=dropout, dtype=tf.float32,
                                          stateful=stateful, return_state=return_states)
    return tf.keras.layers.Bidirectional(forward_layer,
                                         backward_layer=backward_layer,
                                         input_shape=(batch_size, int(2 * lstm_units)))

def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

import tensorflow as tf

from utils.text import TextFlag, log


def create_bidir_lstm_layer(batch_size, lstm_units, return_sequences=True, dropout=0.0, stateful=True):
    forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dtype=tf.float32,
                                         dropout=dropout, stateful=stateful)
    backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                          dropout=dropout, dtype=tf.float32, stateful=stateful)
    return tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                         input_shape=(batch_size, int(2 * lstm_units)))


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            log(TextFlag.INFO, f"Physical GPUs {len(gpus)} Logical GPUs{len(logical_gpus)}")
        except RuntimeError as e:
            log(TextFlag.ERROR, e)

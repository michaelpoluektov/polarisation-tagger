import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa E402
from keras.layers import (
    MultiHeadAttention,
    Dense,
    Dropout,
    Flatten,
    Layer,
    Masking,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
    Concatenate,
    Input
)  # noqa E402
from tensorflow.keras import Model  # noqa E402


def get_model(config) -> Model:
    if config.activation == "gelu":
        activation = tf.keras.activations.gelu
    elif config.activation == "relu":
        activation = tf.keras.activations.relu
    else:
        raise NotImplementedError
    ins = Input(shape=(None, 10))
    x = Masking(mask_value=0.)(ins)
    for i in range(config.num_blocks):
        mha = MultiHeadAttention(num_heads=config.num_heads,
                                 key_dim=10, attention_axes=1)(x, x)
        x = x + mha
        ln = LayerNormalization()(x)
        x = Dense(config.hidden_states, activation=activation)(ln)
        x = Dropout(config.dropout_rate)(x)
        x = Dense(10, activation=activation)(x)
        x = ln + x
    x = GlobalAveragePooling1D()(x)
    # x = Dense(config.top_out, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=ins, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

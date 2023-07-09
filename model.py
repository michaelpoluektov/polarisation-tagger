import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa E402
from keras.layers import (
    MultiHeadAttention,
    Dense,
    Dropout,
    Masking,
    LayerNormalization,
    GlobalAveragePooling1D,
    Input
)  # noqa E402
from tensorflow.keras import Model  # noqa E402


NUM_FEATURES = 11


def get_activation(ion: str):
    if ion == "gelu":
        activation = tf.keras.activations.gelu
    elif ion == "relu":
        activation = tf.keras.activations.relu
    elif ion == "swish":
        activation = tf.keras.activations.swish
    else:
        raise NotImplementedError
    return activation


def get_model(config) -> Model:
    activation = get_activation(config.activation)
    ins = Input(shape=(None, NUM_FEATURES))
    x = Masking(mask_value=0.)(ins)
    for i in range(config.num_blocks):
        mha = MultiHeadAttention(num_heads=config.num_heads,
                                 key_dim=NUM_FEATURES, attention_axes=1)(x, x)
        x = x + mha
        ln = LayerNormalization()(x)
        x = Dense(config.hidden_states, activation=activation)(ln)
        x = Dropout(config.dropout_rate)(x)
        x = Dense(NUM_FEATURES, activation=activation)(x)
        x = ln + x
    x = GlobalAveragePooling1D()(x)
    # x = Dense(config.top_out, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=ins, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def get_model_3(config) -> Model:
    activation = get_activation(config.activation)

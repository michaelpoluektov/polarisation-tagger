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
    Input,
    Layer,
)  # noqa E402
from tensorflow.keras import Model  # noqa E402


NUM_FEATURES = 11


class ThreeWayAttention(Layer):
    def __init__(self, units=11):
        super(ThreeWayAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.d1 = Dense(self.units)
        self.d2 = Dense(self.units)
        self.d3 = Dense(self.units)

    def call(self, inputs, mask=None):
        X, V = inputs
        seq_len = tf.shape(X)[1]
        if mask is not None:
            mask_X, mask_V = mask
            a = tf.cast(mask_X, tf.float32)
            a = tf.cast(
                (a[:, tf.newaxis, :] + a[:, :, tf.newaxis] == 2), tf.float32) * -1e9
            a += tf.expand_dims(tf.eye(seq_len) * -1e9, 0)
        else:
            a = tf.expand_dims(tf.eye(seq_len) * -1e9, 0)
        Q1 = self.d1(X)
        Q2 = self.d2(X)
        Q3 = self.d3(X)
        dot1 = (tf.matmul(Q1, Q2, transpose_b=True) + a)[:, :, :, tf.newaxis]
        dot2 = (tf.matmul(Q2, Q3, transpose_b=True) + a)[:, tf.newaxis, :, :]
        dot3 = (tf.matmul(Q1, Q3, transpose_b=True) + a)[:, :, tf.newaxis, :]
        threeway_scores = (dot1 + dot2 + dot3) / 9
        weighed_V = self.cartesian_product(
            V) * tf.expand_dims(tf.keras.activations.softmax(threeway_scores, axis=[1, 2, 3]), -1)
        return tf.reduce_sum(weighed_V, axis=[1, 2, 3])

    @staticmethod
    def cartesian_product(X):
        s = tf.shape(X)[1]
        f = tf.shape(X)[-1]
        z = tf.zeros((s, s, s, f))
        X1 = X[:, :, tf.newaxis, tf.newaxis, :]
        X2 = X[:, tf.newaxis, :, tf.newaxis, :]
        X3 = X[:, tf.newaxis, tf.newaxis, :, :]
        return tf.concat([X1 + z, X2 + z, X3 + z], axis=-1)


class PoolingMHA(Layer):
    def __init__(self, d: int, k: int, h: int, activation="relu"):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            activation: activation function
        """
        super(PoolingMHA, self).__init__()
        self.mha = MultiHeadAttention(num_heads=h, key_dim=d, attention_axes=1)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ff1_1 = Dense(d, activation=activation)
        self.ff1_2 = Dense(d, activation=activation)
        self.ff2_1 = Dense(d, activation=activation)
        self.ff2_2 = Dense(d, activation=activation)
        self.seed_vectors = tf.Variable(
            tf.random.normal(shape=(1, k, d)), trainable=True)

    @tf.function
    def call(self, z):
        s = tf.repeat(self.seed_vectors, tf.shape(z)[0], axis=0)
        x = self.ff2_1(self.ff2_2(z))
        h = self.ln1(s + self.mha(s, x, x))
        return self.ln2(h + self.ff1_1(self.ff1_2(h)))


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

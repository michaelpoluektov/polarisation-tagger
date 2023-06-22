import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb  # noqa E402
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint  # noqa E402
import absl.logging  # noqa E402
absl.logging.set_verbosity(absl.logging.ERROR)
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
print(tf.config.list_physical_devices('GPU'))


NAME = "nodrop_30"


wandb.init(
    project="polarisation-trigger-full",
    config={
        "hidden_states": 30,
        "dropout_rate": 0.20,
        "top_out": 20,
        "learning_rate": 0.001,
        "batch_size": 128,
        "num_heads": 8,
        "num_blocks": 6,
        "max_tracks": 30,
        "epochs": 200,
        "activation": "gelu",
    }
)

config = wandb.config
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
    x = Dense(config.hidden_states, activation=tf.keras.activations.gelu)(ln)
    x = Dropout(config.dropout_rate)(x)
    x = Dense(10, activation=tf.keras.activations.gelu)(x)
    x = ln + x
x = GlobalAveragePooling1D()(x)
# x = Dense(config.top_out, activation='relu')(x)
output = Dense(1)(x)
model = Model(inputs=ins, outputs=output)
opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
model.compile(loss='mean_squared_error', optimizer=opt)

track_data = np.load(f"track_data_{config.max_tracks}.npy")
target = np.load(f"target_{config.max_tracks}.npy")

dataset = tf.data.Dataset.from_tensor_slices((track_data, target)).shuffle(
    buffer_size=200000, seed=42).batch(config.batch_size)
test_dataset = dataset.take(100)
train_dataset = dataset.skip(100)
model.fit(
    train_dataset,
    epochs=config.epochs,
    validation_data=test_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint(
            "models/checkpoint_{epoch:02d}.h5", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5),
        tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True)
    ],
    verbose=1
)

model.save(f"models/{NAME}.h5")

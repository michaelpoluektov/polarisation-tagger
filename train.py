from model import get_model
from dataset import get_train_test
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb  # noqa E402
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint  # noqa E402
import absl.logging  # noqa E402
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf  # noqa E402
print(tf.config.list_physical_devices('GPU'))


NAME = "nodrop_30"


wandb.init(
    project="polarisation-tagger-full",
    config={
        "hidden_states": 30,
        "dropout_rate": 0.,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_heads": 6,
        "num_blocks": 5,
        "max_tracks": 30,
        "epochs": 200,
        "min_tracks": 1,
        "min_distance": 0.,
        "activation": "swish",
    }
)

config = wandb.config
model = get_model(config)
print(model.count_params())
train_dataset, test_dataset = get_train_test(
    config.batch_size,
    config.max_tracks,
    num_samples_test=150,
    min_distance=config.min_distance,
    min_tracks=config.min_tracks
)
model.fit(
    train_dataset,
    epochs=config.epochs,
    validation_data=test_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint(
            "models/checkpoint_{epoch:02d}.h5", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, monitor="loss", patience=5),
        tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True)
    ],
    verbose=1
)

model.save(f"models/{NAME}.h5")

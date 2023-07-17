from model import get_model, get_set_transformer, PoolingMHA
from dataset import get_train_test
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb  # noqa E402
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint  # noqa E402
import absl.logging  # noqa E402
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf  # noqa E402
print(tf.config.list_physical_devices('GPU'))


NAME = "nodrop_15"
PARAM_LIMIT = 1e6

sweep_config = {
    "method": "random",  # or "grid"
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "hidden_states": {"min": 20, "max": 40},
        "hidden_states_2": {"min": 6, "max": 10},
        "hidden_states_3": {"min": 15, "max": 25},
        "seed_vectors": {"min": 3, "max": 6},
        "num_heads": {"min": 2, "max": 4},
        "num_blocks": {"value": 2},
        "num_blocks_2": {"value": 2},
        "learning_rate": {"min": -8, "max": -6.3, "distribution": "log_uniform"},
        "batch_size": {"value": 128},
        "max_tracks": {"min": 10, "max": 30},
        "min_tracks": {"value": 3},
        "epochs": {"value": 200},
        "min_distance": {"min": -3, "max": 1, "distribution": "log_uniform"},
        "delta_distance": {"min": 5., "max": 10.},
        "activation": {"value": "swish"}
    }
}


def train():
    run = wandb.init()
    config = run.config
    model = get_set_transformer(config)
    if model.count_params() > PARAM_LIMIT:
        run.finish()
        return

    train_dataset, test_dataset = get_train_test(
        config.batch_size,
        config.max_tracks,
        num_samples_test=150,
        min_distance=config.min_distance,
        delta_distance=config.delta_distance,
        min_tracks=config.min_tracks
    )

    model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=test_dataset,
        callbacks=[
            # WandbMetricsLogger(log_freq=5),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, monitor="loss", patience=5),
            tf.keras.callbacks.EarlyStopping(
                patience=30, restore_best_weights=True)
        ],
        verbose=1
    )
    best_val_loss = min(model.history.history["val_loss"])
    best_train_loss = min(model.history.history["loss"])
    num_samples = train_dataset.cordinality().numpy() * config.batch_size
    wandb.log({"val": best_val_loss,
               "train": best_train_loss,
               "params": model.count_params(),
               "samples": num_samples,
               "params_per_sample": model.count_params() / num_samples
               })


sweep_id = wandb.sweep(sweep_config, project="polarisation-tagger")
wandb.agent(sweep_id, function=train)

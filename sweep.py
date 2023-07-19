from model import get_set_transformer
from dataset import get_train_test
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb  # noqa E402
from wandb.keras import WandbMetricsLogger  # noqa E402
import absl.logging  # noqa E402
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf  # noqa E402
print(tf.config.list_physical_devices('GPU'))


NAME = "nodrop_15"
MAX_PARAM_PER_SAMPLE = 1/5

sweep_config = {
    "method": "random",  # or "grid"
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "seed": {"min": 0, "max": 1000000},
        "hidden_states": {"min": 24, "max": 40},
        "hidden_states_2": {"min": 6, "max": 10},
        "hidden_states_3": {"min": 15, "max": 30},
        "seed_vectors": {"min": 4, "max": 8},
        "num_heads": {"min": 1, "max": 2},
        "num_blocks": {"min": 1, "max": 5},
        "num_blocks_2": {"min": 1, "max": 5},
        "learning_rate": {"min": -8, "max": -4.3, "distribution": "log_uniform"},
        "batch_size": {"min": 128, "max": 1024},
        "max_tracks": {"min": 10, "max": 30},
        "min_tracks": {"value": 3},
        "epochs": {"value": 200},
        "min_distance": {"min": -4, "max": 0, "distribution": "log_uniform"},
        "delta_distance": {"min": 10., "max": 20.},
        "activation": {"value": "swish"}
    }
}


def train():
    run = wandb.init()
    config = run.config
    tf.keras.utils.set_random_seed(config.seed)
    model = get_set_transformer(config)
    train_dataset, test_dataset = get_train_test(
        config,
        m="mu16",
        num_samples_test=(150 * 128) // config.batch_size,
    )
    val_dataset, _ = get_train_test(
        config,
        m="md16",
        num_samples_test=1
    )
    pc = model.count_params()
    ns = tf.data.experimental.cardinality(train_dataset) * config.batch_size
    if pc / ns > MAX_PARAM_PER_SAMPLE:
        print(pc, ns)
        run.finish()
        return

    model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=test_dataset,
        callbacks=[
            WandbMetricsLogger(log_freq=5),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, monitor="loss", patience=5),
            tf.keras.callbacks.EarlyStopping(
                patience=30, restore_best_weights=True, min_delta=0.0002, baseline=0.)
        ],
        verbose=1
    )
    best_val_loss = min(model.history.history["val_loss"])
    best_train_loss = min(model.history.history["loss"])
    num_samples = tf.data.experimental.cardinality(
        train_dataset).numpy() * config.batch_size
    v = model.evaluate(val_dataset)
    wandb.log({"val": best_val_loss,
               "train": best_train_loss,
               "params": model.count_params(),
               "samples": num_samples,
               "params_per_sample": model.count_params() / num_samples,
               "md16": v
               })
    model.save(f"model_{run.id}.h5")


sweep_id = wandb.sweep(sweep_config, project="polarisation-tagger")
wandb.agent(sweep_id, function=train)

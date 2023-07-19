from model import get_set_transformer
from dataset import get_train_test
import wandb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging  # noqa E402
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf  # noqa E402
print(tf.config.list_physical_devices('GPU'))


class Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


api = wandb.Api()
sweep = api.sweep("polarisation-tagger/db3chg7r")
best_run = sorted(
    sweep.runs, key=lambda run: run.summary.get('md16', float('inf')))[0]
print(best_run.config)
best_config = Obj(best_run.config)
model = get_set_transformer(best_config)
train_ds, test_ds = get_train_test(best_config, num_samples_test=(
    150 * 128) // best_config.batch_size, m="mu16")
val_ds, _ = get_train_test(best_config, num_samples_test=1, m="md16")
model.fit(
    train_ds,
    epochs=best_config.epochs,
    validation_data=test_ds,
    callbacks=[
        # WandbMetricsLogger(log_freq=5),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, monitor="loss", patience=5),
        tf.keras.callbacks.EarlyStopping(
            patience=50, restore_best_weights=True)
    ],
    verbose=1
)
i = model.evaluate(val_ds)
print(i)
model.save("models/model_v2.h5")

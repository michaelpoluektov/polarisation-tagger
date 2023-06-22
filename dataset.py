import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.data import Dataset  # noqa E402


def get_train_test(batch_size: int,
                   max_tracks: int = 30,
                   num_samples_test: int = 100,
                   shuffle: bool = True
                   ) -> tuple[Dataset, Dataset]:
    track_data = np.load(f"data/track_data_{max_tracks}.npy")
    target = np.load(f"data/target_{max_tracks}.npy")
    dataset = Dataset.from_tensor_slices((track_data, target))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200000, seed=42)
    dataset = dataset.batch(batch_size)
    test_dataset = dataset.take(100)
    train_dataset = dataset.skip(100)
    return train_dataset, test_dataset

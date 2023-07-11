import numpy as np
import pandas as pd
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.data import Dataset  # noqa E402

TARGET_COL = ["ny", "nx", "nz"]
m = "mu16"

probcols = ["ProbNNpi", "ProbNNK", "ProbNNp", "ProbNNmu", "ProbNNe"]
pcols = ["PX", "PY", "PZ"]
dcols = ["DX", "DY"]
normcols = pcols + dcols
feature_columns = normcols + ["Q"] + probcols


def clip(df):
    df["PX"] = df["PX"].clip(-5000, 5000)
    df["PY"] = df["PY"].clip(-5000, 5000)
    df["PZ"] = df["PZ"].clip(0, 100000)
    for col in probcols:
        df[col] = df[col].clip(0, 1) * 2 - 1
    df["DX"] = df["DX"].clip(-12, 12) * 2
    df["DY"] = df["DY"].clip(-12, 12) * 2


def get_arrays(tracks, combinations, max_tracks, min_distance, min_tracks):
    n_tracks_list = combinations["ntracks"].to_numpy()
    start_idx = 0
    zero_tl = []
    type_arr = (tracks["type"] == 0).to_numpy()
    dx_dy_condition = (
        (tracks["DX"]**2 + tracks["DY"]**2) > min_distance).to_numpy()
    and_arr = dx_dy_condition  # & type_arr
    for i in tqdm(n_tracks_list):
        zero_tl.append((and_arr)[
                       start_idx:start_idx+i].sum())
        start_idx += i
    zero_tl = np.array(zero_tl)
    tracks_np = tracks[and_arr].to_numpy().astype(np.float32)[..., :11]
    num_compat = len(zero_tl[(zero_tl < max_tracks) & (zero_tl >= min_tracks)])
    track_data = np.zeros((num_compat, max_tracks, 11), dtype=np.float32)
    vals = combinations[TARGET_COL].to_numpy()
    start_idx = 0
    target = []
    i = 0
    for val, n_tracks in zip(tqdm(vals), zero_tl):
        if n_tracks < max_tracks and n_tracks >= min_tracks:
            track_data[i, :n_tracks] = tracks_np[start_idx:start_idx + n_tracks]
            target.append(val)
            i += 1
        start_idx += n_tracks
    target = np.array(target)
    # target = (target-target.mean())/target.std()
    return target, track_data


def get_train_test(batch_size: int,
                   max_tracks: int = 30,
                   num_samples_test: int = 100,
                   shuffle: bool = True,
                   min_distance: float = 0.05,
                   min_tracks: int = 3
                   ) -> tuple[Dataset, Dataset]:
    tracks = pd.read_parquet(f"data/tracks_{m}.parquet")
    combinations = pd.read_parquet(f"data/combinations_{m}.parquet")
    clip(tracks)
    tracks[normcols] = ((tracks[normcols] - tracks[normcols].mean()) /
                        tracks[normcols].std()).astype(np.float32)
    target, track_data = get_arrays(
        tracks, combinations, max_tracks, min_distance, min_tracks)
    dataset = Dataset.from_tensor_slices((track_data, target))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=300000, seed=42)
    dataset = dataset.batch(batch_size)
    test_dataset = dataset.take(num_samples_test)
    train_dataset = dataset.skip(num_samples_test)
    return train_dataset, test_dataset

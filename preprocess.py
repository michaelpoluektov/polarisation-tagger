import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import pad_sequences  # noqa E402
import numpy as np  # noqa E402
import pandas as pd  # noqa E402
from tqdm import tqdm  # noqa E402

MAX_TRACKS = 30
TARGET_COL = "c_ID"
m = "md16"

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


tracks = pd.read_parquet(f"data/tracks_{m}.parquet")
combinations = pd.read_parquet(f"data/combinations_{m}.parquet")
clip(tracks)
tracks[normcols] = ((tracks[normcols] - tracks[normcols].mean()) /
                    tracks[normcols].std()).astype(np.float32)


def get_arrays(tracks, combinations):
    n_tracks_list = combinations["ntracks"].to_numpy()
    start_idx = 0
    zero_tl = []
    type_arr = (tracks["type"] == 0).to_numpy()
    for i in n_tracks_list:
        zero_tl.append(type_arr[start_idx:start_idx+i].sum())
        start_idx += i
    zero_tl = np.array(zero_tl)
    tracks_np = tracks[tracks["type"] ==
                       0].to_numpy().astype(np.float32)[..., :11]
    num_compat = len(zero_tl[zero_tl < MAX_TRACKS])
    track_data = np.zeros((num_compat, MAX_TRACKS, 11), dtype=np.float32)
    vals = combinations[TARGET_COL].to_numpy()
    start_idx = 0
    target = []
    i = 0
    for val, n_tracks in zip(vals, zero_tl):
        if n_tracks < MAX_TRACKS:
            track_data[i, :n_tracks] = tracks_np[start_idx:start_idx + n_tracks]
            target.append(val)
            i += 1
        start_idx += n_tracks
    target = np.array(target)
    target = (target-target.mean())/target.std()
    return target, track_data


# target, track_data = get_arrays_slow(tracks, combinations)
target, track_data = get_arrays(tracks, combinations)
np.save(f"data/track_data_{m}_{MAX_TRACKS}.npy", track_data)
np.save(f"data/target_{m}_{MAX_TRACKS}.npy", target)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import pad_sequences  # noqa E402
import numpy as np  # noqa E402
import pandas as pd  # noqa E402
from tqdm import tqdm  # noqa E402

MAX_TRACKS = 30

tracks = pd.read_parquet("tracks.parquet")
combinations = pd.read_parquet("combinations.parquet")
tracks["PX"] = tracks["PX"].clip(-5000, 5000)
tracks["PY"] = tracks["PY"].clip(-5000, 5000)
tracks["PZ"] = tracks["PZ"].clip(0, 100000)
tracks["PIDmu"] = tracks["PIDmu"].clip(-20, 100)
tracks["PIDp"] = tracks["PIDp"].clip(-100, 100)
tracks["PIDK"] = tracks["PIDK"].clip(-100, 100)
tracks["DX"] = tracks["DX"].clip(-12, 12)
tracks["DY"] = tracks["DY"].clip(-12, 12)
cols = tracks.columns[:10]
tracks[cols] = ((tracks[cols] - tracks[cols].mean()) /
                tracks[cols].std()).astype(np.float32)

n_tracks_list = combinations['ntracks'].values
feature_columns = tracks.columns[:10]
track_data = []
vals = combinations["ny"].values
target = []
start_idx = 0
for val, n_tracks in zip(vals, tqdm(n_tracks_list)):
    a = 0
    if n_tracks < MAX_TRACKS:
        df_slice = tracks.iloc[start_idx:start_idx + n_tracks]
        assert not df_slice.iloc[0]["ntrack"]
        df_slice = df_slice[df_slice["type"] == 0]
        a = len(df_slice[df_slice["type"] != 0])
        track_data.append(df_slice[feature_columns].values)
        target.append(val)
    start_idx += (n_tracks - a)
target = np.array(target)
target = (target-target.mean())/target.std()
track_data = np.array(pad_sequences(track_data, dtype=np.float32))
np.save(f"track_data_{MAX_TRACKS}.npy", track_data)
np.save(f"target_{MAX_TRACKS}.npy", target)

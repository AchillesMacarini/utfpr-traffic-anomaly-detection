import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_anomaly_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep=',')
    annotation_dict = {}
    for _, row in df.iterrows():
        video_id = str(row['video_id'])
        start = row['start_time']
        end = row['end_time']
        annotation_dict.setdefault(video_id, []).append((start, end))
    return annotation_dict

def is_anomalous(timestamp, anomaly_intervals):
    for start, end in anomaly_intervals:
        if start <= timestamp <= end:
            return 1
    return 0

def extract_and_save_sequences(json_dir, annotation_dict, sequence_length=10, output_dir="sequences"):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [fname for fname in os.listdir(json_dir) if fname.endswith(".json")]
    print(f"Starting extracting sequences from JSON files in {json_dir}...")

    for fname in tqdm(json_files, desc="Processing JSON files"):
        video_id = fname.split("_")[1].split(".")[0]
        anomaly_intervals = annotation_dict.get(video_id, [])
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
        X, y = [], []
        for car_id, traj in data.items():
            df = pd.DataFrame(traj)
            for i in range(len(df) - sequence_length + 1):
                seq = df.iloc[i:i+sequence_length][["x", "y", "speed"]].values
                timestamps = df.iloc[i:i+sequence_length]["timestamp"].values
                label = max(is_anomalous(ts, anomaly_intervals) for ts in timestamps)
                X.append(seq)
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_sequences.npz")
        np.savez_compressed(output_path, X=X, y=y)
    print(f"Extraction and saving complete. Sequences saved in {output_dir}.")

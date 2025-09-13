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

def extract_and_save_sequences(json_dir, annotation_dict, sequence_length=10, output_dir="sequences", return_df=False):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [fname for fname in os.listdir(json_dir) if fname.endswith(".json")]
    json_files = [f for f in json_files if f != 'anomalies.json']  # Exclude anomalies.json
    json_files = json_files[:100]  # Limit to first 100 files for testing
    print(f"Starting extracting sequences from JSON files in {json_dir}...")

    all_rows = []
    for fname in tqdm(json_files, desc="Processing JSON files"):
        parts = fname.split("_", 1)
        if len(parts) > 1:
            video_id = parts[1].split(".")[0]
        else:
            video_id = os.path.splitext(fname)[0]
        
        anomaly_intervals = annotation_dict.get(video_id, [])
        
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
            
        X, y = [], []
        # Handle the case where data is a dictionary of vehicles
        if isinstance(data, dict):
            for vehicle_id, traj in data.items():
                if not isinstance(traj, list) or len(traj) < sequence_length:
                    continue
                    
                speeds = [point.get('speed', 0) for point in traj]
                
                # Create sequences
                for i in range(len(speeds) - sequence_length + 1):
                    seq = np.array(speeds[i:i+sequence_length])
                    X.append(seq.reshape(-1, 1))  # reshape to (sequence_length, 1)
                    y.append(0)  # Default to non-anomalous if we don't have timestamp info
                    all_rows.append({
                        "video_id": video_id,
                        "vehicle_id": vehicle_id,
                        "sequence": seq,
                        "label": 0
                    })

        if X:  # Only save if we have sequences
            X = np.array(X)
            y = np.array(y)
            output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_sequences.npz")
            np.savez_compressed(output_path, X=X, y=y)
            print(f"Saved {len(X)} sequences from {fname}")
            
    print(f"Extraction and saving complete. Sequences saved in {output_dir}.")
    
    if return_df:
        if all_rows:
            df_out = pd.DataFrame(all_rows)
            print("Sample of extracted DataFrame:")
            print(df_out.head())
            return df_out
        else:
            print("No sequences were extracted.")
            return pd.DataFrame()

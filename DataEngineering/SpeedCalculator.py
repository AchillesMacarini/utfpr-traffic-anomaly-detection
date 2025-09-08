import cv2
import os
from tqdm import tqdm
import sys
import pandas as pd
import json
import numpy as np

video_json = "trajectories_2.json"

class SpeedCalculator:
    def __init__(self, json_path, output_csv_path, fps=30):
        self.json_path = json_path
        self.output_csv_path = output_csv_path
        self.fps = fps

    def calculate_speed_and_direction(self):
        # Load trajectory data from JSON
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        results = []
        for car_id, positions in data.items():
            positions = np.array(positions)
            if len(positions) == 0:
                continue

            timestamps = positions[:, 0]
            xs = positions[:, 1]
            ys = positions[:, 2]

            # First iteration: set speed and direction as None
            results.append({
                'car_id': car_id,
                'timestamp': timestamps[0],
                'x': xs[0],
                'y': ys[0],
                'speed_px_s': None,
                'direction_x': None,
                'direction_y': None
            })

            if len(positions) < 2:
                continue  # Only one point, nothing more to calculate

            # Calculate differences between consecutive positions
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            dt = timestamps[1:] - timestamps[:-1]

            # Speed in px/s: (current_pos - prev_pos) / (current_time - prev_time)
            speed_px_s = np.sqrt(dx**2 + dy**2) / dt

            # Direction vector (normalized): (current_pos - prev_pos) / norm
            directions = np.stack([dx, dy], axis=1)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            direction_vectors = np.divide(directions, norms, out=np.zeros_like(directions), where=norms!=0)

            for i in range(1, len(positions)):
                results.append({
                    'car_id': car_id,
                    'timestamp': timestamps[i],
                    'x': xs[i],
                    'y': ys[i],
                    'speed_px_s': speed_px_s[i-1],
                    'direction_x': direction_vectors[i-1][0],
                    'direction_y': direction_vectors[i-1][1]
                })

        df = pd.DataFrame(results)
        df.to_csv(self.output_csv_path, index=False)
        print(f"Speed and direction data saved to {self.output_csv_path}")

if __name__ == "__main__":
    # Example usage
    json_path = video_json
    output_csv_path = video_csv
    fps = 24  # Change if needed

    calc = SpeedCalculator(json_path, output_csv_path, fps)
    calc.calculate_speed_and_direction()

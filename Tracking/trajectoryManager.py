import json
import csv

class TrajectoryManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = {}

    def add_trajectory(self, car_id, trajectory):
        """Add a trajectory for a specific car ID."""
        self.trajectories[car_id] = trajectory

    def save_trajectories(self):
        """Save the trajectories to a JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.trajectories, f)

    def load_trajectories(self):
        """Load trajectories from a JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                self.trajectories = json.load(f)
        except FileNotFoundError:
            print("No trajectory data found. Starting with an empty dataset.")

    def get_trajectory(self, car_id):
        """Get the trajectory for a specific car ID."""
        return self.trajectories.get(car_id, None)

    def clear_trajectories(self):
        """Clear all stored trajectories."""
        self.trajectories = {}

    def save_trajectories_csv(self, fps, output_csv_path):
        """
        Save trajectories to a CSV file, each column is a second.
        Each cell contains (cx, cy) or is empty if no data for that second.
        """
        # Find the max trajectory length in seconds
        max_frames = max(len(traj) for traj in self.trajectories.values())
        max_seconds = (max_frames // fps) + 1

        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header: car_id, sec_0, sec_1, ..., sec_N
            header = ['car_id'] + [f'sec_{i}' for i in range(max_seconds)]
            writer.writerow(header)

            for car_id, traj in self.trajectories.items():
                row = [car_id]
                for sec in range(max_seconds):
                    # Get frame index for this second
                    frame_idx = sec * fps
                    if frame_idx < len(traj):
                        cx, cy = traj[frame_idx]
                        row.append(f'({cx},{cy})')
                    else:
                        row.append('')
                writer.writerow(row)
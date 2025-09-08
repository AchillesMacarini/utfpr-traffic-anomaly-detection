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
        Save trajectories to a CSV file: car_id, timestamp (s), cx, cy
        """
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['car_id', 'timestamp', 'cx', 'cy'])
            for car_id, traj in self.trajectories.items():
                for (timestamp, cx, cy) in traj:
                    writer.writerow([car_id, round(timestamp, 2), cx, cy])
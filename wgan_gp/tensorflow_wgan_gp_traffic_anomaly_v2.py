import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"TensorFlow using GPU: {physical_devices[0]}")
else:
    print("TensorFlow using CPU")

class ImprovedTrafficTrajectoryDataset:
    """Enhanced dataset for traffic trajectories with focus on normal data"""
    
    def __init__(self, data_dir: str, anomaly_labels: Optional['TrafficAnomalyLabels'] = None, 
                 max_trajectories_per_video: int = 800, validation_split: float = 0.15):
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.max_trajectories_per_video = max_trajectories_per_video
        self.validation_split = validation_split
        
        # Use RobustScaler for better normalization
        self.scaler = RobustScaler()
        
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Load and process data prioritizing normal data quality"""
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"Processing {len(npy_files)} trajectory files...")
        
        normal_trajectories = []
        anomaly_trajectories = []
        
        for npy_file in npy_files:
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) == 0:
                    continue
                
                # Filter quality trajectories
                quality_trajectories = self._filter_quality_trajectories(trajectories)
                
                if len(quality_trajectories) == 0:
                    continue
                
                # Limit quantity per file (more for normal data)
                max_per_file = self.max_trajectories_per_video
                if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                    max_per_file = min(200, len(quality_trajectories))  # Less anomalies
                
                if len(quality_trajectories) > max_per_file:
                    indices = np.random.choice(len(quality_trajectories), max_per_file, replace=False)
                    quality_trajectories = quality_trajectories[indices]
                
                # Classify as normal or anomalous
                if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                    anomaly_trajectories.append(quality_trajectories)
                    print(f"   {npy_file}: {len(quality_trajectories)} ANOMALOUS trajectories")
                else:
                    normal_trajectories.append(quality_trajectories)
                    print(f"   {npy_file}: {len(quality_trajectories)} normal trajectories")
                    
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
        
        # Prepare final datasets
        self._prepare_final_datasets(normal_trajectories, anomaly_trajectories)
        
        print(f"Dataset prepared:")
        print(f"   Training (normal only): {self.train_data.shape}")
        print(f"   Validation: {self.val_data.shape}")
        print(f"   Validation labels - Normal: {np.sum(~self.val_labels)}, Anomalous: {np.sum(self.val_labels)}")
    
    def _filter_quality_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """Filter high-quality trajectories"""
        good_trajectories = []
        
        for traj in trajectories:
            # Check for invalid values
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                continue
            
            # Check for sufficient movement (not stationary)
            pos_variance = np.var(traj[:, :2], axis=0)  # variance in x,y
            if np.sum(pos_variance) < 1e-6:  # very little movement
                continue
            
            # Check for extreme velocities
            velocities = traj[:, 2]  # velocity column
            if np.max(velocities) > 10 or np.std(velocities) > 5:  # very high values
                continue
            
            # Check temporal continuity (no large gaps)
            accelerations = traj[:, 3]  # acceleration column
            if np.max(np.abs(accelerations)) > 8:  # too abrupt acceleration
                continue
            
            good_trajectories.append(traj)
        
        return np.array(good_trajectories) if good_trajectories else np.empty((0, 20, 5))
    
    def _prepare_final_datasets(self, normal_trajectories: List, anomaly_trajectories: List):
        """Prepare datasets with focus on quality normal data"""
        
        if not normal_trajectories:
            raise ValueError("No normal trajectories found!")
        
        # Concatenate normal trajectories
        all_normal = np.concatenate(normal_trajectories, axis=0)
        print(f"Total normal trajectories: {len(all_normal)}")
        
        # Split normal data into training and validation
        n_normal = len(all_normal)
        n_val_normal = int(n_normal * self.validation_split)
        
        indices = np.random.permutation(n_normal)
        val_normal_indices = indices[:n_val_normal]
        train_indices = indices[n_val_normal:]
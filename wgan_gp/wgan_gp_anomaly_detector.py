import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

class AnomalyLabels:
    """Class to manage anomaly labels based on CSV"""
    
    def __init__(self, csv_path: str, fps: float = 30.0):
        self.csv_path = csv_path
        self.fps = fps
        self.anomaly_intervals = {}
        
        print(f"Loading anomaly labels from: {csv_path}")
        self._load_anomaly_data()
    
    def _load_anomaly_data(self):
        """Load anomaly data from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            
            for _, row in df.iterrows():
                video_id = int(row['video_id'])
                start_time = float(row['start_time'])
                end_time = float(row['end_time'])
                
                if video_id not in self.anomaly_intervals:
                    self.anomaly_intervals[video_id] = []
                
                self.anomaly_intervals[video_id].append((start_time, end_time))
            
            total_anomalies = sum(len(intervals) for intervals in self.anomaly_intervals.values())
            print(f"Loaded {total_anomalies} anomaly intervals for {len(self.anomaly_intervals)} videos")
            
        except Exception as e:
            print(f"Error loading anomalies: {e}")
            self.anomaly_intervals = {}
    
    def has_anomalies(self, video_id: int) -> bool:
        """Check if a video has known anomalies"""
        return video_id in self.anomaly_intervals and len(self.anomaly_intervals[video_id]) > 0


class TrajectoryDataset(Dataset):
    """Optimized dataset for loading processed trajectories"""
    
    def __init__(self, data_dir: str, anomaly_labels: Optional[AnomalyLabels] = None, 
                 train_split: float = 0.8, use_only_normal_videos: bool = True, 
                 max_trajectories_per_video: int = 500):  # Limit trajectories per video
        """
        Args:
            data_dir: Directory with processed .npy files
            anomaly_labels: AnomalyLabels instance (optional)
            train_split: Proportion for training
            use_only_normal_videos: If True, use only videos without anomalies for training
            max_trajectories_per_video: Maximum trajectories per video (for optimization)
        """
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.use_only_normal_videos = use_only_normal_videos
        self.max_trajectories_per_video = max_trajectories_per_video
        
        # Load .npy files
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"Loading trajectories from {len(npy_files)} files...")
        
        normal_videos = []
        anomaly_videos = []
        all_trajectories = []
        video_sources = []
        
        with tqdm(total=len(npy_files), desc="Loading data", unit="file") as pbar:
            for npy_file in npy_files:
                video_id = int(npy_file.split('_')[0])
                file_path = os.path.join(data_dir, npy_file)
                
                try:
                    trajectories = np.load(file_path)
                    if len(trajectories) > 0:
                        # Limit number of trajectories per video for optimization
                        if len(trajectories) > self.max_trajectories_per_video:
                            indices = np.random.choice(len(trajectories), self.max_trajectories_per_video, replace=False)
                            trajectories = trajectories[indices]
                        
                        # Classify video
                        if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                            anomaly_videos.append((video_id, trajectories))
                            video_type = "with anomalies"
                        else:
                            normal_videos.append((video_id, trajectories))
                            video_type = "normal"
                        
                        all_trajectories.append(trajectories)
                        video_sources.extend([video_id] * len(trajectories))
                        
                        pbar.set_postfix({'Video': f'{video_id} ({video_type})', 'Trajs': len(trajectories)})
                        
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
                
                pbar.update(1)
        
        if all_trajectories:
            self.all_trajectories = np.concatenate(all_trajectories, axis=0)
            self.video_sources = np.array(video_sources)
        else:
            raise ValueError("No valid trajectories found!")
        
        self._prepare_training_data(normal_videos, anomaly_videos, train_split)
        
        print(f"Dataset loaded:")
        print(f"   Total videos: {len(npy_files)}")
        print(f"   Normal videos: {len(normal_videos)}")
        print(f"   Videos with anomalies: {len(anomaly_videos)}")
        print(f"   Total trajectories: {len(self.all_trajectories)}")
        print(f"   Training: {len(self.train_trajectories)} trajectories")
        print(f"   Validation: {len(self.val_trajectories)} trajectories")
    
    def _prepare_training_data(self, normal_videos: List, anomaly_videos: List, train_split: float):
        """Prepare data for training and validation in an optimized way"""
        
        if self.use_only_normal_videos:
            # Training: only trajectories from normal videos
            normal_trajectories = [traj for _, traj in normal_videos]
            if normal_trajectories:
                all_normal = np.concatenate(normal_trajectories, axis=0)
                
                # Split normal trajectories
                n_normal = len(all_normal)
                n_train_normal = int(n_normal * train_split)
                
                indices = np.random.permutation(n_normal)
                self.train_trajectories = all_normal[indices[:n_train_normal]]
                val_normal_traj = all_normal[indices[n_train_normal:]]
                
                print(f"Using {len(self.train_trajectories)} normal trajectories for training")
            else:
                raise ValueError("No normal videos found!")
            
            # Validation: mix of normal + some anomalous trajectories
            val_trajectories = [val_normal_traj]
            val_labels = [False] * len(val_normal_traj)
            
            if anomaly_videos:
                # Use only some videos with anomalies for validation (optimization)
                n_anomaly_videos = min(3, len(anomaly_videos))  # Maximum 3 anomalous videos
                selected_anomaly_videos = np.random.choice(len(anomaly_videos), n_anomaly_videos, replace=False)
                
                for idx in selected_anomaly_videos:
                    _, anomaly_traj = anomaly_videos[idx]
                    # Limit anomalous trajectories for validation
                    if len(anomaly_traj) > 200:
                        anomaly_indices = np.random.choice(len(anomaly_traj), 200, replace=False)
                        anomaly_traj = anomaly_traj[anomaly_indices]
                    
                    val_trajectories.append(anomaly_traj)
                    val_labels.extend([True] * len(anomaly_traj))
            
            self.val_trajectories = np.concatenate(val_trajectories, axis=0)
            self.val_has_anomaly = np.array(val_labels)
    
    def get_train_data(self):
        return torch.FloatTensor(self.train_trajectories)
    
    def get_val_data(self):
        return torch.FloatTensor(self.val_trajectories), self.val_has_anomaly
    
    def __len__(self):
        return len(self.train_trajectories)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.train_trajectories[idx])


class Generator(nn.Module):
    """Optimized generator with simpler architecture"""
    
    def __init__(self, noise_dim: int = 64, seq_len: int = 20, feature_dim: int = 5):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Simpler and faster architecture
        self.main = nn.Sequential(
            # Initial layer
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            # Expand to sequence
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            # Output layer
            nn.Linear(256, seq_len * feature_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.main(noise)
        return output.view(batch_size, self.seq_len, self.feature_dim)


class Discriminator(nn.Module):
    """Optimized discriminator using Conv1D instead of LSTM"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(Discriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Use Conv1D to be faster than LSTM
        self.conv_layers = nn.Sequential(
            # input: [batch, feature_dim, seq_len]
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # No final activation for WGAN
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, trajectories):
        # Transpose for Conv1D format: [batch, feature_dim, seq_len]
        x = trajectories.transpose(1, 2)
        return self.conv_layers(x).squeeze()


class WGANGPAnomalyDetector:
    """Optimized anomaly detection system using WGAN-GP"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        print(f"Initializing optimized WGAN-GP on device: {self.device}")
        
        # Optimized hyperparameters
        self.noise_dim = 64  # Reduced from 100
        self.lr_g = 0.0002   # Optimized learning rate
        self.lr_d = 0.0001   # Different learning rate for discriminator
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_gp = 10
        self.n_critic = 3    # Reduced from 5 to accelerate
        
        # Initialize networks
        self.generator = Generator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = Discriminator(seq_len, feature_dim).to(self.device)
        
        # Optimizers with different learning rates
        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))
        
        # History
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': []
        }
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Optimized gradient penalty"""
        batch_size = real_data.size(0)
        
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return penalty
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Optimized training"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_wd = []
        epoch_gp = []
        
        pbar = tqdm(dataloader, desc="Training", unit="batch", leave=False)
        
        for batch_idx, real_data in enumerate(pbar):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator (with fewer iterations)
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data
                d_real = self.discriminator(real_data)
                
                # Fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise).detach()
                d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss
                wasserstein_d = torch.mean(d_real) - torch.mean(d_fake)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                
                # Total loss
                d_loss = -wasserstein_d + self.lambda_gp * gp
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)  # Gradient clipping
                self.optim_D.step()
            
            # Train Generator
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise)
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)  # Gradient clipping
            self.optim_G.step()
            
            # Record metrics
            epoch_g_loss.append(g_loss.item())
            epoch_d_loss.append(d_loss.item())
            epoch_wd.append(-wasserstein_d.item())
            epoch_gp.append(gp.item())
            
            # Update progress bar less frequently
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'G': f'{g_loss.item():.3f}',
                    'D': f'{d_loss.item():.3f}',
                    'WD': f'{-wasserstein_d.item():.3f}'
                })
        
        return {
            'g_loss': np.mean(epoch_g_loss),
            'd_loss': np.mean(epoch_d_loss),
            'wasserstein_distance': np.mean(epoch_wd),
            'gradient_penalty': np.mean(epoch_gp)
        }
    
    def train(self, dataset: TrajectoryDataset, epochs: int = 30, batch_size: int = 128, 
              save_dir: str = "models"):
        """Optimized training with fewer epochs and larger batch size"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimized DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=2, pin_memory=True)
        
        print(f"Starting optimized WGAN-GP training:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches per epoch: {len(dataloader)}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        best_wd = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            # Record history
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Print metrics
            print(f"   G Loss: {metrics['g_loss']:.4f}")
            print(f"   D Loss: {metrics['d_loss']:.4f}")
            print(f"   Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
            print(f"   Gradient Penalty: {metrics['gradient_penalty']:.4f}")
            
            # Save best model
            if metrics['wasserstein_distance'] < best_wd:
                best_wd = metrics['wasserstein_distance']
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f"   Best model saved! WD: {best_wd:.4f}")
            
            # Less frequent checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_model(checkpoint_path)
        
        print("\nTraining completed!")
        self.plot_training_history(save_dir)
    
    def anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Simplified anomaly score (discriminator only)"""
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            
            # Use only discriminator score (faster)
            d_score = self.discriminator(trajectories)
            anomaly_scores = -d_score  # Invert: lower D score = more anomalous
            
        return anomaly_scores.cpu().numpy()
    
    def detect_anomalies(self, test_data: np.ndarray, threshold_percentile: float = 90) -> Dict:
        """Optimized anomaly detection"""
        print(f"Detecting anomalies in {len(test_data)} trajectories...")
        
        test_tensor = torch.FloatTensor(test_data)
        anomaly_scores = self.anomaly_score(test_tensor)
        
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        is_anomaly = anomaly_scores > threshold
        
        results = {
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'percentile_used': threshold_percentile
        }
        
        print(f"Detection completed:")
        print(f"   Threshold (percentile {threshold_percentile}): {threshold:.4f}")
        print(f"   Anomalies detected: {results['n_anomalies']}/{len(test_data)}")
        print(f"   Anomaly rate: {results['anomaly_rate']:.2%}")
        
        return results
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 90) -> Dict:
        """Optimized evaluation"""
        print(f"Evaluating model with {len(test_data)} trajectories...")
        
        test_tensor = torch.FloatTensor(test_data)
        anomaly_scores = self.anomaly_score(test_tensor)
        
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        predicted_labels = anomaly_scores > threshold
        
        # Metrics
        tp = np.sum((predicted_labels == True) & (true_labels == True))
        tn = np.sum((predicted_labels == False) & (true_labels == False))
        fp = np.sum((predicted_labels == True) & (true_labels == False))
        fn = np.sum((predicted_labels == False) & (true_labels == True))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(true_labels)
        
        try:
            auc_roc = roc_auc_score(true_labels, anomaly_scores)
            auc_pr = average_precision_score(true_labels, anomaly_scores)
        except:
            auc_roc = auc_pr = 0.0
        
        results = {
            'anomaly_scores': anomaly_scores,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
        
        print(f"Evaluation completed:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        
        return results
    
    def save_model(self, filepath: str):
        """Save model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'history': self.history,
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optim_G.load_state_dict(checkpoint['optim_G.state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_D.state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from: {filepath}")
    
    def plot_training_history(self, save_dir: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(self.history['g_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history['d_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history['wasserstein_distance'])
        axes[1, 0].set_title('Wasserstein Distance')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history['gradient_penalty'])
        axes[1, 1].set_title('Gradient Penalty')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
        plt.close()


def main():
    """Optimized main function"""
    
    # Optimized configurations
    DATA_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    ANOMALY_CSV = r"D:\UTFPR\TCC\AI-City Challenge\train-anomaly-results.csv"
    MODEL_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\wganGp"
    EPOCHS = 30          # Drastically reduced
    BATCH_SIZE = 128     # Increased for efficiency
    
    print("OPTIMIZED ANOMALY DETECTION SYSTEM WITH WGAN-GP")
    print("=" * 60)
    
    try:
        # Load labels
        anomaly_labels = None
        if os.path.exists(ANOMALY_CSV):
            anomaly_labels = AnomalyLabels(ANOMALY_CSV)
        
        # Optimized dataset
        print("\nLoading optimized dataset...")
        dataset = TrajectoryDataset(
            DATA_DIR, 
            anomaly_labels, 
            train_split=0.8,
            use_only_normal_videos=True,
            max_trajectories_per_video=300  # Limit to accelerate
        )
        
        # Optimized detector
        print("\nInitializing optimized WGAN-GP...")
        detector = WGANGPAnomalyDetector(seq_len=20, feature_dim=5)
        
        # Training
        print("\nStarting optimized training...")
        detector.train(
            dataset=dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            save_dir=MODEL_DIR
        )
        
        # Evaluation
        if anomaly_labels:
            print("\nEvaluating model...")
            val_data, val_labels = dataset.get_val_data()
            val_data_np = val_data.numpy()
            
            results = detector.evaluate_with_labels(val_data_np, val_labels)
            
            # Save results
            results_file = os.path.join(MODEL_DIR, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                              for k, v in results.items() 
                              if k not in ['anomaly_scores', 'predicted_labels', 'true_labels']}
                json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved in: {MODEL_DIR}")
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
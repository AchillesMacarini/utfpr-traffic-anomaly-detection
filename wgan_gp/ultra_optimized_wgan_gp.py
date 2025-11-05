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
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class OptimizedAnomalyLabels:
    """Optimized system for temporal anomaly analysis"""
    
    def __init__(self, csv_path: str, fps: float = 30.0):
        self.csv_path = csv_path
        self.fps = fps
        self.anomaly_intervals = {}
        self.video_stats = {}
        
        print(f"Loading optimized ground truth: {csv_path}")
        self._analyze_anomaly_patterns()
    
    def _analyze_anomaly_patterns(self):
        """Advanced analysis of anomaly patterns"""
        try:
            df = pd.read_csv(self.csv_path)
            
            for _, row in df.iterrows():
                video_id = int(row['video_id'])
                start_time = float(row['start_time'])
                end_time = float(row['end_time'])
                duration = end_time - start_time
                
                if video_id not in self.anomaly_intervals:
                    self.anomaly_intervals[video_id] = []
                    self.video_stats[video_id] = {
                        'total_anomaly_time': 0,
                        'anomaly_count': 0,
                        'avg_duration': 0,
                        'severity_score': 0
                    }
                
                self.anomaly_intervals[video_id].append((start_time, end_time, duration))
                
                # Update statistics
                stats = self.video_stats[video_id]
                stats['total_anomaly_time'] += duration
                stats['anomaly_count'] += 1
            
            # Calculate final metrics
            for video_id, stats in self.video_stats.items():
                if stats['anomaly_count'] > 0:
                    stats['avg_duration'] = stats['total_anomaly_time'] / stats['anomaly_count']
                    # Severity score based on duration and frequency
                    stats['severity_score'] = np.log1p(stats['total_anomaly_time']) * stats['anomaly_count']
            
            print(f"Analysis complete: {len(self.anomaly_intervals)} anomalous videos")
            
            # Detailed statistics
            total_anomalies = sum(len(intervals) for intervals in self.anomaly_intervals.values())
            avg_duration = np.mean([dur for intervals in self.anomaly_intervals.values() 
                                  for _, _, dur in intervals])
            
            print(f"Statistics:")
            print(f"   Total anomalies: {total_anomalies}")
            print(f"   Average duration: {avg_duration:.1f}s")
            print(f"   Most severe videos: {self.get_high_severity_videos()[:5]}")
            
        except Exception as e:
            print(f"Error analyzing anomalies: {e}")
    
    def has_anomalies(self, video_id: int) -> bool:
        return video_id in self.anomaly_intervals
    
    def get_severity_score(self, video_id: int) -> float:
        return self.video_stats.get(video_id, {}).get('severity_score', 0.0)
    
    def get_high_severity_videos(self) -> List[int]:
        return sorted(self.video_stats.keys(), 
                     key=lambda v: self.video_stats[v]['severity_score'], reverse=True)


class AdvancedTrajectoryDataset(Dataset):
    """Ultra-optimized dataset with quality analysis - CORRECTED"""
    
    def __init__(self, data_dir: str, anomaly_labels: OptimizedAnomalyLabels = None, 
                 balance_ratio: float = 0.35, quality_threshold: float = 0.01):  # Much smaller threshold
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.balance_ratio = balance_ratio
        self.quality_threshold = quality_threshold
        
        # Robust scaler
        self.scaler = RobustScaler()
        
        self._load_and_optimize_data()
    
    def _load_and_optimize_data(self):
        """Optimized loading with quality analysis - DIAGNOSTICS INCLUDED"""
        
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"Loading {len(npy_files)} files with quality analysis...")
        
        all_trajectories = []
        trajectory_labels = []
        trajectory_qualities = []
        video_sources = []
        
        # Diagnostic statistics
        total_files = 0
        files_with_data = 0
        total_trajectories = 0
        trajectories_after_quality = 0
        quality_stats = []
        
        # First pass: collect data and calculate quality
        for npy_file in tqdm(npy_files, desc="Analyzing files"):
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            total_files += 1
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) > 0:
                    files_with_data += 1
                    total_trajectories += len(trajectories)
                    
                    print(f"   File {npy_file}: {len(trajectories)} trajectories, shape: {trajectories.shape}")
                    
                    # Trajectory quality analysis
                    qualities = self._analyze_trajectory_quality(trajectories)
                    quality_stats.extend(qualities)
                    
                    print(f"      Quality - Min: {np.min(qualities):.4f}, Max: {np.max(qualities):.4f}, Mean: {np.mean(qualities):.4f}")
                    
                    # Filter by quality with adaptive threshold
                    if len(qualities) > 0:
                        # If threshold too restrictive, use percentile
                        adaptive_threshold = min(self.quality_threshold, np.percentile(qualities, 20))
                        high_quality_mask = qualities > adaptive_threshold
                        
                        print(f"      Threshold used: {adaptive_threshold:.4f}, Passed filter: {np.sum(high_quality_mask)}/{len(qualities)}")
                        
                        if np.any(high_quality_mask):
                            filtered_trajectories = trajectories[high_quality_mask]
                            filtered_qualities = qualities[high_quality_mask]
                            trajectories_after_quality += len(filtered_trajectories)
                            
                            # Determine labels
                            is_anomalous_video = (self.anomaly_labels and 
                                                self.anomaly_labels.has_anomalies(video_id))
                            
                            # Add to collection
                            all_trajectories.append(filtered_trajectories)
                            trajectory_labels.extend([is_anomalous_video] * len(filtered_trajectories))
                            trajectory_qualities.extend(filtered_qualities)
                            video_sources.extend([video_id] * len(filtered_trajectories))
                            
                            print(f"      Added {len(filtered_trajectories)} trajectories (Anomalous: {is_anomalous_video})")
                        else:
                            print(f"      No trajectories passed quality filter")
                    else:
                        print(f"      Error in quality calculation")
                else:
                    print(f"   File {npy_file}: empty file")
                        
            except Exception as e:
                print(f"Error processing {npy_file}: {e}")
        
        # Final diagnostics
        print(f"\nCOMPLETE DIAGNOSTICS:")
        print(f"   Total files: {total_files}")
        print(f"   Files with data: {files_with_data}")
        print(f"   Total raw trajectories: {total_trajectories}")
        print(f"   Trajectories after filter: {trajectories_after_quality}")
        
        if quality_stats:
            print(f"   Overall quality - Min: {np.min(quality_stats):.4f}, Max: {np.max(quality_stats):.4f}")
            print(f"   Overall quality - Mean: {np.mean(quality_stats):.4f}, Std: {np.std(quality_stats):.4f}")
            print(f"   Percentiles: 10%={np.percentile(quality_stats, 10):.4f}, 50%={np.percentile(quality_stats, 50):.4f}, 90%={np.percentile(quality_stats, 90):.4f}")
        
        # Concatenate all data
        if all_trajectories:
            self.all_trajectories = np.concatenate(all_trajectories, axis=0)
            self.all_labels = np.array(trajectory_labels, dtype=bool)
            self.all_qualities = np.array(trajectory_qualities)
            self.all_video_sources = np.array(video_sources)
            
            print(f"\nData collected successfully:")
            print(f"   Final shape: {self.all_trajectories.shape}")
            print(f"   Normal labels: {np.sum(~self.all_labels)}")
            print(f"   Anomalous labels: {np.sum(self.all_labels)}")
            
            # Robust normalization
            self._apply_robust_normalization()
            
            # Stratified split
            self._create_stratified_split()
            
            print(f"Optimized dataset loaded:")
            print(f"   Total trajectories: {len(self.all_trajectories)}")
            print(f"   Average quality: {np.mean(self.all_qualities):.3f}")
            print(f"   Training: {len(self.train_trajectories)} trajectories")
            print(f"   Validation: {len(self.val_trajectories)} (Normal: {np.sum(~self.val_labels)}, Anomalous: {np.sum(self.val_labels)})")
        else:
            # Error diagnostics
            print(f"\nERROR: No trajectories passed the filter!")
            print(f"   Possible causes:")
            print(f"   1. Threshold too high ({self.quality_threshold})")
            print(f"   2. Problem in quality calculation")
            print(f"   3. Invalid input data")
            print(f"   4. Corrupted .npy files")
            
            # Try with zero threshold (no filter)
            print(f"\nTrying to load WITHOUT quality filter...")
            self._load_without_quality_filter()
    
    def _load_without_quality_filter(self):
        """Emergency loading without quality filter"""
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        all_trajectories = []
        trajectory_labels = []
        
        for npy_file in tqdm(npy_files[:10], desc="Emergency loading"):  # Only first 10 files
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) > 0:
                    # Check if data is valid
                    if not np.any(np.isnan(trajectories)) and not np.any(np.isinf(trajectories)):
                        # Limit quantity per file
                        max_per_file = 100
                        if len(trajectories) > max_per_file:
                            indices = np.random.choice(len(trajectories), max_per_file, replace=False)
                            trajectories = trajectories[indices]
                        
                        is_anomalous_video = (self.anomaly_labels and 
                                            self.anomaly_labels.has_anomalies(video_id))
                        
                        all_trajectories.append(trajectories)
                        trajectory_labels.extend([is_anomalous_video] * len(trajectories))
                        
                        print(f"   {npy_file}: {len(trajectórias)} trajetórias (Anômalo: {is_anomalous_video})")
                    else:
                        print(f"   {npy_file}: contém NaN/Inf")
                        
            except Exception as e:
                print(f"   {npy_file}: {e}")
        
        if all_trajectories:
            self.all_trajectories = np.concatenate(all_trajectories, axis=0)
            self.all_labels = np.array(trajectory_labels, dtype=bool)
            self.all_qualities = np.ones(len(self.all_trajectories))  # Qualidade uniforme
            self.all_video_sources = np.zeros(len(self.all_trajectórias), dtype=int)

            print(f"🔧 Carregamento de emergência bem-sucedido: {len(self.all_trajectórias)} trajetórias")

            # Aplicar normalização
            self._apply_robust_normalization()
            self._create_stratified_split()
            
        else:
            raise ValueError("Falha completa no carregamento! Verifique os dados de entrada.")
    
    def _analyze_trajectory_quality(self, trajectories: np.ndarray) -> np.ndarray:
        """More robust quality analysis"""
        qualities = []
        
        for traj in trajectories:
            try:
                # Check if trajectory is valid
                if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                    qualities.append(0.0)
                    continue
                
                # 1. Spatial variability (non-trivial movement)
                spatial_var = np.var(traj[:, :2]) if traj.shape[1] >= 2 else 0.0
                spatial_var = max(0.0, min(spatial_var, 10.0))  # Clip extreme values
                
                # 2. Temporal consistency (if velocity exists)
                if traj.shape[1] >= 3:
                    velocity_var = np.var(traj[:, 2])
                    velocity_consistency = 1.0 / (1.0 + velocity_var) if velocity_var > 0 else 1.0
                else:
                    velocity_consistency = 1.0
                
                # 3. Penalty for extreme values (normalized)
                abs_values = np.abs(traj)
                extremes_ratio = np.sum(abs_values > 5) / traj.size  # More permissive threshold
                extremes_penalty = max(0.0, min(extremes_ratio, 1.0))
                
                # 4. Penalty for constant values
                constant_penalty = 0.0
                for col in range(traj.shape[1]):
                    if np.std(traj[:, col]) < 1e-6:  # Column practically constant
                        constant_penalty += 0.2
                
                # Combined quality score (always positive)
                base_quality = spatial_var * velocity_consistency
                penalties = extremes_penalty + constant_penalty
                
                quality = max(0.001, base_quality * (1 - min(penalties, 0.9)))  # Minimum 0.001
                qualities.append(quality)
                
            except Exception as e:
                print(f"      Error calculating trajectory quality: {e}")
                qualities.append(0.001)  # Minimum quality on error
        
        return np.array(qualities)
    
    def _apply_robust_normalization(self):
        """Robust normalization using RobustScaler"""
        original_shape = self.all_trajectories.shape
        
        # Reshape for normalization
        trajectories_flat = self.all_trajectories.reshape(-1, original_shape[-1])
        
        # Apply RobustScaler
        normalized_flat = self.scaler.fit_transform(trajectories_flat)
        
        # Reshape back
        self.all_trajectories = normalized_flat.reshape(original_shape)
        
        print(f"Robust normalization applied")
    
    def _create_stratified_split(self):
        """Balanced stratified split"""
        normal_indices = np.where(~self.all_labels)[0]
        anomaly_indices = np.where(self.all_labels)[0]
        
        # Training: only normal trajectories (80%)
        n_train_normal = int(len(normal_indices) * 0.8)
        np.random.seed(42)
        normal_train_idx = np.random.choice(normal_indices, n_train_normal, replace=False)
        
        self.train_trajectories = self.all_trajectories[normal_train_idx]
        
        # Balanced validation
        normal_val_idx = np.setdiff1d(normal_indices, normal_train_idx)
        n_val_normal = len(normal_val_idx)
        n_val_anomaly = int(n_val_normal * self.balance_ratio / (1 - self.balance_ratio))
        
        if len(anomaly_indices) > 0:
            # Select high-quality anomalies
            anomaly_qualities = self.all_qualities[anomaly_indices]
            top_anomaly_idx = anomaly_indices[np.argsort(anomaly_qualities)[-n_val_anomaly:]]
            
            val_indices = np.concatenate([normal_val_idx, top_anomaly_idx])
        else:
            val_indices = normal_val_idx
        
        self.val_trajectories = self.all_trajectories[val_indices]
        self.val_labels = self.all_labels[val_indices]
        
        # Shuffle validation
        val_shuffle = np.random.permutation(len(val_indices))
        self.val_trajectories = self.val_trajectories[val_shuffle]
        self.val_labels = self.val_labels[val_shuffle]
    
    def get_train_data(self):
        return torch.FloatTensor(self.train_trajectories)
    
    def get_val_data(self):
        return torch.FloatTensor(self.val_trajectories), self.val_labels
    
    def __len__(self):
        return len(self.train_trajectories)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.train_trajectories[idx])


class UltraGenerator(nn.Module):
    """Ultra-optimized generator with attention and skip connections"""
    
    def __init__(self, noise_dim: int = 256, seq_len: int = 20, feature_dim: int = 5):
        super(UltraGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Noise encoder with skip connections
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Feature decoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, 1024))
        
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Encode noise
        encoded = self.noise_encoder(noise)  # [batch, 1024]
        
        # Expand to sequence with positional encoding
        sequence = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, 1024]
        sequence = sequence + self.pos_encoding
        
        # Apply transformer
        transformed = self.transformer(sequence)  # [batch, seq_len, 1024]
        
        # Decode to features
        output = self.feature_decoder(transformed.reshape(-1, 1024))  # [batch*seq_len, feature_dim]
        
        return output.view(batch_size, self.seq_len, self.feature_dim)


class UltraDiscriminator(nn.Module):
    """Ultra-optimized discriminator with hybrid architecture"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(UltraDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Multi-scale convolutions
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, 128, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for k in [3, 5, 7]
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Transformer for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, trajectories):
        batch_size = trajectories.size(0)
        
        # Multi-scale convolutions
        x = trajectories.transpose(1, 2)  # [batch, feature_dim, seq_len]
        
        conv_outputs = []
        for conv_branch in self.conv_branches:
            conv_out = conv_branch(x)
            conv_outputs.append(conv_out)
        
        # Fuse multi-scale features
        fused = torch.cat(conv_outputs, dim=1)  # [batch, 384, seq_len]
        fused = self.fusion(fused)  # [batch, 512, seq_len]
        
        # Transpose for transformer
        fused = fused.transpose(1, 2)  # [batch, seq_len, 512]
        
        # Apply transformer
        transformed = self.transformer(fused)  # [batch, seq_len, 512]
        
        # Attention pooling
        query = torch.mean(transformed, dim=1, keepdim=True)  # [batch, 1, 512]
        attended, _ = self.attention_pooling(query, transformed, transformed)  # [batch, 1, 512]
        
        # Classification
        output = self.classifier(attended.squeeze(1))  # [batch, 1]
        
        return output.squeeze()


class UltraWGANGPDetector:
    """Ultra-optimized anomaly detection system"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Ultra-optimized hyperparameters
        self.noise_dim = 256
        self.lr_g = 0.00005
        self.lr_d = 0.0002
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 10
        self.n_critic = 3
        
        # Networks
        self.generator = UltraGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = UltraDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Optimizers with scheduler
        self.optim_G = optim.AdamW(self.generator.parameters(), lr=self.lr_g, 
                                  betas=(self.beta1, self.beta2), weight_decay=1e-6)
        self.optim_D = optim.AdamW(self.discriminator.parameters(), lr=self.lr_d, 
                                  betas=(self.beta1, self.beta2), weight_decay=1e-6)
        
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optim_G, T_max=50)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optim_D, T_max=50)
        
        self.history = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        print(f"Ultra WGAN-GP initialized on device: {self.device}")
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Ultra-stable gradient penalty"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Multiple interpolations for stability
        penalties = []
        for _ in range(2):
            alpha = torch.rand(batch_size, 1, 1).to(device)
            
            # Interpolation with small noise
            epsilon = torch.randn_like(real_data) * 0.001
            interpolated = alpha * real_data + (1 - alpha) * fake_data + epsilon
            interpolated.requires_grad_(True)
            
            # Forward pass
            d_interpolated = self.discriminator(interpolated)
            
            # Gradients
            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interpolated).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Penalty
            gradients_flat = gradients.reshape(batch_size, -1)
            gradient_norm = gradients_flat.norm(2, dim=1)
            penalty = torch.mean((gradient_norm - 1) ** 2)
            penalties.append(penalty)
        
        return torch.mean(torch.stack(penalties))
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Ultra-stable training"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        for batch_idx, real_data in enumerate(tqdm(dataloader, desc="Ultra-Training", leave=False)):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            d_losses = []
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data with small noise
                noise_factor = 0.01 * (1 - batch_idx / len(dataloader))  # Decay during epoch
                real_noisy = real_data + torch.randn_like(real_data) * noise_factor
                
                d_real = self.discriminator(real_noisy)
                
                # Fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                with torch.no_grad():
                    fake_data = self.generator(noise)
                
                d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss
                wasserstein_d = torch.mean(d_real) - torch.mean(d_fake)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                
                # Total loss
                d_loss = -wasserstein_d + self.lambda_gp * gp
                d_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optim_D.step()
                
                d_losses.append(d_loss.item())
            
            # Train Generator
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise)
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            self.optim_G.step()
            
            # Record metrics
            epoch_metrics['g_loss'].append(g_loss.item())
            epoch_metrics['d_loss'].append(np.mean(d_losses))
            epoch_metrics['wasserstein_distance'].append(-wasserstein_d.item())
            epoch_metrics['gradient_penalty'].append(gp.item())
        
        # Update schedulers
        avg_g_loss = np.mean(epoch_metrics['g_loss'])
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def train(self, dataset: AdvancedTrajectoryDataset, epochs: int = 60, 
              batch_size: int = 32, save_dir: str = "ultra_models"):
        """Ultra-complete training"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=4, pin_memory=True)
        
        print(f"Starting Ultra-Training:")
        print(f"   Epochs: {epochs} | Batch: {batch_size} | Batches/epoch: {len(dataloader)}")
        
        best_wd = float('inf')
        patience = 0
        max_patience = 15
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            current_wd = abs(metrics['wasserstein_distance'])
            
            print(f"   G: {metrics['g_loss']:.4f} | D: {metrics['d_loss']:.4f}")
            print(f"   WD: {metrics['wasserstein_distance']:.4f} | GP: {metrics['gradient_penalty']:.4f}")
            print(f"   LR_G: {self.optim_G.param_groups[0]['lr']:.2e} | LR_D: {self.optim_D.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if current_wd < best_wd:
                best_wd = current_wd
                patience = 0
                self.save_model(os.path.join(save_dir, 'ultra_best_model.pth'))
                print(f"   BEST MODEL! WD: {best_wd:.4f}")
            else:
                patience += 1
            
            # Early stopping
            if patience >= max_patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
            
            # Checkpoint
            if (epoch + 1) % 15 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_e{epoch+1}.pth'))
        
        print(f"\nTraining completed! Best WD: {best_wd:.4f}")
    
    def ultra_anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Ultra-advanced scoring system"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Discriminator score with multiple attempts
            disc_scores = []
            for _ in range(5):
                d_score = self.discriminator(trajectories)
                disc_scores.append(-d_score.cpu())
            
            final_disc_score = torch.mean(torch.stack(disc_scores), dim=0)
            
            # 2. Advanced reconstruction score
            reconstruction_scores = []
            n_generations = 10
            
            all_fakes = []
            for _ in range(n_generations):
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_batch = self.generator(noise)
                all_fakes.append(fake_batch.cpu())
            
            all_fakes = torch.cat(all_fakes, dim=0)
            
            for i in range(batch_size):
                real_traj = trajectories[i].cpu()
                
                # Multiple distance metrics
                l2_distances = torch.sum((all_fakes - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                l1_distances = torch.sum(torch.abs(all_fakes - real_traj.unsqueeze(0)), dim=(1, 2))
                
                # Combined distance
                combined_distances = 0.7 * l2_distances + 0.3 * l1_distances
                min_distance = torch.min(combined_distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_scores = torch.tensor(reconstruction_scores)
            
            # 3. Temporal complexity score
            complexity_scores = []
            for i in range(batch_size):
                traj = trajectories[i].cpu()
                
                # Temporal variation
                temporal_diffs = torch.diff(traj, dim=0)
                complexity = torch.var(temporal_diffs, dim=0).mean()
                
                # Sudden changes
                sudden_changes = torch.sum(torch.abs(temporal_diffs) > 2 * torch.std(temporal_diffs))
                
                complexity_score = complexity + 0.1 * sudden_changes
                complexity_scores.append(complexity_score.item())
            
            complexity_scores = torch.tensor(complexity_scores)
            
            # 4. Ultra-intelligent combination
            def robust_normalize(scores):
                if len(scores) <= 1:
                    return torch.zeros_like(scores)
                
                q25, q75 = torch.quantile(scores, 0.25), torch.quantile(scores, 0.75)
                iqr = q75 - q25 + 1e-8
                
                normalized = (scores - q25) / iqr
                return torch.clamp(normalized, -3, 3)
            
            disc_norm = robust_normalize(final_disc_score)
            recon_norm = robust_normalize(reconstruction_scores)
            complex_norm = robust_normalize(complexity_scores)
            
            # Adaptive weights
            weights = torch.softmax(torch.tensor([
                torch.var(disc_norm) + 1e-8,
                torch.var(recon_norm) + 1e-8,
                torch.var(complex_norm) + 1e-8
            ]), dim=0)
            
            # Final score
            ultra_score = (weights[0] * disc_norm + 
                          weights[1] * recon_norm + 
                          weights[2] * complex_norm)
            
            # Final smoothing
            final_score = torch.tanh(ultra_score * 0.5)
        
        return final_score.numpy()
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 80) -> Dict:
        """Ultra-complete evaluation"""
        print(f"Ultra-Evaluation with {len(test_data)} trajectories...")
        
        test_tensor = torch.FloatTensor(test_data)
        ultra_scores = self.ultra_anomaly_score(test_tensor)
        
        threshold = np.percentile(ultra_scores, threshold_percentile)
        predicted_labels = ultra_scores > threshold
        
        # Detailed metrics
        tp = int(np.sum((predicted_labels == True) & (true_labels == True)))
        tn = int(np.sum((predicted_labels == False) & (true_labels == False)))
        fp = int(np.sum((predicted_labels == True) & (true_labels == False)))
        fn = int(np.sum((predicted_labels == False) & (true_labels == True)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels)
        
        try:
            auc_roc = roc_auc_score(true_labels, ultra_scores)
            auc_pr = average_precision_score(true_labels, ultra_scores)
        except:
            auc_roc = auc_pr = 0.0
        
        results = {
            'ultra_scores': ultra_scores,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        print(f"Ultra-Evaluation:")
        print(f"   F1-Score: {f1_score:.3f} (target: >0.50)")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f} (target: >0.75)")
        
        return results
    
    def save_model(self, filepath: str):
        """Save complete model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'history': self.history,
            'hyperparameters': {
                'noise_dim': self.noise_dim,
                'lr_g': self.lr_g,
                'lr_d': self.lr_d,
                'lambda_gp': self.lambda_gp
            }
        }, filepath)


class StableUltraDiscriminator(nn.Module):
    """Ultra-stable discriminator with improved normalization"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(StableUltraDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # More stable convolutions
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=k, padding=k//2),  # Reduced from 128 to 64
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Dropout(0.1)
            ) for k in [3, 5, 7]
        ])
        
        # More conservative fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # Reduced
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Simple LSTM instead of Transformer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,  # Reduced from 3 to 2
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 (bidirectional)
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        
        # Conservative initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very low gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, trajectories):
        batch_size = trajectories.size(0)
        
        # Input clipping for stability
        trajectories = torch.clamp(trajectories, -5, 5)
        
        # Multi-scale convolutions
        x = trajectories.transpose(1, 2)  # [batch, feature_dim, seq_len]
        
        conv_outputs = []
        for conv_branch in self.conv_branches:
            conv_out = conv_branch(x)
            conv_outputs.append(conv_out)
        
        # Fuse multi-scale features
        fused = torch.cat(conv_outputs, dim=1)  # [batch, 192, seq_len]
        fused = self.fusion(fused)  # [batch, 256, seq_len]
        
        # Transpose for LSTM
        fused = fused.transpose(1, 2)  # [batch, seq_len, 256]
        
        # LSTM
        lstm_out, _ = self.lstm(fused)  # [batch, seq_len, 256]
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # [batch, 256]
        
        # Classification
        output = self.classifier(pooled)  # [batch, 1]
        
        return output.squeeze()


class StableUltraGenerator(nn.Module):
    """Ultra-stable generator"""
    
    def __init__(self, noise_dim: int = 128, seq_len: int = 20, feature_dim: int = 5):  # Reduced noise dim
        super(StableUltraGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Simpler encoder
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # Simple LSTM instead of Transformer
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # Decoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, feature_dim),
            nn.Tanh()
        )
        
        # Conservative initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Encode noise
        encoded = self.noise_encoder(noise)  # [batch, 512]
        
        # Expand to sequence
        sequence = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, 512]
        
        # LSTM
        lstm_out, _ = self.lstm(sequence)  # [batch, seq_len, 256]
        
        # Decode to features
        output = self.feature_decoder(lstm_out.reshape(-1, 256))  # [batch*seq_len, feature_dim]
        
        return output.view(batch_size, self.seq_len, self.feature_dim)


class StableWGANGPDetector:
    """Ultra-stable detector"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Ultra-conservative hyperparameters
        self.noise_dim = 128  # Reduced
        self.lr_g = 0.00001   # Much lower
        self.lr_d = 0.00005   # Much lower
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 1.0  # DRASTICALLY REDUCED from 10 to 1
        self.n_critic = 2     # Reduced
        
        # Stable networks
        self.generator = StableUltraGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = StableUltraDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Conservative optimizers
        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lr_g, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        
        # Smoother schedulers
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optim_G, gamma=0.99)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optim_D, gamma=0.99)
        
        self.history = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        print(f"Ultra-Stable WGAN-GP initialized:")
        print(f"   Device: {self.device}")
        print(f"   LR_G: {self.lr_g:.2e}, LR_D: {self.lr_d:.2e}")
        print(f"   Lambda GP: {self.lambda_gp} (reduced for stability)")
    
    def stable_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Ultra-stable gradient penalty"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Preventive clipping
        real_data = torch.clamp(real_data, -3, 3)
        fake_data = torch.clamp(fake_data, -3, 3)
        
        # Simple more stable interpolation
        alpha = torch.rand(batch_size, 1, 1).to(device)
        
        # Interpolation without additional noise
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Forward pass
        d_interpolated = self.discriminator(interpolated)
        
        # Gradients with verification
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Penalty with clipping
        gradients_flat = gradients.reshape(batch_size, -1)
        gradient_norm = gradients_flat.norm(2, dim=1)
        
        # Clipping norm
        gradient_norm = torch.clamp(gradient_norm, 0, 10)
        
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        # Final penalty clipping
        penalty = torch.clamp(penalty, 0, 100)
        
        return penalty
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Ultra-stable training"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        for batch_idx, real_data in enumerate(tqdm(dataloader, desc="Stable-Training", leave=False)):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Input clipping
            real_data = torch.clamp(real_data, -3, 3)
            
            # Train Discriminator (less times)
            d_losses = []
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data with minimal noise
                noise_factor = 0.001  # Much reduced
                real_noisy = real_data + torch.randn_like(real_data) * noise_factor
                real_noisy = torch.clamp(real_noisy, -3, 3)
                
                d_real = self.discriminator(real_noisy)
                
                # Fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                noise = torch.clamp(noise, -2, 2)  # Noise clipping
                
                with torch.no_grad():
                    fake_data = self.generator(noise)
                    fake_data = torch.clamp(fake_data, -3, 3)
                
                d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss with clipping
                d_real_mean = torch.clamp(torch.mean(d_real), -100, 100)
                d_fake_mean = torch.clamp(torch.mean(d_fake), -100, 100)
                wasserstein_d = d_real_mean - d_fake_mean
                
                # Stable gradient penalty
                gp = self.stable_gradient_penalty(real_data, fake_data)
                
                # Total loss with clipping
                d_loss = -wasserstein_d + self.lambda_gp * gp
                d_loss = torch.clamp(d_loss, -1000, 1000)
                
                # Check if loss is valid
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    
                    # Aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.1)
                    
                    self.optim_D.step()
                else:
                    print(f"Invalid D loss detected: {d_loss.item()}")
                    continue
                
                d_losses.append(d_loss.item())
            
            # Train Generator
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            noise = torch.clamp(noise, -2, 2)
            
            fake_data = self.generator(noise)
            fake_data = torch.clamp(fake_data, -3, 3)
            
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            g_loss = torch.clamp(g_loss, -1000, 1000)
            
            # Check if loss is valid
            if torch.isfinite(g_loss):
                g_loss.backward()
                
                # Aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1)
                
                self.optim_G.step()
            else:
                print(f"Invalid G loss detected: {g_loss.item()}")
                continue
            
            # Record metrics
            if len(d_losses) > 0:
                epoch_metrics['g_loss'].append(g_loss.item())
                epoch_metrics['d_loss'].append(np.mean(d_losses))
                epoch_metrics['wasserstein_distance'].append(-wasserstein_d.item())
                epoch_metrics['gradient_penalty'].append(gp.item())
        
        # Update schedulers smoothly
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
    
    def train(self, dataset: AdvancedTrajectoryDataset, epochs: int = 30,  # Reduced
              batch_size: int = 16, save_dir: str = "stable_models"):  # Smaller batch
        """Ultra-stable training"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=2, pin_memory=True)  # Reduced workers
        
        print(f"Starting Ultra-Stable Training:")
        print(f"   Epochs: {epochs} | Batch: {batch_size} | Batches/epoch: {len(dataloader)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            # Check stability
            is_stable = all(
                abs(v) < 1000 for v in [
                    metrics['g_loss'], 
                    metrics['d_loss'], 
                    metrics['wasserstein_distance'],
                    metrics['gradient_penalty']
                ]
            )
            
            if not is_stable:
                print(f"INSTABILITY DETECTED! Reducing learning rates...")
                for param_group in self.optim_G.param_groups:
                    param_group['lr'] *= 0.5
                for param_group in self.optim_D.param_groups:
                    param_group['lr'] *= 0.5
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            print(f"   G: {metrics['g_loss']:.4f} | D: {metrics['d_loss']:.4f}")
            print(f"   WD: {metrics['wasserstein_distance']:.4f} | GP: {metrics['gradient_penalty']:.4f}")
            print(f"   LR_G: {self.optim_G.param_groups[0]['lr']:.2e} | LR_D: {self.optim_D.param_groups[0]['lr']:.2e}")
            print(f"   Stable: {'Yes' if is_stable else 'No'}")
            
            # Checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_dir, f'stable_checkpoint_e{epoch+1}.pth'))
        
        print(f"\nStable training completed!")
    
    def ultra_anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Ultra-advanced scoring system"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Discriminator score with multiple attempts
            disc_scores = []
            for _ in range(5):
                d_score = self.discriminator(trajectories)
                disc_scores.append(-d_score.cpu())
            
            final_disc_score = torch.mean(torch.stack(disc_scores), dim=0)
            
            # 2. Advanced reconstruction score
            reconstruction_scores = []
            n_generations = 10
            
            all_fakes = []
            for _ in range(n_generations):
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_batch = self.generator(noise)
                all_fakes.append(fake_batch.cpu())
            
            all_fakes = torch.cat(all_fakes, dim=0)
            
            for i in range(batch_size):
                real_traj = trajectories[i].cpu()
                
                # Multiple distance metrics
                l2_distances = torch.sum((all_fakes - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                l1_distances = torch.sum(torch.abs(all_fakes - real_traj.unsqueeze(0)), dim=(1, 2))
                
                # Combined distance
                combined_distances = 0.7 * l2_distances + 0.3 * l1_distances
                min_distance = torch.min(combined_distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_scores = torch.tensor(reconstruction_scores)
            
            # 3. Temporal complexity score
            complexity_scores = []
            for i in range(batch_size):
                traj = trajectories[i].cpu()
                
                # Temporal variation
                temporal_diffs = torch.diff(traj, dim=0)
                complexity = torch.var(temporal_diffs, dim=0).mean()
                
                # Sudden changes
                sudden_changes = torch.sum(torch.abs(temporal_diffs) > 2 * torch.std(temporal_diffs))
                
                complexity_score = complexity + 0.1 * sudden_changes
                complexity_scores.append(complexity_score.item())
            
            complexity_scores = torch.tensor(complexity_scores)
            
            # 4. Ultra-intelligent combination
            def robust_normalize(scores):
                if len(scores) <= 1:
                    return torch.zeros_like(scores)
                
                q25, q75 = torch.quantile(scores, 0.25), torch.quantile(scores, 0.75)
                iqr = q75 - q25 + 1e-8
                
                normalized = (scores - q25) / iqr
                return torch.clamp(normalized, -3, 3)
            
            disc_norm = robust_normalize(final_disc_score)
            recon_norm = robust_normalize(reconstruction_scores)
            complex_norm = robust_normalize(complexity_scores)
            
            # Adaptive weights
            weights = torch.softmax(torch.tensor([
                torch.var(disc_norm) + 1e-8,
                torch.var(recon_norm) + 1e-8,
                torch.var(complex_norm) + 1e-8
            ]), dim=0)
            
            # Final score
            ultra_score = (weights[0] * disc_norm + 
                          weights[1] * recon_norm + 
                          weights[2] * complex_norm)
            
            # Final smoothing
            final_score = torch.tanh(ultra_score * 0.5)
        
        return final_score.numpy()
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 80) -> Dict:
        """Ultra-complete evaluation"""
        print(f"Ultra-Evaluation with {len(test_data)} trajectories...")
        
        test_tensor = torch.FloatTensor(test_data)
        ultra_scores = self.ultra_anomaly_score(test_tensor)
        
        threshold = np.percentile(ultra_scores, threshold_percentile)
        predicted_labels = ultra_scores > threshold
        
        # Detailed metrics
        tp = int(np.sum((predicted_labels == True) & (true_labels == True)))
        tn = int(np.sum((predicted_labels == False) & (true_labels == False)))
        fp = int(np.sum((predicted_labels == True) & (true_labels == False)))
        fn = int(np.sum((predicted_labels == False) & (true_labels == True)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels)
        
        try:
            auc_roc = roc_auc_score(true_labels, ultra_scores)
            auc_pr = average_precision_score(true_labels, ultra_scores)
        except:
            auc_roc = auc_pr = 0.0
        
        results = {
            'ultra_scores': ultra_scores,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        print(f"Ultra-Evaluation:")
        print(f"   F1-Score: {f1_score:.3f} (target: >0.50)")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f} (target: >0.75)")
        
        return results
    
    def save_model(self, filepath: str):
        """Save complete model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'history': self.history,
            'hyperparameters': {
                'noise_dim': self.noise_dim,
                'lr_g': self.lr_g,
                'lr_d': self.lr_d,
                'lambda_gp': self.lambda_gp
            }
        }, filepath)


class StableUltraDiscriminator(nn.Module):
    """Ultra-stable discriminator with improved normalization"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(StableUltraDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # More stable convolutions
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=k, padding=k//2),  # Reduced from 128 to 64
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),  # Changed to LeakyReLU
                nn.Dropout(0.1)
            ) for k in [3, 5, 7]
        ])
        
        # More conservative fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # Reduced
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Simple LSTM instead of Transformer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,  # Reduced from 3 to 2
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 (bidirectional)
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        
        # Conservative initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very low gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, trajectories):
        batch_size = trajectories.size(0)
        
        # Input clipping for stability
        trajectories = torch.clamp(trajectories, -5, 5)
        
        # Multi-scale convolutions
        x = trajectories.transpose(1, 2)  # [batch, feature_dim, seq_len]
        
        conv_outputs = []
        for conv_branch in self.conv_branches:
            conv_out = conv_branch(x)
            conv_outputs.append(conv_out)
        
        # Fuse multi-scale features
        fused = torch.cat(conv_outputs, dim=1)  # [batch, 192, seq_len]
        fused = self.fusion(fused)  # [batch, 256, seq_len]
        
        # Transpose for LSTM
        fused = fused.transpose(1, 2)  # [batch, seq_len, 256]
        
        # LSTM
        lstm_out, _ = self.lstm(fused)  # [batch, seq_len, 256]
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # [batch, 256]
        
        # Classification
        output = self.classifier(pooled)  # [batch, 1]
        
        return output.squeeze()


class StableUltraGenerator(nn.Module):
    """Ultra-stable generator"""
    
    def __init__(self, noise_dim: int = 128, seq_len: int = 20, feature_dim: int = 5):  # Reduced noise dim
        super(StableUltraGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Simpler encoder
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # Simple LSTM instead of Transformer
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # Decoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, feature_dim),
            nn.Tanh()
        )
        
        # Conservative initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Encode noise
        encoded = self.noise_encoder(noise)  # [batch, 512]
        
        # Expand to sequence
        sequence = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, 512]
        
        # LSTM
        lstm_out, _ = self.lstm(sequence)  # [batch, seq_len, 256]
        
        # Decode to features
        output = self.feature_decoder(lstm_out.reshape(-1, 256))  # [batch*seq_len, feature_dim]
        
        return output.view(batch_size, self.seq_len, self.feature_dim)


class StableWGANGPDetector:
    """Ultra-stable detector"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Ultra-conservative hyperparameters
        self.noise_dim = 128  # Reduced
        self.lr_g = 0.00001   # Much lower
        self.lr_d = 0.00005   # Much lower
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 1.0  # DRASTICALLY REDUCED from 10 to 1
        self.n_critic = 2     # Reduced
        
        # Stable networks
        self.generator = StableUltraGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = StableUltraDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Conservative optimizers
        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lr_g, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        
        # Smoother schedulers
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optim_G, gamma=0.99)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optim_D, gamma=0.99)
        
        self.history = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        print(f"Ultra-Stable WGAN-GP initialized:")
        print(f"   Device: {self.device}")
        print(f"   LR_G: {self.lr_g:.2e}, LR_D: {self.lr_d:.2e}")
        print(f"   Lambda GP: {self.lambda_gp} (reduced for stability)")
    
    def stable_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Ultra-stable gradient penalty"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Preventive clipping
        real_data = torch.clamp(real_data, -3, 3)
        fake_data = torch.clamp(fake_data, -3, 3)
        
        # Simple more stable interpolation
        alpha = torch.rand(batch_size, 1, 1).to(device)
        
        # Interpolation without additional noise
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Forward pass
        d_interpolated = self.discriminator(interpolated)
        
        # Gradients with verification
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Penalty with clipping
        gradients_flat = gradients.reshape(batch_size, -1)
        gradient_norm = gradients_flat.norm(2, dim=1)
        
        # Clipping norm
        gradient_norm = torch.clamp(gradient_norm, 0, 10)
        
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        # Final penalty clipping
        penalty = torch.clamp(penalty, 0, 100)
        
        return penalty
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Ultra-stable training"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        for batch_idx, real_data in enumerate(tqdm(dataloader, desc="Stable-Training", leave=False)):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Input clipping
            real_data = torch.clamp(real_data, -3, 3)
            
            # Train Discriminator (less times)
            d_losses = []
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data with minimal noise
                noise_factor = 0.001  # Much reduced
                real_noisy = real_data + torch.randn_like(real_data) * noise_factor
                real_noisy = torch.clamp(real_noisy, -3, 3)
                
                d_real = self.discriminator(real_noisy)
                
                # Fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                noise = torch.clamp(noise, -2, 2)  # Noise clipping
                
                with torch.no_grad():
                    fake_data = self.generator(noise)
                    fake_data = torch.clamp(fake_data, -3, 3)
                
                d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss with clipping
                d_real_mean = torch.clamp(torch.mean(d_real), -100, 100)
                d_fake_mean = torch.clamp(torch.mean(d_fake), -100, 100)
                wasserstein_d = d_real_mean - d_fake_mean
                
                # Stable gradient penalty
                gp = self.stable_gradient_penalty(real_data, fake_data)
                
                # Total loss with clipping
                d_loss = -wasserstein_d + self.lambda_gp * gp
                d_loss = torch.clamp(d_loss, -1000, 1000)
                
                # Check if loss is valid
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    
                    # Aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.1)
                    
                    self.optim_D.step()
                else:
                    print(f"Invalid D loss detected: {d_loss.item()}")
                    continue
                
                d_losses.append(d_loss.item())
            
            # Train Generator
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            noise = torch.clamp(noise, -2, 2)
            
            fake_data = self.generator(noise)
            fake_data = torch.clamp(fake_data, -3, 3)
            
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            g_loss = torch.clamp(g_loss, -1000, 1000)
            
            # Check if loss is valid
            if torch.isfinite(g_loss):
                g_loss.backward()
                
                # Aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1)
                
                self.optim_G.step()
            else:
                print(f"Invalid G loss detected: {g_loss.item()}")
                continue
            
            # Record metrics
            if len(d_losses) > 0:
                epoch_metrics['g_loss'].append(g_loss.item())
                epoch_metrics['d_loss'].append(np.mean(d_losses))
                epoch_metrics['wasserstein_distance'].append(-wasserstein_d.item())
                epoch_metrics['gradient_penalty'].append(gp.item())
        
        # Update schedulers smoothly
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
    
    def train(self, dataset: AdvancedTrajectoryDataset, epochs: int = 30,  # Reduced
              batch_size: int = 16, save_dir: str = "stable_models"):  # Smaller batch
        """Ultra-stable training"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=2, pin_memory=True)  # Reduced workers
        
        print(f"Starting Ultra-Stable Training:")
        print(f"   Epochs: {epochs} | Batch: {batch_size} | Batches/epoch: {len(dataloader)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            # Check stability
            is_stable = all(
                abs(v) < 1000 for v in [
                    metrics['g_loss'], 
                    metrics['d_loss'], 
                    metrics['wasserstein_distance'],
                    metrics['gradient_penalty']
                ]
            )
            
            if not is_stable:
                print(f"INSTABILITY DETECTED! Reducing learning rates...")
                for param_group in self.optim_G.param_groups:
                    param_group['lr'] *= 0.5
                for param_group in self.optim_D.param_groups:
                    param_group['lr'] *= 0.5
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            print(f"   G: {metrics['g_loss']:.4f} | D: {metrics['d_loss']:.4f}")
            print(f"   WD: {metrics['wasserstein_distance']:.4f} | GP: {metrics['gradient_penalty']:.4f}")
            print(f"   LR_G: {self.optim_G.param_groups[0]['lr']:.2e} | LR_D: {self.optim_D.param_groups[0]['lr']:.2e}")
            print(f"   Stable: {'Yes' if is_stable else 'No'}")
            
            # Checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_dir, f'stable_checkpoint_e{epoch+1}.pth'))
        
        print(f"\nStable training completed!")
    
    def ultra_anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Ultra-advanced scoring system"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Discriminator score with multiple attempts
            disc_scores = []
            for _ in range(5):
                d_score = self.discriminator(trajectories)
                disc_scores.append(-d_score.cpu())
            
            final_disc_score = torch.mean(torch.stack(disc_scores), dim=0)
            
            # 2. Advanced reconstruction score
            reconstruction_scores = []
            n_generations = 10
            
            all_fakes = []
            for _ in range(n_generations):
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_batch = self.generator(noise)
                all_fakes.append(fake_batch.cpu())
            
            all_fakes = torch.cat(all_fakes, dim=0)
            
            for i in range(batch_size):
                real_traj = trajectories[i].cpu()
                
                # Multiple distance metrics
                l2_distances = torch.sum((all_fakes - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                l1_distances = torch.sum(torch.abs(all_fakes - real_traj.unsqueeze(0)), dim=(1, 2))
                
                # Combined distance
                combined_distances = 0.7 * l2_distances + 0.3 * l1_distances
                min_distance = torch.min(combined_distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_scores = torch.tensor(reconstruction_scores)
            
            # 3. Temporal complexity score
            complexity_scores = []
            for i in range(batch_size):
                traj = trajectories[i].cpu()
                
                # Temporal variation
                temporal_diffs = torch.diff(traj, dim=0)
                complexity = torch.var(temporal_diffs, dim=0).mean()
                
                # Sudden changes
                sudden_changes = torch.sum(torch.abs(temporal_diffs) > 2 * torch.std(temporal_diffs))
                
                complexity_score = complexity + 0.1 * sudden_changes
                complexity_scores.append(complexity_score.item())
            
            complexity_scores = torch.tensor(complexity_scores)
            
            # 4. Ultra-intelligent combination
            def robust_normalize(scores):
                if len(scores) <= 1:
                    return torch.zeros_like(scores)
                
                q25, q75 = torch.quantile(scores, 0.25), torch.quantile(scores, 0.75)
                iqr = q75 - q25 + 1e-8
                
                normalized = (scores - q25) / iqr
                return torch.clamp(normalized, -3, 3)
            
            disc_norm = robust_normalize(final_disc_score)
            recon_norm = robust_normalize(reconstruction_scores)
            complex_norm = robust_normalize(complexity_scores)
            
            # Adaptive weights
            weights = torch.softmax(torch.tensor([
                torch.var(disc_norm) + 1e-8,
                torch.var(recon_norm) + 1e-8,
                torch.var(complex_norm) + 1e-8
            ]), dim=0)
            
            # Final score
            ultra_score = (weights[0] * disc_norm + 
                          weights[1] * recon_norm + 
                          weights[2] * complex_norm)
            
            # Final smoothing
            final_score = torch.tanh(ultra_score * 0.5)
        
        return final_score.numpy()
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 80) -> Dict:
        """Ultra-complete evaluation"""
        print(f"Ultra-Evaluation with {len(test_data)} trajectories...")
        
        test_tensor = torch.FloatTensor(test_data)
        ultra_scores = self.ultra_anomaly_score(test_tensor)
        
        threshold = np.percentile(ultra_scores, threshold_percentile)
        predicted_labels = ultra_scores > threshold
        
        # Detailed metrics
        tp = int(np.sum((predicted_labels == True) & (true_labels == True)))
        tn = int(np.sum((predicted_labels == False) & (true_labels == False)))
        fp = int(np.sum((predicted_labels == True) & (true_labels == False)))
        fn = int(np.sum((predicted_labels == False) & (true_labels == True)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels)
        
        try:
            auc_roc = roc_auc_score(true_labels, ultra_scores)
            auc_pr = average_precision_score(true_labels, ultra_scores)
        except:
            auc_roc = auc_pr = 0.0
        
        results = {
            'ultra_scores': ultra_scores,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        print(f"Ultra-Evaluation:")
        print(f"   F1-Score: {f1_score:.3f} (target: >0.50)")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f} (target: >0.75)")
        
        return results
    
    def save_model(self, filepath: str):
        """Save complete model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'history': self.history,
            'hyperparameters': {
                'noise_dim': self.noise_dim,
                'lr_g': self.lr_g,
                'lr_d': self.lr_d,
                'lambda_gp': self.lambda_gp
            }
        }, filepath)


def main():
    """Função principal ultra-otimizada"""
    
    DATA_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    ANOMALY_CSV = r"D:\UTFPR\TCC\AI-City Challenge\train-anomaly-results.csv"
    MODEL_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\wganGp_ultra_v2"
    
    print("🚀 SISTEMA ULTRA-OTIMIZADO V2.0")
    print("=" * 50)
    
    try:
        # Dataset ultra-otimizado
        anomaly_labels = OptimizedAnomalyLabels(ANOMALY_CSV)
        
        dataset = AdvancedTrajectoryDataset(
            DATA_DIR, 
            anomaly_labels,
            balance_ratio=0.35,  # 35% de anomalias na validação
            quality_threshold=0.1
        )
        
        # Detector ultra-avançado
        detector = UltraWGANGPDetector(seq_len=20, feature_dim=5)
        
        # Treinamento
        detector.train(dataset, epochs=60, batch_size=32, save_dir=MODEL_DIR)
        
        # Avaliação com múltiplos thresholds
        val_data, val_labels = dataset.get_val_data()
        
        best_f1 = 0
        best_results = None
        
        print("\n🔍 Testando múltiplos thresholds:")
        for percentile in range(75, 95, 2):
            results = detector.evaluate_with_labels(val_data.numpy(), val_labels, 
                                                   threshold_percentile=percentile)
            
            print(f"   {percentile}%: F1={results['f1_score']:.3f}, P={results['precision']:.3f}, R={results['recall']:.3f}")
            
            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_results = results
        
        print(f"\n🏆 MELHOR RESULTADO: F1-Score = {best_f1:.3f}")
        print(f"   Threshold: {best_results['threshold_percentile']}% percentil")
        print(f"   Precisão: {best_results['precision']:.3f}")
        print(f"   Recall: {best_results['recall']:.3f}")
        print(f"   AUC-ROC: {best_results['auc_roc']:.3f}")
        
        # Salvar resultados
        results_file = os.path.join(MODEL_DIR, 'ultra_v2_results.json')
        with open(results_file, 'w') as f:
            save_results = {k: v for k, v in best_results.items() 
                           if k not in ['ultra_scores']}
            json.dump(save_results, f, indent=2)
        
        print(f"\n📄 Resultados salvos: {results_file}")
        print("\n🎉 OTIMIZAÇÃO ULTRA V2.0 CONCLUÍDA!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
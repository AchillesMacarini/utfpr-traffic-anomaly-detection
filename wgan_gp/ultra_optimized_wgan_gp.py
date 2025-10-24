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
    """Sistema otimizado para análise temporal de anomalias"""
    
    def __init__(self, csv_path: str, fps: float = 30.0):
        self.csv_path = csv_path
        self.fps = fps
        self.anomaly_intervals = {}
        self.video_stats = {}
        
        print(f"📋 Carregando ground truth otimizado: {csv_path}")
        self._analyze_anomaly_patterns()
    
    def _analyze_anomaly_patterns(self):
        """Análise avançada dos padrões de anomalias"""
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
                
                # Atualizar estatísticas
                stats = self.video_stats[video_id]
                stats['total_anomaly_time'] += duration
                stats['anomaly_count'] += 1
            
            # Calcular métricas finais
            for video_id, stats in self.video_stats.items():
                if stats['anomaly_count'] > 0:
                    stats['avg_duration'] = stats['total_anomaly_time'] / stats['anomaly_count']
                    # Score de severidade baseado em duração e frequência
                    stats['severity_score'] = np.log1p(stats['total_anomaly_time']) * stats['anomaly_count']
            
            print(f"✅ Análise completa: {len(self.anomaly_intervals)} vídeos anômalos")
            
            # Estatísticas detalhadas
            total_anomalies = sum(len(intervals) for intervals in self.anomaly_intervals.values())
            avg_duration = np.mean([dur for intervals in self.anomaly_intervals.values() 
                                  for _, _, dur in intervals])
            
            print(f"📊 Estatísticas:")
            print(f"   Total de anomalias: {total_anomalies}")
            print(f"   Duração média: {avg_duration:.1f}s")
            print(f"   Vídeos mais severos: {self.get_high_severity_videos()[:5]}")
            
        except Exception as e:
            print(f"❌ Erro ao analisar anomalias: {e}")
    
    def has_anomalies(self, video_id: int) -> bool:
        return video_id in self.anomaly_intervals
    
    def get_severity_score(self, video_id: int) -> float:
        return self.video_stats.get(video_id, {}).get('severity_score', 0.0)
    
    def get_high_severity_videos(self) -> List[int]:
        return sorted(self.video_stats.keys(), 
                     key=lambda v: self.video_stats[v]['severity_score'], reverse=True)


class AdvancedTrajectoryDataset(Dataset):
    """Dataset ultra-otimizado com análise de qualidade - CORRIGIDO"""
    
    def __init__(self, data_dir: str, anomaly_labels: OptimizedAnomalyLabels = None, 
                 balance_ratio: float = 0.35, quality_threshold: float = 0.01):  # Threshold muito menor
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.balance_ratio = balance_ratio
        self.quality_threshold = quality_threshold
        
        # Scaler robusto
        self.scaler = RobustScaler()
        
        self._load_and_optimize_data()
    
    def _load_and_optimize_data(self):
        """Carregamento otimizado com análise de qualidade - DIAGNÓSTICO INCLUÍDO"""
        
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"📁 Carregando {len(npy_files)} arquivos com análise de qualidade...")
        
        all_trajectories = []
        trajectory_labels = []
        trajectory_qualities = []
        video_sources = []
        
        # Estatísticas de diagnóstico
        total_files = 0
        files_with_data = 0
        total_trajectories = 0
        trajectories_after_quality = 0
        quality_stats = []
        
        # Primeira passada: coletar dados e calcular qualidade
        for npy_file in tqdm(npy_files, desc="Analisando arquivos"):
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            total_files += 1
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) > 0:
                    files_with_data += 1
                    total_trajectories += len(trajectories)
                    
                    print(f"   📄 {npy_file}: {len(trajectories)} trajetórias, shape: {trajectories.shape}")
                    
                    # Análise de qualidade das trajetórias
                    qualities = self._analyze_trajectory_quality(trajectories)
                    quality_stats.extend(qualities)
                    
                    print(f"      Qualidade - Min: {np.min(qualities):.4f}, Max: {np.max(qualities):.4f}, Média: {np.mean(qualities):.4f}")
                    
                    # Filtrar por qualidade com threshold adaptativo
                    if len(qualities) > 0:
                        # Se threshold muito restritivo, usar percentil
                        adaptive_threshold = min(self.quality_threshold, np.percentile(qualities, 20))
                        high_quality_mask = qualities > adaptive_threshold
                        
                        print(f"      Threshold usado: {adaptive_threshold:.4f}, Passou filtro: {np.sum(high_quality_mask)}/{len(qualities)}")
                        
                        if np.any(high_quality_mask):
                            filtered_trajectories = trajectories[high_quality_mask]
                            filtered_qualities = qualities[high_quality_mask]
                            trajectories_after_quality += len(filtered_trajectories)
                            
                            # Determinar labels
                            is_anomalous_video = (self.anomaly_labels and 
                                                self.anomaly_labels.has_anomalies(video_id))
                            
                            # Adicionar à coleção
                            all_trajectories.append(filtered_trajectories)
                            trajectory_labels.extend([is_anomalous_video] * len(filtered_trajectories))
                            trajectory_qualities.extend(filtered_qualities)
                            video_sources.extend([video_id] * len(filtered_trajectories))
                            
                            print(f"      ✅ Adicionadas {len(filtered_trajectories)} trajetórias (Anômalo: {is_anomalous_video})")
                        else:
                            print(f"      ⚠️ Nenhuma trajetória passou no filtro de qualidade")
                    else:
                        print(f"      ⚠️ Erro no cálculo de qualidade")
                else:
                    print(f"   📄 {npy_file}: arquivo vazio")
                        
            except Exception as e:
                print(f"⚠️ Erro ao processar {npy_file}: {e}")
        
        # Diagnóstico final
        print(f"\n📊 DIAGNÓSTICO COMPLETO:")
        print(f"   Total de arquivos: {total_files}")
        print(f"   Arquivos com dados: {files_with_data}")
        print(f"   Total de trajetórias brutas: {total_trajectories}")
        print(f"   Trajetórias após filtro: {trajectories_after_quality}")
        
        if quality_stats:
            print(f"   Qualidade geral - Min: {np.min(quality_stats):.4f}, Max: {np.max(quality_stats):.4f}")
            print(f"   Qualidade geral - Média: {np.mean(quality_stats):.4f}, Std: {np.std(quality_stats):.4f}")
            print(f"   Percentis: 10%={np.percentile(quality_stats, 10):.4f}, 50%={np.percentile(quality_stats, 50):.4f}, 90%={np.percentile(quality_stats, 90):.4f}")
        
        # Concatenar todos os dados
        if all_trajectories:
            self.all_trajectories = np.concatenate(all_trajectories, axis=0)
            self.all_labels = np.array(trajectory_labels, dtype=bool)
            self.all_qualities = np.array(trajectory_qualities)
            self.all_video_sources = np.array(video_sources)
            
            print(f"\n✅ Dados coletados com sucesso:")
            print(f"   Shape final: {self.all_trajectories.shape}")
            print(f"   Labels normais: {np.sum(~self.all_labels)}")
            print(f"   Labels anômalos: {np.sum(self.all_labels)}")
            
            # Normalização robusta
            self._apply_robust_normalization()
            
            # Divisão estratificada
            self._create_stratified_split()
            
            print(f"✅ Dataset otimizado carregado:")
            print(f"   Total de trajetórias: {len(self.all_trajectories)}")
            print(f"   Qualidade média: {np.mean(self.all_qualities):.3f}")
            print(f"   Treino: {len(self.train_trajectories)} trajetórias")
            print(f"   Validação: {len(self.val_trajectories)} (Normal: {np.sum(~self.val_labels)}, Anômalo: {np.sum(self.val_labels)})")
        else:
            # Diagnóstico do erro
            print(f"\n❌ ERRO: Nenhuma trajetória passou no filtro!")
            print(f"   Possíveis causas:")
            print(f"   1. Threshold muito alto ({self.quality_threshold})")
            print(f"   2. Problema no cálculo de qualidade")
            print(f"   3. Dados de entrada inválidos")
            print(f"   4. Arquivos .npy corrompidos")
            
            # Tentar com threshold zero (sem filtro)
            print(f"\n🔧 Tentando carregar SEM filtro de qualidade...")
            self._load_without_quality_filter()
    
    def _load_without_quality_filter(self):
        """Carregamento de emergência sem filtro de qualidade"""
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        all_trajectories = []
        trajectory_labels = []
        
        for npy_file in tqdm(npy_files[:10], desc="Carregamento de emergência"):  # Apenas primeiros 10 arquivos
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) > 0:
                    # Verificar se dados são válidos
                    if not np.any(np.isnan(trajectories)) and not np.any(np.isinf(trajectories)):
                        # Limitar quantidade por arquivo
                        max_per_file = 100
                        if len(trajectories) > max_per_file:
                            indices = np.random.choice(len(trajectories), max_per_file, replace=False)
                            trajectories = trajectories[indices]
                        
                        is_anomalous_video = (self.anomaly_labels and 
                                            self.anomaly_labels.has_anomalies(video_id))
                        
                        all_trajectories.append(trajectories)
                        trajectory_labels.extend([is_anomalous_video] * len(trajectories))
                        
                        print(f"   ✅ {npy_file}: {len(trajectories)} trajetórias (Anômalo: {is_anomalous_video})")
                    else:
                        print(f"   ⚠️ {npy_file}: contém NaN/Inf")
                        
            except Exception as e:
                print(f"   ❌ {npy_file}: {e}")
        
        if all_trajectories:
            self.all_trajectories = np.concatenate(all_trajectories, axis=0)
            self.all_labels = np.array(trajectory_labels, dtype=bool)
            self.all_qualities = np.ones(len(self.all_trajectories))  # Qualidade uniforme
            self.all_video_sources = np.zeros(len(self.all_trajectories), dtype=int)

            print(f"🔧 Carregamento de emergência bem-sucedido: {len(self.all_trajectories)} trajetórias")

            # Aplicar normalização
            self._apply_robust_normalization()
            self._create_stratified_split()
            
        else:
            raise ValueError("Falha completa no carregamento! Verifique os dados de entrada.")
    
    def _analyze_trajectory_quality(self, trajectories: np.ndarray) -> np.ndarray:
        """Análise de qualidade mais robusta"""
        qualities = []
        
        for traj in trajectories:
            try:
                # Verificar se trajetória é válida
                if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                    qualities.append(0.0)
                    continue
                
                # 1. Variabilidade espacial (movimento não trivial)
                spatial_var = np.var(traj[:, :2]) if traj.shape[1] >= 2 else 0.0
                spatial_var = max(0.0, min(spatial_var, 10.0))  # Clipar valores extremos
                
                # 2. Consistência temporal (se velocidade existe)
                if traj.shape[1] >= 3:
                    velocity_var = np.var(traj[:, 2])
                    velocity_consistency = 1.0 / (1.0 + velocity_var) if velocity_var > 0 else 1.0
                else:
                    velocity_consistency = 1.0
                
                # 3. Penalidade por valores extremos (normalizada)
                abs_values = np.abs(traj)
                extremes_ratio = np.sum(abs_values > 5) / traj.size  # Threshold mais permissivo
                extremes_penalty = max(0.0, min(extremes_ratio, 1.0))
                
                # 4. Penalidade por valores constantes
                constant_penalty = 0.0
                for col in range(traj.shape[1]):
                    if np.std(traj[:, col]) < 1e-6:  # Coluna praticamente constante
                        constant_penalty += 0.2
                
                # Score de qualidade combinado (sempre positivo)
                base_quality = spatial_var * velocity_consistency
                penalties = extremes_penalty + constant_penalty
                
                quality = max(0.001, base_quality * (1 - min(penalties, 0.9)))  # Mínimo 0.001
                qualities.append(quality)
                
            except Exception as e:
                print(f"      Erro ao calcular qualidade de trajetória: {e}")
                qualities.append(0.001)  # Qualidade mínima em caso de erro
        
        return np.array(qualities)
    
    def _apply_robust_normalization(self):
        """Normalização robusta usando RobustScaler"""
        original_shape = self.all_trajectories.shape
        
        # Reshape para normalização
        trajectories_flat = self.all_trajectories.reshape(-1, original_shape[-1])
        
        # Aplicar RobustScaler
        normalized_flat = self.scaler.fit_transform(trajectories_flat)
        
        # Reshape de volta
        self.all_trajectories = normalized_flat.reshape(original_shape)
        
        print(f"🔧 Normalização robusta aplicada")
    
    def _create_stratified_split(self):
        """Divisão estratificada balanceada"""
        normal_indices = np.where(~self.all_labels)[0]
        anomaly_indices = np.where(self.all_labels)[0]
        
        # Treino: apenas trajetórias normais (80%)
        n_train_normal = int(len(normal_indices) * 0.8)
        np.random.seed(42)
        normal_train_idx = np.random.choice(normal_indices, n_train_normal, replace=False)
        
        self.train_trajectories = self.all_trajectories[normal_train_idx]
        
        # Validação balanceada
        normal_val_idx = np.setdiff1d(normal_indices, normal_train_idx)
        n_val_normal = len(normal_val_idx)
        n_val_anomaly = int(n_val_normal * self.balance_ratio / (1 - self.balance_ratio))
        
        if len(anomaly_indices) > 0:
            # Selecionar anomalias de alta qualidade
            anomaly_qualities = self.all_qualities[anomaly_indices]
            top_anomaly_idx = anomaly_indices[np.argsort(anomaly_qualities)[-n_val_anomaly:]]
            
            val_indices = np.concatenate([normal_val_idx, top_anomaly_idx])
        else:
            val_indices = normal_val_idx
        
        self.val_trajectories = self.all_trajectories[val_indices]
        self.val_labels = self.all_labels[val_indices]
        
        # Embaralhar validação
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
    """Gerador ultra-otimizado com atenção e skip connections"""
    
    def __init__(self, noise_dim: int = 256, seq_len: int = 20, feature_dim: int = 5):
        super(UltraGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Encoder do ruído com skip connections
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
        
        # Transformer temporal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Decoder para features
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
    """Discriminador ultra-otimizado com arquitetura híbrida"""
    
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
    """Sistema ultra-otimizado de detecção de anomalias"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Hiperparâmetros ultra-otimizados
        self.noise_dim = 256
        self.lr_g = 0.00005
        self.lr_d = 0.0002
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 10
        self.n_critic = 3
        
        # Redes
        self.generator = UltraGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = UltraDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Otimizadores com scheduler
        self.optim_G = optim.AdamW(self.generator.parameters(), lr=self.lr_g, 
                                  betas=(self.beta1, self.beta2), weight_decay=1e-6)
        self.optim_D = optim.AdamW(self.discriminator.parameters(), lr=self.lr_d, 
                                  betas=(self.beta1, self.beta2), weight_decay=1e-6)
        
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optim_G, T_max=50)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optim_D, T_max=50)
        
        self.history = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        print(f"🚀 Ultra WGAN-GP inicializado no device: {self.device}")
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Gradient penalty ultra-estável"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Múltiplas interpolações para estabilidade
        penalties = []
        for _ in range(2):
            alpha = torch.rand(batch_size, 1, 1).to(device)
            
            # Interpolação com pequeno ruído
            epsilon = torch.randn_like(real_data) * 0.001
            interpolated = alpha * real_data + (1 - alpha) * fake_data + epsilon
            interpolated.requires_grad_(True)
            
            # Forward pass
            d_interpolated = self.discriminator(interpolated)
            
            # Gradientes
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
        """Treinamento ultra-estável"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        for batch_idx, real_data in enumerate(tqdm(dataloader, desc="Ultra-Training", leave=False)):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Treinar Discriminador
            d_losses = []
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data com pequeno ruído
                noise_factor = 0.01 * (1 - batch_idx / len(dataloader))  # Decair durante a época
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
            
            # Treinar Gerador
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise)
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            self.optim_G.step()
            
            # Registrar métricas
            epoch_metrics['g_loss'].append(g_loss.item())
            epoch_metrics['d_loss'].append(np.mean(d_losses))
            epoch_metrics['wasserstein_distance'].append(-wasserstein_d.item())
            epoch_metrics['gradient_penalty'].append(gp.item())
        
        # Atualizar schedulers
        avg_g_loss = np.mean(epoch_metrics['g_loss'])
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def train(self, dataset: AdvancedTrajectoryDataset, epochs: int = 60, 
              batch_size: int = 32, save_dir: str = "ultra_models"):
        """Treinamento ultra-completo"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=4, pin_memory=True)
        
        print(f"🔥 Iniciando Ultra-Treinamento:")
        print(f"   Épocas: {epochs} | Batch: {batch_size} | Batches/época: {len(dataloader)}")
        
        best_wd = float('inf')
        patience = 0
        max_patience = 15
        
        for epoch in range(epochs):
            print(f"\n🚀 Época {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            current_wd = abs(metrics['wasserstein_distance'])
            
            print(f"   G: {metrics['g_loss']:.4f} | D: {metrics['d_loss']:.4f}")
            print(f"   WD: {metrics['wasserstein_distance']:.4f} | GP: {metrics['gradient_penalty']:.4f}")
            print(f"   LR_G: {self.optim_G.param_groups[0]['lr']:.2e} | LR_D: {self.optim_D.param_groups[0]['lr']:.2e}")
            
            # Salvar melhor modelo
            if current_wd < best_wd:
                best_wd = current_wd
                patience = 0
                self.save_model(os.path.join(save_dir, 'ultra_best_model.pth'))
                print(f"   💎 MELHOR MODELO! WD: {best_wd:.4f}")
            else:
                patience += 1
            
            # Early stopping
            if patience >= max_patience:
                print(f"\n⏹️ Early stopping após {patience} épocas sem melhoria")
                break
            
            # Checkpoint
            if (epoch + 1) % 15 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_e{epoch+1}.pth'))
        
        print(f"\n🎉 Treinamento concluído! Melhor WD: {best_wd:.4f}")
    
    def ultra_anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Sistema ultra-avançado de scoring"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Score discriminador com múltiplas tentativas
            disc_scores = []
            for _ in range(5):
                d_score = self.discriminator(trajectories)
                disc_scores.append(-d_score.cpu())
            
            final_disc_score = torch.mean(torch.stack(disc_scores), dim=0)
            
            # 2. Score de reconstrução avançado
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
                
                # Múltiplas métricas de distância
                l2_distances = torch.sum((all_fakes - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                l1_distances = torch.sum(torch.abs(all_fakes - real_traj.unsqueeze(0)), dim=(1, 2))
                
                # Distância combinada
                combined_distances = 0.7 * l2_distances + 0.3 * l1_distances
                min_distance = torch.min(combined_distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_scores = torch.tensor(reconstruction_scores)
            
            # 3. Score de complexidade temporal
            complexity_scores = []
            for i in range(batch_size):
                traj = trajectories[i].cpu()
                
                # Variação temporal
                temporal_diffs = torch.diff(traj, dim=0)
                complexity = torch.var(temporal_diffs, dim=0).mean()
                
                # Mudanças abruptas
                sudden_changes = torch.sum(torch.abs(temporal_diffs) > 2 * torch.std(temporal_diffs))
                
                complexity_score = complexity + 0.1 * sudden_changes
                complexity_scores.append(complexity_score.item())
            
            complexity_scores = torch.tensor(complexity_scores)
            
            # 4. Combinação ultra-inteligente
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
            
            # Pesos adaptativos
            weights = torch.softmax(torch.tensor([
                torch.var(disc_norm) + 1e-8,
                torch.var(recon_norm) + 1e-8,
                torch.var(complex_norm) + 1e-8
            ]), dim=0)
            
            # Score final
            ultra_score = (weights[0] * disc_norm + 
                          weights[1] * recon_norm + 
                          weights[2] * complex_norm)
            
            # Suavização final
            final_score = torch.tanh(ultra_score * 0.5)
        
        return final_score.numpy()
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 80) -> Dict:
        """Avaliação ultra-completa"""
        print(f"🔍 Ultra-Avaliação com {len(test_data)} trajetórias...")
        
        test_tensor = torch.FloatTensor(test_data)
        ultra_scores = self.ultra_anomaly_score(test_tensor)
        
        threshold = np.percentile(ultra_scores, threshold_percentile)
        predicted_labels = ultra_scores > threshold
        
        # Métricas detalhadas
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
        
        print(f"✅ Ultra-Avaliação:")
        print(f"   🎯 F1-Score: {f1_score:.3f} (target: >0.50)")
        print(f"   📊 Precisão: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   🏆 AUC-ROC: {auc_roc:.3f} (target: >0.75)")
        
        return results
    
    def save_model(self, filepath: str):
        """Salvar modelo completo"""
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
    """Discriminador ultra-estável com normalização aprimorada"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(StableUltraDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Convoluções mais estáveis
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=k, padding=k//2),  # Reduzido de 128 para 64
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),  # Mudança para LeakyReLU
                nn.Dropout(0.1)
            ) for k in [3, 5, 7]
        ])
        
        # Fusão mais conservadora
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # Reduzido
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # LSTM simples ao invés de Transformer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,  # Reduzido de 3 para 2
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Classificador mais simples
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
        
        # Inicialização conservadora
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Gain muito baixo
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
        
        # Clipping de entrada para estabilidade
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
    """Gerador ultra-estável"""
    
    def __init__(self, noise_dim: int = 128, seq_len: int = 20, feature_dim: int = 5):  # Noise dim reduzido
        super(StableUltraGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Encoder mais simples
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
        
        # LSTM simples ao invés de Transformer
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
        
        # Inicialização conservadora
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
    """Detector ultra-estável"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Hiperparâmetros ultra-conservadores
        self.noise_dim = 128  # Reduzido
        self.lr_g = 0.00001   # Muito mais baixo
        self.lr_d = 0.00005   # Muito mais baixo
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 1.0  # REDUZIDO drasticamente de 10 para 1
        self.n_critic = 2     # Reduzido
        
        # Redes estáveis
        self.generator = StableUltraGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = StableUltraDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Otimizadores conservadores
        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lr_g, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, 
                                 betas=(self.beta1, self.beta2), weight_decay=1e-5)
        
        # Schedulers mais suaves
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optim_G, gamma=0.99)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optim_D, gamma=0.99)
        
        self.history = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        print(f"🛡️ WGAN-GP Ultra-Estável inicializado:")
        print(f"   Device: {self.device}")
        print(f"   LR_G: {self.lr_g:.2e}, LR_D: {self.lr_d:.2e}")
        print(f"   Lambda GP: {self.lambda_gp} (reduzido para estabilidade)")
    
    def stable_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Gradient penalty ultra-estável"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Clipping preventivo
        real_data = torch.clamp(real_data, -3, 3)
        fake_data = torch.clamp(fake_data, -3, 3)
        
        # Uma única interpolação mais estável
        alpha = torch.rand(batch_size, 1, 1).to(device)
        
        # Interpolação sem ruído adicional
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Forward pass
        d_interpolated = self.discriminator(interpolated)
        
        # Gradientes com verificação
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Penalty com clipping
        gradients_flat = gradients.reshape(batch_size, -1)
        gradient_norm = gradients_flat.norm(2, dim=1)
        
        # Clipping da norma
        gradient_norm = torch.clamp(gradient_norm, 0, 10)
        
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        # Clipping final do penalty
        penalty = torch.clamp(penalty, 0, 100)
        
        return penalty
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Treinamento ultra-estável"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein_distance': [], 'gradient_penalty': []}
        
        for batch_idx, real_data in enumerate(tqdm(dataloader, desc="Stable-Training", leave=False)):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Clipping de entrada
            real_data = torch.clamp(real_data, -3, 3)
            
            # Treinar Discriminador (menos vezes)
            d_losses = []
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Real data com ruído mínimo
                noise_factor = 0.001  # Muito reduzido
                real_noisy = real_data + torch.randn_like(real_data) * noise_factor
                real_noisy = torch.clamp(real_noisy, -3, 3)
                
                d_real = self.discriminator(real_noisy)
                
                # Fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                noise = torch.clamp(noise, -2, 2)  # Clipping do ruído
                
                with torch.no_grad():
                    fake_data = self.generator(noise)
                    fake_data = torch.clamp(fake_data, -3, 3)
                
                d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss com clipping
                d_real_mean = torch.clamp(torch.mean(d_real), -100, 100)
                d_fake_mean = torch.clamp(torch.mean(d_fake), -100, 100)
                wasserstein_d = d_real_mean - d_fake_mean
                
                # Stable gradient penalty
                gp = self.stable_gradient_penalty(real_data, fake_data)
                
                # Total loss com clipping
                d_loss = -wasserstein_d + self.lambda_gp * gp
                d_loss = torch.clamp(d_loss, -1000, 1000)
                
                # Verificar se loss é válido
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    
                    # Gradient clipping agressivo
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.1)
                    
                    self.optim_D.step()
                else:
                    print(f"⚠️ D loss inválido detectado: {d_loss.item()}")
                    continue
                
                d_losses.append(d_loss.item())
            
            # Treinar Gerador
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            noise = torch.clamp(noise, -2, 2)
            
            fake_data = self.generator(noise)
            fake_data = torch.clamp(fake_data, -3, 3)
            
            d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            g_loss = torch.clamp(g_loss, -1000, 1000)
            
            # Verificar se loss é válido
            if torch.isfinite(g_loss):
                g_loss.backward()
                
                # Gradient clipping agressivo
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1)
                
                self.optim_G.step()
            else:
                print(f"⚠️ G loss inválido detectado: {g_loss.item()}")
                continue
            
            # Registrar métricas
            if len(d_losses) > 0:
                epoch_metrics['g_loss'].append(g_loss.item())
                epoch_metrics['d_loss'].append(np.mean(d_losses))
                epoch_metrics['wasserstein_distance'].append(-wasserstein_d.item())
                epoch_metrics['gradient_penalty'].append(gp.item())
        
        # Atualizar schedulers suavemente
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
    
    def train(self, dataset: AdvancedTrajectoryDataset, epochs: int = 30,  # Reduzido
              batch_size: int = 16, save_dir: str = "stable_models"):  # Batch menor
        """Treinamento ultra-estável"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=2, pin_memory=True)  # Workers reduzido
        
        print(f"🛡️ Iniciando Treinamento Ultra-Estável:")
        print(f"   Épocas: {epochs} | Batch: {batch_size} | Batches/época: {len(dataloader)}")
        
        for epoch in range(epochs):
            print(f"\n🛡️ Época {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            # Verificar estabilidade
            is_stable = all(
                abs(v) < 1000 for v in [
                    metrics['g_loss'], 
                    metrics['d_loss'], 
                    metrics['wasserstein_distance'],
                    metrics['gradient_penalty']
                ]
            )
            
            if not is_stable:
                print(f"⚠️ INSTABILIDADE DETECTADA! Reduzindo learning rates...")
                for param_group in self.optim_G.param_groups:
                    param_group['lr'] *= 0.5
                for param_group in self.optim_D.param_groups:
                    param_group['lr'] *= 0.5
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            print(f"   G: {metrics['g_loss']:.4f} | D: {metrics['d_loss']:.4f}")
            print(f"   WD: {metrics['wasserstein_distance']:.4f} | GP: {metrics['gradient_penalty']:.4f}")
            print(f"   LR_G: {self.optim_G.param_groups[0]['lr']:.2e} | LR_D: {self.optim_D.param_groups[0]['lr']:.2e}")
            print(f"   Estável: {'✅' if is_stable else '⚠️'}")
            
            # Checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_dir, f'stable_checkpoint_e{epoch+1}.pth'))
        
        print(f"\n🎉 Treinamento estável concluído!")
    
    def ultra_anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Sistema ultra-avançado de scoring"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Score discriminador com múltiplas tentativas
            disc_scores = []
            for _ in range(5):
                d_score = self.discriminator(trajectories)
                disc_scores.append(-d_score.cpu())
            
            final_disc_score = torch.mean(torch.stack(disc_scores), dim=0)
            
            # 2. Score de reconstrução avançado
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
                
                # Múltiplas métricas de distância
                l2_distances = torch.sum((all_fakes - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                l1_distances = torch.sum(torch.abs(all_fakes - real_traj.unsqueeze(0)), dim=(1, 2))
                
                # Distância combinada
                combined_distances = 0.7 * l2_distances + 0.3 * l1_distances
                min_distance = torch.min(combined_distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_scores = torch.tensor(reconstruction_scores)
            
            # 3. Score de complexidade temporal
            complexity_scores = []
            for i in range(batch_size):
                traj = trajectories[i].cpu()
                
                # Variação temporal
                temporal_diffs = torch.diff(traj, dim=0)
                complexity = torch.var(temporal_diffs, dim=0).mean()
                
                # Mudanças abruptas
                sudden_changes = torch.sum(torch.abs(temporal_diffs) > 2 * torch.std(temporal_diffs))
                
                complexity_score = complexity + 0.1 * sudden_changes
                complexity_scores.append(complexity_score.item())
            
            complexity_scores = torch.tensor(complexity_scores)
            
            # 4. Combinação ultra-inteligente
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
            
            # Pesos adaptativos
            weights = torch.softmax(torch.tensor([
                torch.var(disc_norm) + 1e-8,
                torch.var(recon_norm) + 1e-8,
                torch.var(complex_norm) + 1e-8
            ]), dim=0)
            
            # Score final
            ultra_score = (weights[0] * disc_norm + 
                          weights[1] * recon_norm + 
                          weights[2] * complex_norm)
            
            # Suavização final
            final_score = torch.tanh(ultra_score * 0.5)
        
        return final_score.numpy()
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 80) -> Dict:
        """Avaliação ultra-completa"""
        print(f"🔍 Ultra-Avaliação com {len(test_data)} trajetórias...")
        
        test_tensor = torch.FloatTensor(test_data)
        ultra_scores = self.ultra_anomaly_score(test_tensor)
        
        threshold = np.percentile(ultra_scores, threshold_percentile)
        predicted_labels = ultra_scores > threshold
        
        # Métricas detalhadas
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
        
        print(f"✅ Ultra-Avaliação:")
        print(f"   🎯 F1-Score: {f1_score:.3f} (target: >0.50)")
        print(f"   📊 Precisão: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   🏆 AUC-ROC: {auc_roc:.3f} (target: >0.75)")
        
        return results
    
    def save_model(self, filepath: str):
        """Salvar modelo completo"""
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
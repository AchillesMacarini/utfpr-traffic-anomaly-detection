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

class ImprovedAnomalyLabels:
    """Classe melhorada para gerenciar rótulos de anomalias"""
    
    def __init__(self, csv_path: str, fps: float = 30.0):
        self.csv_path = csv_path
        self.fps = fps
        self.anomaly_intervals = {}
        self.video_durations = {}  # Adicionar durações dos vídeos
        
        print(f"📋 Carregando rótulos de anomalias de: {csv_path}")
        self._load_anomaly_data()
    
    def _load_anomaly_data(self):
        """Carrega dados de anomalias do CSV com melhor processamento"""
        try:
            df = pd.read_csv(self.csv_path)
            
            for _, row in df.iterrows():
                video_id = int(row['video_id'])
                start_time = float(row['start_time'])
                end_time = float(row['end_time'])
                
                if video_id not in self.anomaly_intervals:
                    self.anomaly_intervals[video_id] = []
                
                self.anomaly_intervals[video_id].append((start_time, end_time))
                
                # Estimar duração do vídeo baseada na maior anomalia
                if video_id not in self.video_durations:
                    self.video_durations[video_id] = end_time
                else:
                    self.video_durations[video_id] = max(self.video_durations[video_id], end_time)
            
            # Adicionar margem de segurança às durações
            for video_id in self.video_durations:
                self.video_durations[video_id] += 100  # +100 segundos de margem
            
            total_anomalies = sum(len(intervals) for intervals in self.anomaly_intervals.values())
            print(f"✅ Carregados {total_anomalies} intervalos de anomalias para {len(self.anomaly_intervals)} vídeos")
            
            # Estatísticas detalhadas
            for video_id, intervals in self.anomaly_intervals.items():
                total_anomaly_time = sum(end - start for start, end in intervals)
                video_duration = self.video_durations.get(video_id, 900)  # Default 15 min
                anomaly_percentage = (total_anomaly_time / video_duration) * 100
                print(f"   Vídeo {video_id}: {len(intervals)} anomalias, {total_anomaly_time:.1f}s ({anomaly_percentage:.1f}% do vídeo)")
            
        except Exception as e:
            print(f"❌ Erro ao carregar anomalias: {e}")
            self.anomaly_intervals = {}
    
    def has_anomalies(self, video_id: int) -> bool:
        """Verifica se um vídeo possui anomalias conhecidas"""
        return video_id in self.anomaly_intervals and len(self.anomaly_intervals[video_id]) > 0
    
    def get_anomaly_ratio(self, video_id: int) -> float:
        """Retorna a proporção de tempo anômalo no vídeo"""
        if not self.has_anomalies(video_id):
            return 0.0
        
        total_anomaly_time = sum(end - start for start, end in self.anomaly_intervals[video_id])
        video_duration = self.video_durations.get(video_id, 900)
        return total_anomaly_time / video_duration


class BalancedTrajectoryDataset(Dataset):
    """Dataset balanceado para melhor treinamento"""
    
    def __init__(self, data_dir: str, anomaly_labels: Optional[ImprovedAnomalyLabels] = None, 
                 train_split: float = 0.8, use_only_normal_videos: bool = True, 
                 max_trajectories_per_video: int = 200, balance_ratio: float = 0.3):
        """
        Args:
            balance_ratio: Proporção de trajetórias anômalas na validação (0.3 = 30%)
        """
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.use_only_normal_videos = use_only_normal_videos
        self.max_trajectories_per_video = max_trajectories_per_video
        self.balance_ratio = balance_ratio
        
        # Carregar e processar dados
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Carrega e processa dados com melhor balanceamento"""
        
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"📁 Carregando trajetórias de {len(npy_files)} arquivos...")
        
        normal_videos = []
        anomaly_videos = []
        
        with tqdm(total=len(npy_files), desc="Carregando dados", unit="file") as pbar:
            for npy_file in npy_files:
                video_id = int(npy_file.split('_')[0])
                file_path = os.path.join(self.data_dir, npy_file)
                
                try:
                    trajectories = np.load(file_path)
                    if len(trajectories) > 0:
                        # Amostrar trajetórias para evitar overfitting
                        if len(trajectories) > self.max_trajectories_per_video:
                            indices = np.random.choice(len(trajectories), self.max_trajectories_per_video, replace=False)
                            trajectories = trajectories[indices]
                        
                        # Classificar vídeo
                        if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                            anomaly_ratio = self.anomaly_labels.get_anomaly_ratio(video_id)
                            anomaly_videos.append((video_id, trajectories, anomaly_ratio))
                            video_type = f"anômalo ({anomaly_ratio:.1%})"
                        else:
                            normal_videos.append((video_id, trajectories))
                            video_type = "normal"
                        
                        pbar.set_postfix({'Vídeo': f'{video_id} ({video_type})', 'Trajs': len(trajectories)})
                        
                except Exception as e:
                    print(f"⚠️ Erro ao carregar {npy_file}: {e}")
                
                pbar.update(1)
        
        self._prepare_balanced_data(normal_videos, anomaly_videos)
        
        print(f"✅ Dataset balanceado carregado:")
        print(f"   Vídeos normais: {len(normal_videos)}")
        print(f"   Vídeos com anomalias: {len(anomaly_videos)}")
        print(f"   Treino: {len(self.train_trajectories)} trajetórias")
        print(f"   Validação: {len(self.val_trajectories)} trajetórias")
        print(f"   Validação - Normal: {np.sum(~self.val_has_anomaly)}, Anômalo: {np.sum(self.val_has_anomaly)}")
    
    def _prepare_balanced_data(self, normal_videos: List, anomaly_videos: List):
        """Prepara dados com melhor balanceamento"""
        
        # Treino: apenas vídeos normais
        if self.use_only_normal_videos:
            normal_trajectories = [traj for _, traj in normal_videos]
            if normal_trajectories:
                all_normal = np.concatenate(normal_trajectories, axis=0)
                
                # Dividir trajetórias normais
                n_normal = len(all_normal)
                n_train_normal = int(n_normal * 0.8)  # 80% para treino
                
                indices = np.random.permutation(n_normal)
                self.train_trajectories = all_normal[indices[:n_train_normal]]
                val_normal_traj = all_normal[indices[n_train_normal:]]
                
                print(f"🔍 Usando {len(self.train_trajectories)} trajetórias normais para treino")
            else:
                raise ValueError("Nenhum vídeo normal encontrado!")
        
        # Validação balanceada
        n_val_normal = len(val_normal_traj)
        n_val_anomaly = int(n_val_normal * self.balance_ratio / (1 - self.balance_ratio))
        
        # Selecionar trajetórias anômalas para validação
        val_anomaly_trajectories = []
        if anomaly_videos and n_val_anomaly > 0:
            # Priorizar vídeos com maior proporção de anomalias
            anomaly_videos.sort(key=lambda x: x[2], reverse=True)
            
            collected_anomaly = 0
            for video_id, trajectories, anomaly_ratio in anomaly_videos:
                if collected_anomaly >= n_val_anomaly:
                    break
                
                # Pegar uma amostra das trajetórias deste vídeo
                n_take = min(len(trajectories), n_val_anomaly - collected_anomaly)
                if n_take > 0:
                    indices = np.random.choice(len(trajectories), n_take, replace=False)
                    selected_trajs = trajectories[indices]
                    val_anomaly_trajectories.append(selected_trajs)
                    collected_anomaly += n_take
            
            if val_anomaly_trajectories:
                val_anomaly_trajectories = np.concatenate(val_anomaly_trajectories, axis=0)
            else:
                val_anomaly_trajectories = np.empty((0, val_normal_traj.shape[1], val_normal_traj.shape[2]))
        else:
            val_anomaly_trajectories = np.empty((0, val_normal_traj.shape[1], val_normal_traj.shape[2]))
        
        # Combinar validação
        self.val_trajectories = np.concatenate([val_normal_traj, val_anomaly_trajectories], axis=0)
        self.val_has_anomaly = np.concatenate([
            np.zeros(len(val_normal_traj), dtype=bool),
            np.ones(len(val_anomaly_trajectories), dtype=bool)
        ])
        
        # Embaralhar validação
        val_indices = np.random.permutation(len(self.val_trajectories))
        self.val_trajectories = self.val_trajectories[val_indices]
        self.val_has_anomaly = self.val_has_anomaly[val_indices]
    
    def get_train_data(self):
        return torch.FloatTensor(self.train_trajectories)
    
    def get_val_data(self):
        return torch.FloatTensor(self.val_trajectories), self.val_has_anomaly
    
    def __len__(self):
        return len(self.train_trajectories)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.train_trajectories[idx])


class ImprovedGenerator(nn.Module):
    """Gerador melhorado com arquitetura mais expressiva"""
    
    def __init__(self, noise_dim: int = 128, seq_len: int = 20, feature_dim: int = 5):
        super(ImprovedGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Arquitetura mais profunda e expressiva
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        
        # LSTM para dependências temporais
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Camadas finais
        self.output_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Projetar ruído
        projected = self.noise_proj(noise)  # [batch, 512]
        
        # Expandir para sequência temporal
        sequence = projected.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, 512]
        
        # LSTM para dependências temporais
        with torch.backends.cudnn.flags(enabled=False):  # Para evitar problemas com gradient penalty
            lstm_out, _ = self.lstm(sequence)  # [batch, seq_len, 256]
        
        # Reshape e output
        lstm_out = lstm_out.contiguous().view(-1, 256)  # [batch*seq_len, 256]
        output = self.output_proj(lstm_out)  # [batch*seq_len, feature_dim]
        
        return output.view(batch_size, self.seq_len, self.feature_dim)


class ImprovedDiscriminator(nn.Module):
    """Discriminador melhorado com arquitetura híbrida"""
    
    def __init__(self, seq_len: int = 20, feature_dim: int = 5):
        super(ImprovedDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Camadas convolucionais para features espaciais
        self.conv_layers = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        
        # LSTM para dependências temporais
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 por causa do bidirectional
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, trajectories):
        batch_size = trajectories.size(0)
        
        # Conv1D para features espaciais: [batch, feature_dim, seq_len]
        x = trajectories.transpose(1, 2)
        conv_out = self.conv_layers(x)  # [batch, 256, seq_len]
        
        # Voltar para formato LSTM: [batch, seq_len, 256]
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM para dependências temporais
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, (hidden, _) = self.lstm(conv_out)
        
        # Usar último estado oculto
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, 256]
        
        # Classificação final
        output = self.classifier(final_hidden)
        
        return output.squeeze()


class ImprovedWGANGPAnomalyDetector:
    """Sistema melhorado de detecção de anomalias"""
    
    def __init__(self, device: str = None, seq_len: int = 20, feature_dim: int = 5):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        print(f"🔧 Inicializando WGAN-GP melhorado no device: {self.device}")
        
        # Hiperparâmetros otimizados
        self.noise_dim = 128
        self.lr_g = 0.0001    # Reduzido para estabilidade
        self.lr_d = 0.0004    # Maior para discriminador
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.lambda_gp = 10
        self.n_critic = 5     # Voltar para 5 para melhor treinamento
        
        # Inicializar redes
        self.generator = ImprovedGenerator(self.noise_dim, seq_len, feature_dim).to(self.device)
        self.discriminator = ImprovedDiscriminator(seq_len, feature_dim).to(self.device)
        
        # Otimizadores
        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))
        
        # Histórico
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': []
        }
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Gradient penalty otimizado"""
        batch_size = real_data.size(0)
        
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Desabilitar CuDNN temporariamente para o forward pass
        with torch.backends.cudnn.flags(enabled=False):
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
        """Treinamento otimizado de uma época"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_wd = []
        epoch_gp = []
        
        pbar = tqdm(dataloader, desc="Treinando", unit="batch", leave=False)
        
        for batch_idx, real_data in enumerate(pbar):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Treinar Discriminador
            for _ in range(self.n_critic):
                self.optim_D.zero_grad()
                
                # Dados reais
                with torch.backends.cudnn.flags(enabled=False):
                    d_real = self.discriminator(real_data)
                
                # Dados falsos
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                with torch.backends.cudnn.flags(enabled=False):
                    fake_data = self.generator(noise).detach()
                    d_fake = self.discriminator(fake_data)
                
                # Wasserstein loss
                wasserstein_d = torch.mean(d_real) - torch.mean(d_fake)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                
                # Loss total
                d_loss = -wasserstein_d + self.lambda_gp * gp
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                self.optim_D.step()
            
            # Treinar Gerador
            self.optim_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            with torch.backends.cudnn.flags(enabled=False):
                fake_data = self.generator(noise)
                d_fake = self.discriminator(fake_data)
            
            g_loss = -torch.mean(d_fake)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            self.optim_G.step()
            
            # Registrar métricas
            epoch_g_loss.append(g_loss.item())
            epoch_d_loss.append(d_loss.item())
            epoch_wd.append(-wasserstein_d.item())
            epoch_gp.append(gp.item())
            
            # Atualizar barra menos frequentemente
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
    
    def train(self, dataset: BalancedTrajectoryDataset, epochs: int = 50, batch_size: int = 64, 
              save_dir: str = "models"):
        """Treinamento completo do modelo"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # DataLoader otimizado
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=2, pin_memory=True)
        
        print(f"🚀 Iniciando treinamento WGAN-GP melhorado:")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches por época: {len(dataloader)}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        best_wd = float('inf')
        
        for epoch in range(epochs):
            print(f"\n📊 Época {epoch+1}/{epochs}")
            
            metrics = self.train_epoch(dataloader)
            
            # Registrar histórico
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Imprimir métricas
            print(f"   G Loss: {metrics['g_loss']:.4f}")
            print(f"   D Loss: {metrics['d_loss']:.4f}")
            print(f"   Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
            print(f"   Gradient Penalty: {metrics['gradient_penalty']:.4f}")
            
            # Salvar melhor modelo
            if metrics['wasserstein_distance'] < best_wd:
                best_wd = metrics['wasserstein_distance']
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f"   💾 Melhor modelo salvo! WD: {best_wd:.4f}")
            
            # Checkpoint menos frequente
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_model(checkpoint_path)
                print(f"   📁 Checkpoint salvo: {checkpoint_path}")
        
        print("\n🎉 Treinamento concluído!")
        self.plot_training_history(save_dir)
    
    def anomaly_score(self, trajectories: torch.Tensor) -> np.ndarray:
        """Score de anomalia melhorado usando múltiplas métricas"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            trajectories = trajectories.to(self.device)
            batch_size = trajectories.size(0)
            
            # 1. Score do discriminador
            with torch.backends.cudnn.flags(enabled=False):
                d_score = self.discriminator(trajectories)
            discriminator_score = -d_score  # Inverter
            
            # 2. Score de reconstrução simplificado
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            with torch.backends.cudnn.flags(enabled=False):
                fake_trajectories = self.generator(noise)
            
            # Encontrar trajetória sintética mais próxima para cada real
            reconstruction_scores = []
            for i in range(batch_size):
                real_traj = trajectories[i]
                
                # Calcular distâncias para todas as trajetórias sintéticas
                distances = torch.sum((fake_trajectories - real_traj.unsqueeze(0)) ** 2, dim=(1, 2))
                min_distance = torch.min(distances)
                reconstruction_scores.append(min_distance.item())
            
            reconstruction_score = torch.tensor(reconstruction_scores)
            
            # 3. Combinar scores
            # Normalizar scores
            if len(discriminator_score) > 1:
                disc_norm = (discriminator_score - discriminator_score.min()) / (discriminator_score.max() - discriminator_score.min() + 1e-8)
            else:
                disc_norm = torch.zeros_like(discriminator_score)
            
            if len(reconstruction_score) > 1:
                recon_norm = (reconstruction_score - reconstruction_score.min()) / (reconstruction_score.max() - reconstruction_score.min() + 1e-8)
            else:
                recon_norm = torch.zeros_like(reconstruction_score)
            
            # Score final combinado
            combined_score = 0.7 * disc_norm.cpu() + 0.3 * recon_norm
            
        return combined_score.numpy()
    
    def detect_anomalies(self, test_data: np.ndarray, threshold_percentile: float = 90) -> Dict:
        """Detecção otimizada de anomalias"""
        print(f"🔍 Detectando anomalias em {len(test_data)} trajetórias...")
        
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
        
        print(f"✅ Detecção concluída:")
        print(f"   Threshold (percentil {threshold_percentile}): {threshold:.4f}")
        print(f"   Anomalias detectadas: {results['n_anomalies']}/{len(test_data)}")
        print(f"   Taxa de anomalia: {results['anomaly_rate']:.2%}")
        
        return results
    
    def evaluate_with_labels(self, test_data: np.ndarray, true_labels: np.ndarray, 
                           threshold_percentile: float = 90) -> Dict:
        """Avaliação com rótulos verdadeiros"""
        print(f"🔍 Avaliando modelo com {len(test_data)} trajetórias...")
        
        test_tensor = torch.FloatTensor(test_data)
        anomaly_scores = self.anomaly_score(test_tensor)
        
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        predicted_labels = anomaly_scores > threshold
        
        # Métricas
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
            'threshold_percentile': threshold_percentile,
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
        
        print(f"✅ Avaliação concluída:")
        print(f"   Precisão: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        
        return results
    
    def save_model(self, filepath: str):
        """Salvar modelo"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'history': self.history,
            'hyperparameters': {
                'noise_dim': self.noise_dim,
                'lr_g': self.lr_g,
                'lr_d': self.lr_d,
                'lambda_gp': self.lambda_gp,
                'n_critic': self.n_critic
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Carregar modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
        self.history = checkpoint['history']
        
        # Carregar hiperparâmetros se disponíveis
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.noise_dim = hyperparams.get('noise_dim', self.noise_dim)
            self.lr_g = hyperparams.get('lr_g', self.lr_g)
            self.lr_d = hyperparams.get('lr_d', self.lr_d)
            self.lambda_gp = hyperparams.get('lambda_gp', self.lambda_gp)
            self.n_critic = hyperparams.get('n_critic', self.n_critic)
        
        print(f"✅ Modelo carregado de: {filepath}")
    
    def plot_training_history(self, save_dir: str):
        """Plot do histórico de treinamento"""
        if not self.history['g_loss']:
            print("⚠️ Nenhum histórico de treinamento para plotar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator Loss
        axes[0, 0].plot(self.history['g_loss'], color='blue', linewidth=2)
        axes[0, 0].set_title('Generator Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator Loss
        axes[0, 1].plot(self.history['d_loss'], color='red', linewidth=2)
        axes[0, 1].set_title('Discriminator Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Wasserstein Distance
        axes[1, 0].plot(self.history['wasserstein_distance'], color='green', linewidth=2)
        axes[1, 0].set_title('Wasserstein Distance', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Distance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient Penalty
        axes[1, 1].plot(self.history['gradient_penalty'], color='orange', linewidth=2)
        axes[1, 1].set_title('Gradient Penalty', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Penalty')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Histórico de treinamento salvo em: {plot_path}")
        
        # Salvar histórico em JSON
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"📄 Histórico JSON salvo em: {history_path}")


def main():
    """Função principal melhorada"""
    
    # Configurações
    DATA_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    ANOMALY_CSV = r"D:\UTFPR\TCC\AI-City Challenge\train-anomaly-results.csv"
    MODEL_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\wganGp_improved"
    EPOCHS = 50
    BATCH_SIZE = 64  # Reduzido para melhor estabilidade
    
    print("🎯 SISTEMA MELHORADO DE DETECÇÃO DE ANOMALIAS COM WGAN-GP")
    print("=" * 60)
    
    try:
        # Carregar labels melhorados
        anomaly_labels = None
        if os.path.exists(ANOMALY_CSV):
            anomaly_labels = ImprovedAnomalyLabels(ANOMALY_CSV)
        else:
            print("⚠️ Arquivo de anomalias não encontrado. Procedendo sem rótulos.")
        
        # Dataset balanceado
        print("\n📂 Carregando dataset balanceado...")
        dataset = BalancedTrajectoryDataset(
            DATA_DIR, 
            anomaly_labels, 
            train_split=0.8,
            use_only_normal_videos=True,
            max_trajectories_per_video=200,
            balance_ratio=0.3  # 30% de anomalias na validação
        )
        
        # Detector melhorado
        print("\n🤖 Inicializando WGAN-GP melhorado...")
        detector = ImprovedWGANGPAnomalyDetector(seq_len=20, feature_dim=5)
        
        # Treinamento com mais épocas
        print("\n🚀 Iniciando treinamento melhorado...")
        detector.train(
            dataset=dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            save_dir=MODEL_DIR
        )
        
        # Avaliação com múltiplos thresholds
        if anomaly_labels:
            print("\n🔍 Avaliando modelo melhorado...")
            val_data, val_labels = dataset.get_val_data()
            val_data_np = val_data.numpy()
            
            best_f1 = 0
            best_threshold = 90
            
            for percentile in [85, 90, 95, 97, 99]:
                results = detector.evaluate_with_labels(val_data_np, val_labels, threshold_percentile=percentile)
                
                if results['f1_score'] > best_f1:
                    best_f1 = results['f1_score']
                    best_threshold = percentile
                
                print(f"   Percentil {percentile}: F1={results['f1_score']:.3f}, Precisão={results['precision']:.3f}, Recall={results['recall']:.3f}")
            
            print(f"\n🏆 Melhor threshold: {best_threshold} percentil (F1-Score: {best_f1:.3f})")
            
            # Salvar melhor resultado
            best_results = detector.evaluate_with_labels(val_data_np, val_labels, threshold_percentile=best_threshold)
            
            results_file = os.path.join(MODEL_DIR, 'improved_evaluation_results.json')
            with open(results_file, 'w') as f:
                json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                              for k, v in best_results.items() 
                              if k not in ['anomaly_scores', 'predicted_labels', 'true_labels']}
                json.dump(json_results, f, indent=2)
        else:
            print("\n⚠️ Sem rótulos de anomalias - apenas teste de detecção...")
            val_data, _ = dataset.get_val_data()
            val_data_np = val_data.numpy()
            
            results = detector.detect_anomalies(val_data_np, threshold_percentile=90)
            
            # Salvar resultados básicos
            results_file = os.path.join(MODEL_DIR, 'detection_results.json')
            with open(results_file, 'w') as f:
                json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                              for k, v in results.items() 
                              if k not in ['anomaly_scores', 'is_anomaly']}
                json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Resultados melhorados salvos em: {MODEL_DIR}")
        print("\n🎉 Processamento melhorado concluído!")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
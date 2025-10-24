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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"🚀 TensorFlow usando GPU: {physical_devices[0]}")
else:
    print("🔧 TensorFlow usando CPU")

class TrafficAnomalyLabels:
    """Gerenciador otimizado de rótulos de anomalias de tráfego"""
    
    def __init__(self, csv_path: str, fps: float = 30.0):
        self.csv_path = csv_path
        self.fps = fps
        self.anomaly_intervals = {}
        self.video_stats = {}
        
        print(f"📋 Carregando ground truth: {csv_path}")
        self._load_anomaly_data()
    
    def _load_anomaly_data(self):
        """Carrega e processa dados de anomalias"""
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
                        'severity_score': 0
                    }
                
                self.anomaly_intervals[video_id].append((start_time, end_time))
                
                # Atualizar estatísticas
                stats = self.video_stats[video_id]
                stats['total_anomaly_time'] += duration
                stats['anomaly_count'] += 1
                stats['severity_score'] = np.log1p(stats['total_anomaly_time']) * stats['anomaly_count']
            
            print(f"✅ {len(self.anomaly_intervals)} vídeos com anomalias carregados")
            
        except Exception as e:
            print(f"❌ Erro ao carregar anomalias: {e}")
            self.anomaly_intervals = {}
    
    def has_anomalies(self, video_id: int) -> bool:
        return video_id in self.anomaly_intervals
    
    def get_severity_score(self, video_id: int) -> float:
        return self.video_stats.get(video_id, {}).get('severity_score', 0.0)


class TrafficTrajectoryDataset:
    """Dataset otimizado para trajetórias de tráfego"""
    
    def __init__(self, data_dir: str, anomaly_labels: Optional[TrafficAnomalyLabels] = None, 
                 max_trajectories_per_video: int = 500, validation_split: float = 0.2):
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.max_trajectories_per_video = max_trajectories_per_video
        self.validation_split = validation_split
        
        # Scaler para normalização
        self.scaler = StandardScaler()
        
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Carrega e processa dados de trajetórias"""
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('_trajectories_processed.npy')]
        npy_files.sort(key=lambda x: int(x.split('_')[0]))
        
        print(f"📁 Processando {len(npy_files)} arquivos de trajetórias...")
        
        normal_trajectories = []
        anomaly_trajectories = []
        
        for npy_file in npy_files:
            video_id = int(npy_file.split('_')[0])
            file_path = os.path.join(self.data_dir, npy_file)
            
            try:
                trajectories = np.load(file_path)
                
                if len(trajectories) == 0:
                    continue
                
                # Limitar quantidade por arquivo
                if len(trajectories) > self.max_trajectories_per_video:
                    indices = np.random.choice(len(trajectories), self.max_trajectories_per_video, replace=False)
                    trajectories = trajectories[indices]
                
                # Verificar qualidade dos dados
                if not self._is_valid_trajectory_data(trajectories):
                    print(f"⚠️ Dados inválidos em {npy_file}")
                    continue
                
                # Classificar como normal ou anômalo
                if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                    anomaly_trajectories.append(trajectories)
                    print(f"   📹 {npy_file}: {len(trajectories)} trajetórias ANÔMALAS")
                else:
                    normal_trajectories.append(trajectories)
                    print(f"   📹 {npy_file}: {len(trajectories)} trajetórias normais")
                    
            except Exception as e:
                print(f"❌ Erro ao carregar {npy_file}: {e}")
        
        # Preparar dados finais
        self._prepare_final_datasets(normal_trajectories, anomaly_trajectories)
        
        print(f"✅ Dataset preparado:")
        print(f"   Treino: {self.train_data.shape}")
        print(f"   Validação: {self.val_data.shape}")
        print(f"   Rótulos val - Normal: {np.sum(~self.val_labels)}, Anômalo: {np.sum(self.val_labels)}")
    
    def _is_valid_trajectory_data(self, trajectories: np.ndarray) -> bool:
        """Verifica se os dados de trajetória são válidos"""
        if trajectories.size == 0:
            return False
        
        # Verificar NaN/Inf
        if np.any(np.isnan(trajectories)) or np.any(np.isinf(trajectories)):
            return False
        
        # Verificar dimensões
        if len(trajectories.shape) != 3:
            return False
        
        return True
    
    def _prepare_final_datasets(self, normal_trajectories: List, anomaly_trajectories: List):
        """Prepara datasets finais de treino e validação"""
        
        # Concatenar trajetórias normais
        if normal_trajectories:
            all_normal = np.concatenate(normal_trajectories, axis=0)
        else:
            raise ValueError("Nenhuma trajetória normal encontrada!")
        
        # Dividir trajetórias normais em treino e validação
        n_normal = len(all_normal)
        n_val_normal = int(n_normal * self.validation_split)
        
        indices = np.random.permutation(n_normal)
        val_normal_indices = indices[:n_val_normal]
        train_indices = indices[n_val_normal:]
        
        # Dados de treino (apenas normais)
        self.train_data = all_normal[train_indices]
        val_normal_data = all_normal[val_normal_indices]
        
        # Preparar dados de validação balanceados
        if anomaly_trajectories:
            all_anomalies = np.concatenate(anomaly_trajectories, axis=0)
            
            # Balancear: usar proporção similar de anomalias
            n_val_anomaly = min(len(all_anomalies), n_val_normal // 2)  # 33% anomalias
            
            if n_val_anomaly > 0:
                anomaly_indices = np.random.choice(len(all_anomalies), n_val_anomaly, replace=False)
                val_anomaly_data = all_anomalies[anomaly_indices]
                
                # Combinar validação
                self.val_data = np.concatenate([val_normal_data, val_anomaly_data], axis=0)
                self.val_labels = np.concatenate([
                    np.zeros(len(val_normal_data), dtype=bool),
                    np.ones(len(val_anomaly_data), dtype=bool)
                ])
            else:
                self.val_data = val_normal_data
                self.val_labels = np.zeros(len(val_normal_data), dtype=bool)
        else:
            self.val_data = val_normal_data
            self.val_labels = np.zeros(len(val_normal_data), dtype=bool)
        
        # Embaralhar validação
        val_indices = np.random.permutation(len(self.val_data))
        self.val_data = self.val_data[val_indices]
        self.val_labels = self.val_labels[val_indices]
        
        # Normalizar dados
        self._normalize_data()
    
    def _normalize_data(self):
        """Normaliza os dados usando StandardScaler"""
        # Reshape para normalização
        original_train_shape = self.train_data.shape
        original_val_shape = self.val_data.shape
        
        train_flat = self.train_data.reshape(-1, original_train_shape[-1])
        val_flat = self.val_data.reshape(-1, original_val_shape[-1])
        
        # Fit no treino, transform em ambos
        train_normalized = self.scaler.fit_transform(train_flat)
        val_normalized = self.scaler.transform(val_flat)
        
        # Reshape de volta
        self.train_data = train_normalized.reshape(original_train_shape)
        self.val_data = val_normalized.reshape(original_val_shape)
        
        print("🔧 Dados normalizados com StandardScaler")
    
    def get_train_dataset(self, batch_size: int = 64):
        """Retorna dataset TensorFlow para treino"""
        dataset = tf.data.Dataset.from_tensor_slices(self.train_data.astype(np.float32))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def get_validation_data(self):
        """Retorna dados de validação"""
        return self.val_data.astype(np.float32), self.val_labels


# =============================================================================
# DEFININDO PARÂMETROS
# =============================================================================

# Parâmetros da arquitetura
SEQUENCE_LENGTH = 20    # Comprimento das sequências de trajetória
FEATURE_DIM = 5        # Dimensões das features (x, y, vx, vy, etc.)
NOISE_DIM = 128        # Dimensão do ruído de entrada
BATCH_SIZE = 64        # Tamanho do batch

# Parâmetros de treinamento
EPOCHS = 100           # Número de épocas
D_STEPS = 2           # Passos do discriminador por passo do gerador
GP_WEIGHT = 5.0      # Peso do gradient penalty
LEARNING_RATE_G = 0.0002  # Taxa de aprendizado do gerador
LEARNING_RATE_D = 0.0001  # Taxa de aprendizado do discriminador

print("🔧 Parâmetros definidos:")
print(f"   Sequence Length: {SEQUENCE_LENGTH}")
print(f"   Feature Dim: {FEATURE_DIM}")
print(f"   Noise Dim: {NOISE_DIM}")
print(f"   Batch Size: {BATCH_SIZE}")

# =============================================================================
# CONSTRUINDO A ARQUITETURA DO GERADOR
# =============================================================================

def build_generator(noise_dim: int, sequence_length: int, feature_dim: int):
    """
    Constrói o gerador para sequências de trajetórias
    """
    
    # Input layer
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
    
    # Dense layers para projeção inicial
    x = layers.Dense(256, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Projeção para sequência temporal
    x = layers.Dense(sequence_length * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Reshape para sequência
    x = layers.Reshape((sequence_length, 256))(x)
    
    # LSTM layers para dependências temporais
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    trajectory_output = layers.TimeDistributed(
        layers.Dense(feature_dim, activation='tanh')
    )(x)
    
    generator = keras.Model(
        inputs=noise_input, 
        outputs=trajectory_output, 
        name="trajectory_generator"
    )
    
    return generator

# Criar gerador
generator = build_generator(NOISE_DIM, SEQUENCE_LENGTH, FEATURE_DIM)
generator.summary()

# =============================================================================
# CONSTRUINDO A ARQUITETURA DO DISCRIMINADOR
# =============================================================================

def build_discriminator(sequence_length: int, feature_dim: int):
    """
    Constrói o discriminador para sequências de trajetórias
    """
    
    # Input layer
    trajectory_input = layers.Input(
        shape=(sequence_length, feature_dim), 
        name="trajectory_input"
    )
    
    # Convolutional layers para features espaciais-temporais
    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(trajectory_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # LSTM para capturar dependências temporais
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64)(x)  # return_sequences=False para último estado
    x = layers.Dropout(0.3)(x)
    
    # Dense layers finais
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (sem ativação para WGAN)
    validity = layers.Dense(1, name="validity")(x)
    
    discriminator = keras.Model(
        inputs=trajectory_input, 
        outputs=validity, 
        name="trajectory_discriminator"
    )
    
    return discriminator

# Criar discriminador
discriminator = build_discriminator(SEQUENCE_LENGTH, FEATURE_DIM)
discriminator.summary()

# =============================================================================
# CRIANDO O MODELO WGAN GERAL
# =============================================================================

class TrafficWGAN(keras.Model):
    """
    Implementação completa do WGAN-GP para detecção de anomalias em tráfego
    """
    
    def __init__(self, discriminator, generator, noise_dim, gp_weight=10.0, d_steps=5):
        super(TrafficWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight
        self.d_steps = d_steps
        
        # Métricas para tracking
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_tracker = keras.metrics.Mean(name="gradient_penalty")
        self.wd_tracker = keras.metrics.Mean(name="wasserstein_distance")
    
    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.gp_tracker, self.wd_tracker]
    
    def compile(self, d_optimizer, g_optimizer):
        super(TrafficWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def gradient_penalty(self, batch_size, real_trajectories, fake_trajectories):
        """Calcula gradient penalty para WGAN-GP"""
        
        # Gerar valores alpha aleatórios
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        
        # Interpolação entre trajetórias reais e falsas
        interpolated = alpha * real_trajectories + (1 - alpha) * fake_trajectories
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        # Calcular gradientes
        grads = tape.gradient(pred, [interpolated])[0]
        
        # Calcular norma dos gradientes
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        
        # Gradient penalty
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
    
    def train_step(self, real_trajectories):
        """Um passo de treinamento do WGAN-GP"""
        
        batch_size = tf.shape(real_trajectories)[0]
        
        # Treinar discriminador múltiplas vezes
        for _ in range(self.d_steps):
            
            # Gerar ruído aleatório
            random_noise = tf.random.normal([batch_size, self.noise_dim])
            
            with tf.GradientTape() as tape:
                # Gerar trajetórias falsas
                fake_trajectories = self.generator(random_noise, training=True)
                
                # Obter predições do discriminador
                real_pred = self.discriminator(real_trajectories, training=True)
                fake_pred = self.discriminator(fake_trajectories, training=True)
                
                # Calcular Wasserstein loss
                d_cost = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                
                # Calcular gradient penalty
                gp = self.gradient_penalty(batch_size, real_trajectories, fake_trajectories)
                
                # Loss total do discriminador
                d_loss = d_cost + gp * self.gp_weight
            
            # Calcular gradientes e atualizar discriminador
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        # Treinar gerador
        random_noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as tape:
            # Gerar trajetórias falsas
            fake_trajectories = self.generator(random_noise, training=True)
            
            # Obter predição do discriminador
            fake_pred = self.discriminator(fake_trajectories, training=True)
            
            # Loss do gerador (quer maximizar fake_pred)
            g_loss = -tf.reduce_mean(fake_pred)
        
        # Calcular gradientes e atualizar gerador
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )
        
        # Atualizar métricas
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.gp_tracker.update_state(gp)
        self.wd_tracker.update_state(-d_cost)  # Wasserstein distance
        
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "gradient_penalty": self.gp_tracker.result(),
            "wasserstein_distance": self.wd_tracker.result(),
        }


# Criar modelo WGAN-GP
wgan = TrafficWGAN(
    discriminator=discriminator,
    generator=generator,
    noise_dim=NOISE_DIM,
    gp_weight=GP_WEIGHT,
    d_steps=D_STEPS
)

# Compilar com otimizadores
wgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=0.0, beta_2=0.9),
    g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=0.0, beta_2=0.9)
)

print("🤖 Modelo WGAN-GP criado e compilado!")

# =============================================================================
# CLASSE PARA DETECÇÃO DE ANOMALIAS
# =============================================================================

class TrafficAnomalyDetector:
    """Detector de anomalias usando WGAN-GP treinado"""
    
    def __init__(self, wgan_model: TrafficWGAN, scaler: StandardScaler):
        self.wgan = wgan_model
        self.scaler = scaler
        self.generator = wgan_model.generator
        self.discriminator = wgan_model.discriminator
    
    def compute_anomaly_scores(self, trajectories: np.ndarray, 
                             n_samples: int = 50) -> np.ndarray:
        """
        Computa scores de anomalia para trajetórias
        
        Args:
            trajectories: Array de trajetórias [n_traj, seq_len, features]
            n_samples: Número de amostras sintéticas para comparação
            
        Returns:
            Array de scores de anomalia
        """
        
        # Normalizar trajetórias
        original_shape = trajectories.shape
        trajectories_flat = trajectories.reshape(-1, original_shape[-1])
        trajectories_normalized = self.scaler.transform(trajectories_flat)
        trajectories_normalized = trajectories_normalized.reshape(original_shape)
        
        trajectories_tensor = tf.constant(trajectories_normalized, dtype=tf.float32)
        
        # 1. Score do discriminador (invertido)
        discriminator_scores = -self.discriminator(trajectories_tensor, training=False)
        discriminator_scores = discriminator_scores.numpy().flatten()
        
        # 2. Score de reconstrução baseado em distância
        reconstruction_scores = []
        
        for i in range(len(trajectories)):
            real_traj = trajectories_normalized[i]
            
            # Gerar múltiplas trajetórias sintéticas
            noise = tf.random.normal([n_samples, self.wgan.noise_dim])
            synthetic_trajs = self.generator(noise, training=False).numpy()
            
            # Calcular distâncias
            distances = np.sum((synthetic_trajs - real_traj[np.newaxis, :, :]) ** 2, axis=(1, 2))
            min_distance = np.min(distances)
            reconstruction_scores.append(min_distance)
        
        reconstruction_scores = np.array(reconstruction_scores)
        
        # 3. Combinar scores
        # Normalizar scores para [0, 1]
        def normalize_scores(scores):
            if len(scores) <= 1:
                return np.zeros_like(scores)
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score == min_score:
                return np.zeros_like(scores)
            return (scores - min_score) / (max_score - min_score)
        
        disc_norm = normalize_scores(discriminator_scores)
        recon_norm = normalize_scores(reconstruction_scores)
        
        # Score final combinado (70% discriminador, 30% reconstrução)
        combined_scores = 0.7 * disc_norm + 0.3 * recon_norm
        
        return combined_scores
    
    def detect_anomalies(self, trajectories: np.ndarray, 
                        threshold_percentile: float = 90) -> Dict:
        """Detecta anomalias em trajetórias"""
        
        print(f"🔍 Detectando anomalias em {len(trajectories)} trajetórias...")
        
        anomaly_scores = self.compute_anomaly_scores(trajectories)
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        is_anomaly = anomaly_scores > threshold
        
        results = {
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'threshold_percentile': threshold_percentile
        }
        
        print(f"✅ Detecção concluída:")
        print(f"   Threshold: {threshold:.4f}")
        print(f"   Anomalias: {results['n_anomalies']}/{len(trajectories)}")
        print(f"   Taxa: {results['anomaly_rate']:.2%}")
        
        return results
    
    def evaluate_with_labels(self, trajectories: np.ndarray, 
                           true_labels: np.ndarray,
                           threshold_percentile: float = 90) -> Dict:
        """Avalia modelo com rótulos verdadeiros"""
        
        anomaly_scores = self.compute_anomaly_scores(trajectories)
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        predicted_labels = anomaly_scores > threshold
        
        # Calcular métricas
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
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'anomaly_scores': anomaly_scores,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        }
        
        print(f"📊 Avaliação:")
        print(f"   Precisão: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        
        return results


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal para treinar e avaliar o modelo"""
    
    # Configurações de paths
    DATA_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    ANOMALY_CSV = r"D:\UTFPR\TCC\AI-City Challenge\train-anomaly-results.csv"
    MODEL_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\tensorflow_wgan_gp"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("🚗 SISTEMA TENSORFLOW WGAN-GP PARA DETECÇÃO DE ANOMALIAS EM TRÁFEGO")
    print("=" * 70)
    
    try:
        # Carregar labels de anomalias
        anomaly_labels = None
        if os.path.exists(ANOMALY_CSV):
            anomaly_labels = TrafficAnomalyLabels(ANOMALY_CSV)
        else:
            print("⚠️ Arquivo de anomalias não encontrado")
        
        # Carregar dataset
        print("\n📂 Carregando dataset de trajetórias...")
        dataset = TrafficTrajectoryDataset(
            DATA_DIR, 
            anomaly_labels, 
            max_trajectories_per_video=300,
            validation_split=0.2
        )
        
        # Preparar dados de treino
        train_dataset = dataset.get_train_dataset(BATCH_SIZE)
        val_data, val_labels = dataset.get_validation_data()
        
        print(f"\n🚀 Iniciando treinamento por {EPOCHS} épocas...")
        
        # Callback para salvar melhor modelo
        class ModelSaveCallback(keras.callbacks.Callback):
            def __init__(self, save_path):
                self.save_path = save_path
                self.best_wd = float('inf')
            
            def on_epoch_end(self, epoch, logs=None):
                current_wd = logs.get('wasserstein_distance', float('inf'))
                if current_wd < self.best_wd:
                    self.best_wd = current_wd
                    # FIX: Remove save_format and use .keras extension
                    self.model.generator.save(os.path.join(self.save_path, 'best_generator.keras'))
                    self.model.discriminator.save(os.path.join(self.save_path, 'best_discriminator.keras'))
                    print(f"\n💾 Melhor modelo salvo! WD: {current_wd:.4f}")

        save_callback = ModelSaveCallback(MODEL_DIR)
        
        # Treinar modelo
        history = wgan.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=[save_callback],
            verbose=1
        )
        
        print("\n🎉 Treinamento concluído!")
        
        # Salvar histórico
        history_path = os.path.join(MODEL_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            # Converter numpy arrays para listas
            hist_dict = {}
            for key, values in history.history.items():
                hist_dict[key] = [float(v) for v in values]
            json.dump(hist_dict, f, indent=2)
        
        # Plotar histórico de treinamento
        plot_training_history(history, MODEL_DIR)
        
        # Carregar melhor modelo para avaliação
        wgan.generator = keras.models.load_model(os.path.join(MODEL_DIR, 'best_generator.keras'))
        wgan.discriminator = keras.models.load_model(os.path.join(MODEL_DIR, 'best_discriminator.keras'))

        # Criar detector
        detector = TrafficAnomalyDetector(wgan, dataset.scaler)
        
        # Avaliar modelo
        if anomaly_labels:
            print("\n🔍 Avaliando modelo...")
            
            best_f1 = 0
            best_results = None
            
            # 3. Try different thresholds
            for percentile in [75, 80, 85, 90]:
                results = detector.evaluate_with_labels(val_data, val_labels, percentile)
                
                if results['f1_score'] > best_f1:
                    best_f1 = results['f1_score']
                    best_results = results
                
                print(f"   {percentile}%: F1={results['f1_score']:.3f}")
            
            print(f"\n🏆 Melhor F1-Score: {best_f1:.3f}")
            
            # Salvar resultados
            results_path = os.path.join(MODEL_DIR, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                save_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                              for k, v in best_results.items() 
                              if k not in ['anomaly_scores', 'predicted_labels', 'true_labels']}
                json.dump(save_results, f, indent=2)
        
        else:
            print("\n🔍 Teste de detecção (sem rótulos)...")
            results = detector.detect_anomalies(val_data)
        
        print(f"\n📄 Resultados salvos em: {MODEL_DIR}")
        print("✅ Processamento concluído!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


def plot_training_history(history, save_dir):
    """Plota histórico de treinamento"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator Loss
    axes[0, 0].plot(history.history['g_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Generator Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Loss
    axes[0, 1].plot(history.history['d_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Discriminator Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Wasserstein Distance
    axes[1, 0].plot(history.history['wasserstein_distance'], 'g-', linewidth=2)
    axes[1, 0].set_title('Wasserstein Distance', fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Penalty
    axes[1, 1].plot(history.history['gradient_penalty'], 'orange', linewidth=2)
    axes[1, 1].set_title('Gradient Penalty', fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Penalty')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráficos salvos em: {os.path.join(save_dir, 'training_history.png')}")


if __name__ == "__main__":
    main()
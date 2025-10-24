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
    print(f"🚀 TensorFlow usando GPU: {physical_devices[0]}")
else:
    print("🔧 TensorFlow usando CPU")

class ImprovedTrafficTrajectoryDataset:
    """Dataset melhorado para trajetórias de tráfego com foco em dados normais"""
    
    def __init__(self, data_dir: str, anomaly_labels: Optional['TrafficAnomalyLabels'] = None, 
                 max_trajectories_per_video: int = 800, validation_split: float = 0.15):
        self.data_dir = data_dir
        self.anomaly_labels = anomaly_labels
        self.max_trajectories_per_video = max_trajectories_per_video
        self.validation_split = validation_split
        
        # Usar RobustScaler para melhor normalização
        self.scaler = RobustScaler()
        
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Carrega e processa dados priorizando qualidade dos dados normais"""
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
                
                # Filtrar trajetórias de qualidade
                quality_trajectories = self._filter_quality_trajectories(trajectories)
                
                if len(quality_trajectories) == 0:
                    continue
                
                # Limitar quantidade por arquivo (mais para normais)
                max_per_file = self.max_trajectories_per_video
                if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                    max_per_file = min(200, len(quality_trajectories))  # Menos anomalias
                
                if len(quality_trajectories) > max_per_file:
                    indices = np.random.choice(len(quality_trajectories), max_per_file, replace=False)
                    quality_trajectories = quality_trajectories[indices]
                
                # Classificar como normal ou anômalo
                if self.anomaly_labels and self.anomaly_labels.has_anomalies(video_id):
                    anomaly_trajectories.append(quality_trajectories)
                    print(f"   📹 {npy_file}: {len(quality_trajectories)} trajetórias ANÔMALAS")
                else:
                    normal_trajectories.append(quality_trajectories)
                    print(f"   📹 {npy_file}: {len(quality_trajectories)} trajetórias normais")
                    
            except Exception as e:
                print(f"❌ Erro ao carregar {npy_file}: {e}")
        
        # Preparar dados finais
        self._prepare_final_datasets(normal_trajectories, anomaly_trajectories)
        
        print(f"✅ Dataset preparado:")
        print(f"   Treino (só normais): {self.train_data.shape}")
        print(f"   Validação: {self.val_data.shape}")
        print(f"   Rótulos val - Normal: {np.sum(~self.val_labels)}, Anômalo: {np.sum(self.val_labels)}")
    
    def _filter_quality_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """Filtra trajetórias de boa qualidade"""
        good_trajectories = []
        
        for traj in trajectories:
            # Verificar se não tem valores inválidos
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                continue
            
            # Verificar se há movimento suficiente (não estacionário)
            pos_variance = np.var(traj[:, :2], axis=0)  # variância em x,y
            if np.sum(pos_variance) < 1e-6:  # muito pouco movimento
                continue
            
            # Verificar se velocidades não são extremas
            velocities = traj[:, 2]  # coluna de velocidade
            if np.max(velocities) > 10 or np.std(velocities) > 5:  # valores muito altos
                continue
            
            # Verificar continuidade temporal (sem gaps grandes)
            accelerations = traj[:, 3]  # coluna de aceleração
            if np.max(np.abs(accelerations)) > 8:  # aceleração muito abrupta
                continue
            
            good_trajectories.append(traj)
        
        return np.array(good_trajectories) if good_trajectories else np.empty((0, 20, 5))
    
    def _prepare_final_datasets(self, normal_trajectories: List, anomaly_trajectories: List):
        """Prepara datasets com foco em dados normais de qualidade"""
        
        if not normal_trajectories:
            raise ValueError("Nenhuma trajetória normal encontrada!")
        
        # Concatenar trajetórias normais
        all_normal = np.concatenate(normal_trajectories, axis=0)
        print(f"📊 Total de trajetórias normais: {len(all_normal)}")
        
        # Dividir normais em treino e validação
        n_normal = len(all_normal)
        n_val_normal = int(n_normal * self.validation_split)
        
        indices = np.random.permutation(n_normal)
        val_normal_indices = indices[:n_val_normal]
        train_indices = indices[n_val_normal:]
        
        # TREINO: apenas dados normais (chave do WGAN-GP!)
        self.train_data = all_normal[train_indices]
        val_normal_data = all_normal[val_normal_indices]
        
        # VALIDAÇÃO: mistura balanceada
        if anomaly_trajectories:
            all_anomalies = np.concatenate(anomaly_trajectories, axis=0)
            print(f"📊 Total de trajetórias anômalas: {len(all_anomalies)}")
            
            # Usar proporção 70% normal, 30% anomalia na validação
            n_val_anomaly = min(len(all_anomalies), int(n_val_normal * 0.4))
            
            if n_val_anomaly > 0:
                anomaly_indices = np.random.choice(len(all_anomalies), n_val_anomaly, replace=False)
                val_anomaly_data = all_anomalies[anomaly_indices]
                
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
        """Normaliza os dados usando RobustScaler (melhor para outliers)"""
        original_train_shape = self.train_data.shape
        original_val_shape = self.val_data.shape
        
        # Flatten para normalização
        train_flat = self.train_data.reshape(-1, original_train_shape[-1])
        val_flat = self.val_data.reshape(-1, original_val_shape[-1])
        
        # Fit apenas nos dados de treino (normais)
        train_normalized = self.scaler.fit_transform(train_flat)
        val_normalized = self.scaler.transform(val_flat)
        
        # Reshape de volta
        self.train_data = train_normalized.reshape(original_train_shape)
        self.val_data = val_normalized.reshape(original_val_shape)
        
        print("🔧 Dados normalizados com RobustScaler")
        print(f"   Dados treino - mean: {np.mean(self.train_data):.3f}, std: {np.std(self.train_data):.3f}")
    
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
# ARQUITETURA MELHORADA DO GERADOR
# =============================================================================

def build_improved_generator(noise_dim: int, sequence_length: int, feature_dim: int):
    """Gerador melhorado com atenção temporal"""
    
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
    
    # Projeção inicial mais robusta
    x = layers.Dense(512, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(1024, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Projeção para sequência
    x = layers.Dense(sequence_length * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((sequence_length, 128))(x)
    
    # Camadas LSTM bidirecionais para melhor modelagem temporal
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    # Camada de atenção temporal
    attention_weights = layers.TimeDistributed(layers.Dense(1, activation='softmax'))(x)
    attended = layers.Multiply()([x, attention_weights])
    
    # Output layer com ativação suave
    trajectory_output = layers.TimeDistributed(
        layers.Dense(feature_dim, activation='tanh')
    )(attended)
    
    generator = keras.Model(inputs=noise_input, outputs=trajectory_output, name="improved_generator")
    return generator


# =============================================================================
# DISCRIMINADOR MELHORADO
# =============================================================================

def build_improved_discriminator(sequence_length: int, feature_dim: int):
    """Discriminador melhorado com múltiplas escalas"""
    
    trajectory_input = layers.Input(shape=(sequence_length, feature_dim), name="trajectory_input")
    
    # Múltiplas escalas convolucionais
    conv_outputs = []
    
    # Escala 1: kernel pequeno
    x1 = layers.Conv1D(64, 3, padding='same')(trajectory_input)
    x1 = layers.LeakyReLU(0.2)(x1)
    x1 = layers.Conv1D(128, 3, strides=2, padding='same')(x1)
    x1 = layers.LeakyReLU(0.2)(x1)
    conv_outputs.append(x1)
    
    # Escala 2: kernel médio
    x2 = layers.Conv1D(64, 5, padding='same')(trajectory_input)
    x2 = layers.LeakyReLU(0.2)(x2)
    x2 = layers.Conv1D(128, 5, strides=2, padding='same')(x2)
    x2 = layers.LeakyReLU(0.2)(x2)
    conv_outputs.append(x2)
    
    # Concatenar diferentes escalas
    x = layers.Concatenate(axis=-1)(conv_outputs)
    x = layers.Dropout(0.3)(x)
    
    # Mais camadas convolucionais
    x = layers.Conv1D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # LSTM para dependências temporais
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    
    # Camadas finais
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    validity = layers.Dense(1)(x)
    
    discriminator = keras.Model(inputs=trajectory_input, outputs=validity, name="improved_discriminator")
    return discriminator


# =============================================================================
# WGAN-GP MELHORADO
# =============================================================================

class ImprovedTrafficWGAN(keras.Model):
    """WGAN-GP melhorado com estabilidade aprimorada"""
    
    def __init__(self, discriminator, generator, noise_dim, gp_weight=10.0, d_steps=3):
        super(ImprovedTrafficWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight
        self.d_steps = d_steps
        
        # Métricas melhoradas
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_tracker = keras.metrics.Mean(name="gradient_penalty")
        self.wd_tracker = keras.metrics.Mean(name="wasserstein_distance")
        
        # Rastreamento de qualidade do gerador
        self.gen_quality_tracker = keras.metrics.Mean(name="generator_quality")
    
    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.gp_tracker, 
                self.wd_tracker, self.gen_quality_tracker]
    
    def compile(self, d_optimizer, g_optimizer):
        super(ImprovedTrafficWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def gradient_penalty(self, batch_size, real_trajectories, fake_trajectories):
        """Gradient penalty melhorado"""
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_trajectories + (1 - alpha) * fake_trajectories
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
    
    def train_step(self, real_trajectories):
        """Passo de treinamento melhorado"""
        batch_size = tf.shape(real_trajectories)[0]
        
        # Treinar discriminador múltiplas vezes
        d_losses = []
        for _ in range(self.d_steps):
            random_noise = tf.random.normal([batch_size, self.noise_dim])
            
            with tf.GradientTape() as tape:
                fake_trajectories = self.generator(random_noise, training=True)
                
                real_pred = self.discriminator(real_trajectories, training=True)
                fake_pred = self.discriminator(fake_trajectories, training=True)
                
                d_cost = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                gp = self.gradient_penalty(batch_size, real_trajectories, fake_trajectories)
                d_loss = d_cost + gp * self.gp_weight
            
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            d_losses.append(d_loss)
        
        # Treinar gerador
        random_noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as tape:
            fake_trajectories = self.generator(random_noise, training=True)
            fake_pred = self.discriminator(fake_trajectories, training=True)
            g_loss = -tf.reduce_mean(fake_pred)
        
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )
        
        # Atualizar métricas
        avg_d_loss = tf.reduce_mean(d_losses)
        self.d_loss_tracker.update_state(avg_d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.gp_tracker.update_state(gp)
        self.wd_tracker.update_state(-d_cost)
        
        # Qualidade do gerador (variabilidade das trajetórias geradas)
        gen_variance = tf.reduce_mean(tf.math.reduce_variance(fake_trajectories, axis=0))
        self.gen_quality_tracker.update_state(gen_variance)
        
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "gradient_penalty": self.gp_tracker.result(),
            "wasserstein_distance": self.wd_tracker.result(),
            "generator_quality": self.gen_quality_tracker.result(),
        }


# =============================================================================
# DETECTOR DE ANOMALIAS MELHORADO
# =============================================================================

class ImprovedAnomalyDetector:
    """Detector de anomalias melhorado com múltiplas estratégias"""
    
    def __init__(self, wgan_model: ImprovedTrafficWGAN, scaler):
        self.wgan = wgan_model
        self.scaler = scaler
        self.generator = wgan_model.generator
        self.discriminator = wgan_model.discriminator
        
        # Gerar dados normais de referência
        self._generate_normal_reference_data()
    
    def _generate_normal_reference_data(self, n_samples: int = 2000):
        """Gera dados normais de referência para comparação"""
        print("🎯 Gerando dados normais de referência...")
        
        noise = tf.random.normal([n_samples, self.wgan.noise_dim])
        self.reference_normal_data = self.generator(noise, training=False).numpy()
        
        print(f"✅ Gerados {n_samples} exemplos normais de referência")
    
    def compute_anomaly_scores(self, trajectories: np.ndarray) -> np.ndarray:
        """Computa scores usando múltiplas estratégias"""
        
        # Normalizar entrada
        original_shape = trajectories.shape
        trajectories_flat = trajectories.reshape(-1, original_shape[-1])
        trajectories_normalized = self.scaler.transform(trajectories_flat)
        trajectories_normalized = trajectories_normalized.reshape(original_shape)
        
        trajectories_tensor = tf.constant(trajectories_normalized, dtype=tf.float32)
        
        # 1. Score do discriminador (principal indicador)
        discriminator_scores = -self.discriminator(trajectories_tensor, training=False)
        discriminator_scores = discriminator_scores.numpy().flatten()
        
        # 2. Distância estatística para dados normais gerados
        statistical_scores = self._compute_statistical_distance(trajectories_normalized)
        
        # 3. Score de densidade local
        density_scores = self._compute_density_scores(trajectories_normalized)
        
        # Normalizar scores
        def robust_normalize(scores):
            q75, q25 = np.percentile(scores, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(scores)
            normalized = (scores - q25) / iqr
            return np.clip(normalized, 0, 5)  # clip extremos
        
        disc_norm = robust_normalize(discriminator_scores)
        stat_norm = robust_normalize(statistical_scores)
        density_norm = robust_normalize(density_scores)
        
        # Combinar scores com pesos otimizados
        combined_scores = 0.6 * disc_norm + 0.25 * stat_norm + 0.15 * density_norm
        
        return combined_scores
    
    def _compute_statistical_distance(self, trajectories: np.ndarray) -> np.ndarray:
        """Computa distância estatística para dados normais"""
        scores = []
        
        # Estatísticas dos dados de referência normais
        ref_means = np.mean(self.reference_normal_data, axis=(0, 1))
        ref_stds = np.std(self.reference_normal_data, axis=(0, 1))
        
        for traj in trajectories:
            traj_mean = np.mean(traj, axis=0)
            traj_std = np.std(traj, axis=0)
            
            # Distância das estatísticas
            mean_distance = np.linalg.norm(traj_mean - ref_means)
            std_distance = np.linalg.norm(traj_std - ref_stds)
            
            score = mean_distance + std_distance
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_density_scores(self, trajectories: np.ndarray) -> np.ndarray:
        """Computa scores baseados em densidade local"""
        scores = []
        
        # Flatten trajetórias para análise de densidade
        ref_flat = self.reference_normal_data.reshape(len(self.reference_normal_data), -1)
        
        for traj in trajectories:
            traj_flat = traj.flatten()
            
            # Calcular distâncias para k vizinhos mais próximos
            distances = np.linalg.norm(ref_flat - traj_flat, axis=1)
            k = min(50, len(distances))
            k_nearest_distances = np.partition(distances, k-1)[:k]
            
            # Score baseado na distância média dos k vizinhos
            density_score = np.mean(k_nearest_distances)
            scores.append(density_score)
        
        return np.array(scores)
    
    def generate_normal_samples(self, n_samples: int = 100) -> np.ndarray:
        """Gera amostras normais usando o gerador treinado"""
        print(f"🎯 Gerando {n_samples} amostras normais...")
        
        noise = tf.random.normal([n_samples, self.wgan.noise_dim])
        generated_samples = self.generator(noise, training=False).numpy()
        
        # Desnormalizar amostras
        original_shape = generated_samples.shape
        samples_flat = generated_samples.reshape(-1, original_shape[-1])
        samples_denormalized = self.scaler.inverse_transform(samples_flat)
        samples_final = samples_denormalized.reshape(original_shape)
        
        print(f"✅ Geradas {n_samples} amostras normais")
        return samples_final


# =============================================================================
# PARÂMETROS MELHORADOS
# =============================================================================

# Arquitetura
SEQUENCE_LENGTH = 20
FEATURE_DIM = 5
NOISE_DIM = 256  # Aumentado para mais diversidade
BATCH_SIZE = 32  # Reduzido para melhor gradientes

# Treinamento
EPOCHS = 50  # Mais épocas
D_STEPS = 2   # Menos passos do discriminador
GP_WEIGHT = 10.0  # Gradient penalty padrão
LEARNING_RATE_G = 0.0001  # Mais conservador
LEARNING_RATE_D = 0.0002  # Discriminador um pouco mais rápido

print("🔧 Parâmetros otimizados definidos")


def main():
    """Função principal melhorada"""
    
    # Configurações
    DATA_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    ANOMALY_CSV = r"D:\UTFPR\TCC\AI-City Challenge\train-anomaly-results.csv"
    MODEL_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\tensorflow_wgan_gp_improved_v2"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("🚗 SISTEMA WGAN-GP MELHORADO PARA DETECÇÃO DE ANOMALIAS")
    print("=" * 70)
    
    try:
        # Carregar labels
        from tensorflow_wgan_gp_traffic_anomaly import TrafficAnomalyLabels  # usar classe original
        anomaly_labels = None
        if os.path.exists(ANOMALY_CSV):
            anomaly_labels = TrafficAnomalyLabels(ANOMALY_CSV)
        
        # Dataset melhorado
        print("\n📂 Carregando dataset melhorado...")
        dataset = ImprovedTrafficTrajectoryDataset(
            DATA_DIR, 
            anomaly_labels, 
            max_trajectories_per_video=800,
            validation_split=0.15
        )
        
        # Construir modelos melhorados
        print("\n🏗️ Construindo arquitetura melhorada...")
        generator = build_improved_generator(NOISE_DIM, SEQUENCE_LENGTH, FEATURE_DIM)
        discriminator = build_improved_discriminator(SEQUENCE_LENGTH, FEATURE_DIM)
        
        # Modelo WGAN melhorado
        wgan = ImprovedTrafficWGAN(
            discriminator=discriminator,
            generator=generator,
            noise_dim=NOISE_DIM,
            gp_weight=GP_WEIGHT,
            d_steps=D_STEPS
        )
        
        # Otimizadores com scheduling
        d_optimizer = keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                LEARNING_RATE_D, decay_steps=1000, decay_rate=0.96
            ),
            beta_1=0.0, beta_2=0.9
        )
        
        g_optimizer = keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                LEARNING_RATE_G, decay_steps=1000, decay_rate=0.96
            ),
            beta_1=0.0, beta_2=0.9
        )
        
        wgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer)
        
        # Preparar dados
        train_dataset = dataset.get_train_dataset(BATCH_SIZE)
        val_data, val_labels = dataset.get_validation_data()
        
        # Callback melhorado
        class ImprovedModelCallback(keras.callbacks.Callback):
            def __init__(self, save_path, validation_data):
                self.save_path = save_path
                self.val_data, self.val_labels = validation_data
                self.best_score = 0
                
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:  # avaliar a cada 10 épocas
                    detector = ImprovedAnomalyDetector(self.model, dataset.scaler)
                    
                    # Testar diferentes thresholds
                    best_f1 = 0
                    for pct in [80, 85, 90, 95]:
                        try:
                            scores = detector.compute_anomaly_scores(self.val_data)
                            threshold = np.percentile(scores, pct)
                            predicted = scores > threshold
                            
                            tp = np.sum((predicted == True) & (self.val_labels == True))
                            fp = np.sum((predicted == True) & (self.val_labels == False))
                            fn = np.sum((predicted == False) & (self.val_labels == True))
                            
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            
                            best_f1 = max(best_f1, f1)
                        except:
                            continue
                    
                    if best_f1 > self.best_score:
                        self.best_score = best_f1
                        self.model.generator.save(os.path.join(self.save_path, 'best_generator.keras'))
                        self.model.discriminator.save(os.path.join(self.save_path, 'best_discriminator.keras'))
                        print(f"\n💾 Melhor modelo salvo! F1: {best_f1:.3f}")
        
        callback = ImprovedModelCallback(MODEL_DIR, (val_data, val_labels))
        
        # Treinamento
        print(f"\n🚀 Iniciando treinamento melhorado por {EPOCHS} épocas...")
        history = wgan.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=[callback],
            verbose=1
        )
        
        # Carregar melhor modelo
        wgan.generator = keras.models.load_model(os.path.join(MODEL_DIR, 'best_generator.keras'))
        wgan.discriminator = keras.models.load_model(os.path.join(MODEL_DIR, 'best_discriminator.keras'))
        
        # Detector final
        detector = ImprovedAnomalyDetector(wgan, dataset.scaler)
        
        # Avaliar com múltiplos thresholds
        print("\n🔍 Avaliação final...")
        best_results = None
        best_f1 = 0
        
        for percentile in [75, 80, 85, 90, 95]:
            scores = detector.compute_anomaly_scores(val_data)
            threshold = np.percentile(scores, percentile)
            predicted = scores > threshold
            
            tp = np.sum((predicted == True) & (val_labels == True))
            tn = np.sum((predicted == False) & (val_labels == False))
            fp = np.sum((predicted == True) & (val_labels == False))
            fn = np.sum((predicted == False) & (val_labels == True))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(val_labels)
            
            try:
                auc_roc = roc_auc_score(val_labels, scores)
            except:
                auc_roc = 0.5
            
            print(f"   {percentile}%: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}, AUC={auc_roc:.3f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_results = {
                    'precision': precision, 'recall': recall, 'f1_score': f1_score,
                    'accuracy': accuracy, 'auc_roc': auc_roc, 'threshold_percentile': percentile
                }
        
        print(f"\n🏆 MELHORES RESULTADOS:")
        print(f"   Precisão: {best_results['precision']:.3f}")
        print(f"   Recall: {best_results['recall']:.3f}")
        print(f"   F1-Score: {best_results['f1_score']:.3f}")
        print(f"   AUC-ROC: {best_results['auc_roc']:.3f}")
        
        # Gerar amostras normais demonstrativas
        print("\n🎯 Gerando amostras normais...")
        normal_samples = detector.generate_normal_samples(100)
        np.save(os.path.join(MODEL_DIR, 'generated_normal_samples.npy'), normal_samples)
        print(f"💾 Amostras normais salvas em: {MODEL_DIR}/generated_normal_samples.npy")
        
        print("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
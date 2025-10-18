import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

class TrajectoryProcessor:
    """
    Classe para processar trajetórias de veículos extraídas do YOLO
    e convertê-las em representações numéricas padronizadas para o WGAN-GP
    """
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, target_length: int = 20):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_length = target_length
        self.trajectories = {}
        self.processed_trajectories = []
        
    def load_tracking_data(self, json_file: str) -> Dict:
        """Carrega dados de tracking do arquivo JSON"""
        print("📁 Carregando dados do arquivo JSON...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Carregados dados de {len(data)} elementos")
        return data
    
    def extract_trajectories(self, tracking_data: Dict) -> Dict[int, List]:
        """
        Extrai trajetórias individuais agrupando por ID do veículo
        """
        print("🔍 Extraindo trajetórias individuais...")
        trajectories = {}
        
        # Verificar se os dados estão no formato novo (com metadata e tracks)
        if 'tracks' in tracking_data:
            # Formato novo do melancolia.py
            tracks_data = tracking_data['tracks']
            
            # Barra de progresso para extração de trajetórias
            with tqdm(total=len(tracks_data), desc="Extraindo trajetórias", unit="track") as pbar:
                for car_id, track_frames in tracks_data.items():
                    # Extrair ID numérico do car_id (ex: "car_1" -> 1)
                    vehicle_id = int(car_id.split('_')[1])
                    
                    if vehicle_id not in trajectories:
                        trajectories[vehicle_id] = []
                    
                    for frame_data in track_frames:
                        bbox = frame_data['bbox']
                        
                        # Centro da bounding box
                        x_center = bbox[0] + (bbox[2] - bbox[0]) / 2
                        y_center = bbox[1] + (bbox[3] - bbox[1]) / 2
                        
                        trajectories[vehicle_id].append({
                            'frame': frame_data['frame'],
                            'x': x_center,
                            'y': y_center,
                            'bbox_width': bbox[2] - bbox[0],
                            'bbox_height': bbox[3] - bbox[1],
                            'confidence': frame_data.get('confidence', 1.0)
                        })
                    
                    pbar.update(1)
        else:
            # Formato antigo (por frame)
            with tqdm(total=len(tracking_data), desc="Extraindo trajetórias", unit="frame") as pbar:
                for frame_num, frame_data in tracking_data.items():
                    frame_int = int(frame_num)
                    
                    if 'detections' in frame_data:
                        for detection in frame_data['detections']:
                            vehicle_id = detection['id']
                            bbox = detection['bbox']
                            
                            # Centro da bounding box
                            x_center = bbox[0] + bbox[2] / 2
                            y_center = bbox[1] + bbox[3] / 2
                            
                            if vehicle_id not in trajectories:
                                trajectories[vehicle_id] = []
                            
                            trajectories[vehicle_id].append({
                                'frame': frame_int,
                                'x': x_center,
                                'y': y_center,
                                'bbox_width': bbox[2],
                                'bbox_height': bbox[3],
                                'confidence': detection.get('confidence', 1.0)
                            })
                    
                    pbar.update(1)
        
        # Ordenar cada trajetória por frame com barra de progresso
        print("📊 Ordenando trajetórias por frame...")
        with tqdm(total=len(trajectories), desc="Ordenando trajetórias", unit="trajectory") as pbar:
            for vehicle_id in trajectories:
                trajectories[vehicle_id].sort(key=lambda x: x['frame'])
                pbar.update(1)
        
        print(f"✅ Extraídas {len(trajectories)} trajetórias únicas")
        return trajectories
    
    def calculate_dynamic_features(self, trajectory: List[Dict]) -> np.ndarray:
        """
        Calcula features dinâmicas: velocidade, aceleração e direção
        """
        if len(trajectory) < 2:
            return None
        
        features = []
        
        for i in range(len(trajectory)):
            point = trajectory[i]
            
            # Posição normalizada (0-1)
            x_norm = point['x'] / self.frame_width
            y_norm = point['y'] / self.frame_height
            
            # Velocidade
            if i == 0:
                vx, vy = 0, 0
            else:
                prev_point = trajectory[i-1]
                dt = point['frame'] - prev_point['frame']
                if dt > 0:
                    vx = (point['x'] - prev_point['x']) / dt
                    vy = (point['y'] - prev_point['y']) / dt
                else:
                    vx, vy = 0, 0
            
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            # Aceleração
            if i <= 1:
                ax, ay = 0, 0
            else:
                prev_point = trajectory[i-1]
                prev_prev_point = trajectory[i-2]
                
                dt1 = point['frame'] - prev_point['frame']
                dt2 = prev_point['frame'] - prev_prev_point['frame']
                
                if dt1 > 0 and dt2 > 0:
                    vx_prev = (prev_point['x'] - prev_prev_point['x']) / dt2
                    vy_prev = (prev_point['y'] - prev_prev_point['y']) / dt2
                    
                    ax = (vx - vx_prev) / dt1
                    ay = (vy - vy_prev) / dt1
                else:
                    ax, ay = 0, 0
            
            a_magnitude = np.sqrt(ax**2 + ay**2)
            
            # Direção (ângulo em radianos)
            if v_magnitude > 0:
                theta = np.arctan2(vy, vx)
            else:
                theta = 0
            
            features.append([x_norm, y_norm, v_magnitude, a_magnitude, theta])
        
        return np.array(features)
    
    def normalize_features(self, all_trajectories: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normaliza velocidade e aceleração globalmente
        """
        print("📏 Coletando estatísticas para normalização...")
        
        # Colete todas as velocidades e acelerações
        all_velocities = []
        all_accelerations = []
        
        with tqdm(total=len(all_trajectories), desc="Coletando estatísticas", unit="trajectory") as pbar:
            for traj in all_trajectories:
                if traj is not None:
                    all_velocities.extend(traj[:, 2])  # coluna velocidade
                    all_accelerations.extend(traj[:, 3])  # coluna aceleração
                pbar.update(1)
        
        max_velocity = np.max(all_velocities) if all_velocities else 1.0
        max_acceleration = np.max(all_accelerations) if all_accelerations else 1.0
        
        print("🔧 Normalizando features...")
        
        # Normalizar cada trajetória
        normalized_trajectories = []
        with tqdm(total=len(all_trajectories), desc="Normalizando trajetórias", unit="trajectory") as pbar:
            for traj in all_trajectories:
                if traj is not None:
                    traj_norm = traj.copy()
                    traj_norm[:, 2] = traj[:, 2] / max_velocity  # normalizar velocidade
                    traj_norm[:, 3] = traj[:, 3] / max_acceleration  # normalizar aceleração
                    # ângulo já está em [-π, π], normalizar para [-1, 1]
                    traj_norm[:, 4] = traj[:, 4] / np.pi
                    normalized_trajectories.append(traj_norm)
                else:
                    normalized_trajectories.append(None)
                pbar.update(1)
        
        print(f"✅ Normalização: max_velocity={max_velocity:.2f}, max_acceleration={max_acceleration:.2f}")
        return normalized_trajectories
    
    def interpolate_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Interpola trajetória para o tamanho fixo target_length
        """
        current_length = len(trajectory)
        
        if current_length == self.target_length:
            return trajectory
        
        if current_length < 2:
            # Trajetória muito curta, preencher com zeros
            result = np.zeros((self.target_length, 5))
            result[:current_length] = trajectory
            return result
        
        # Criar índices originais e de destino
        original_indices = np.linspace(0, current_length - 1, current_length)
        target_indices = np.linspace(0, current_length - 1, self.target_length)
        
        # Interpolar cada feature
        interpolated = np.zeros((self.target_length, 5))
        
        for feature_idx in range(5):
            if feature_idx == 4:  # ângulo - usar interpolação circular
                # Converter para coordenadas complexas para interpolação circular
                complex_angles = np.exp(1j * trajectory[:, feature_idx] * np.pi)
                
                # Interpolar partes real e imaginária separadamente
                interp_real = interp1d(original_indices, complex_angles.real, 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_imag = interp1d(original_indices, complex_angles.imag, 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                
                # Reconstruir ângulos
                real_interp = interp_real(target_indices)
                imag_interp = interp_imag(target_indices)
                complex_interp = real_interp + 1j * imag_interp
                
                interpolated[:, feature_idx] = np.angle(complex_interp) / np.pi
            else:
                # Interpolação linear normal para outras features
                interp_func = interp1d(original_indices, trajectory[:, feature_idx], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated[:, feature_idx] = interp_func(target_indices)
        
        return interpolated
    
    def filter_trajectories(self, trajectories: Dict, min_length: int = 5) -> Dict:
        """
        Filtra trajetórias muito curtas que não são úteis para análise
        """
        print("🔍 Filtrando trajetórias muito curtas...")
        
        filtered = {}
        with tqdm(total=len(trajectories), desc="Filtrando trajetórias", unit="trajectory") as pbar:
            for vehicle_id, trajectory in trajectories.items():
                if len(trajectory) >= min_length:
                    filtered[vehicle_id] = trajectory
                pbar.update(1)
        
        print(f"✅ Filtradas {len(trajectories) - len(filtered)} trajetórias curtas")
        print(f"✅ Restaram {len(filtered)} trajetórias válidas")
        return filtered
    
    def process_all_trajectories(self, json_file: str) -> np.ndarray:
        """
        Pipeline completo de processamento
        """
        print("🚀 Iniciando pipeline de processamento de trajetórias...")
        
        # 1. Carregar dados
        tracking_data = self.load_tracking_data(json_file)
        
        # 2. Extrair trajetórias por ID
        raw_trajectories = self.extract_trajectories(tracking_data)
        
        # 3. Filtrar trajetórias muito curtas
        filtered_trajectories = self.filter_trajectories(raw_trajectories)
        
        # 4. Calcular features dinâmicas
        print("⚡ Calculando features dinâmicas...")
        feature_trajectories = []
        valid_ids = []
        
        with tqdm(total=len(filtered_trajectories), desc="Calculando features", unit="trajectory") as pbar:
            for vehicle_id, trajectory in filtered_trajectories.items():
                features = self.calculate_dynamic_features(trajectory)
                if features is not None:
                    feature_trajectories.append(features)
                    valid_ids.append(vehicle_id)
                pbar.update(1)
        
        print(f"✅ Calculadas features para {len(feature_trajectories)} trajetórias")
        
        # 5. Normalizar features
        normalized_trajectories = self.normalize_features(feature_trajectories)
        
        # 6. Interpolar para tamanho fixo
        print("🔄 Interpolando trajetórias para tamanho fixo...")
        final_trajectories = []
        
        with tqdm(total=len(normalized_trajectories), desc="Interpolando trajetórias", unit="trajectory") as pbar:
            for traj in normalized_trajectories:
                if traj is not None:
                    interpolated = self.interpolate_trajectory(traj)
                    final_trajectories.append(interpolated)
                pbar.update(1)
        
        # 7. Converter para array numpy
        print("🔢 Convertendo para array numpy...")
        result = np.array(final_trajectories)
        print(f"✅ Resultado final: {result.shape} (n_trajectories, time_steps, features)")
        
        return result, valid_ids
    
    def save_trajectories(self, trajectories: np.ndarray, output_file: str):
        """Salva trajetórias processadas em arquivo numpy"""
        print("💾 Salvando trajetórias processadas...")
        np.save(output_file, trajectories)
        print(f"✅ Trajetórias salvas em: {output_file}")
        
    def visualize_sample_trajectories(self, trajectories: np.ndarray, n_samples: int = 5):
        """Visualiza algumas trajetórias de exemplo"""
        print("📊 Gerando visualizações de exemplo...")
        
        n_samples = min(n_samples, len(trajectories))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Exemplos de Trajetórias Processadas')
        
        with tqdm(total=n_samples, desc="Gerando gráficos", unit="trajectory") as pbar:
            for i in range(n_samples):
                traj = trajectories[i]
                
                # Trajetória espacial
                axes[0, 0].plot(traj[:, 0], traj[:, 1], '-o', markersize=2, alpha=0.7)
                axes[0, 0].set_title('Trajetórias Espaciais (X-Y)')
                axes[0, 0].set_xlabel('X normalizado')
                axes[0, 0].set_ylabel('Y normalizado')
                
                # Velocidade ao longo do tempo
                axes[0, 1].plot(traj[:, 2], '-o', markersize=2, alpha=0.7)
                axes[0, 1].set_title('Velocidade vs Tempo')
                axes[0, 1].set_xlabel('Frame')
                axes[0, 1].set_ylabel('Velocidade normalizada')
                
                # Aceleração
                axes[0, 2].plot(traj[:, 3], '-o', markersize=2, alpha=0.7)
                axes[0, 2].set_title('Aceleração vs Tempo')
                axes[0, 2].set_xlabel('Frame')
                axes[0, 2].set_ylabel('Aceleração normalizada')
                
                # Direção
                axes[1, 0].plot(traj[:, 4], '-o', markersize=2, alpha=0.7)
                axes[1, 0].set_title('Direção vs Tempo')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].set_ylabel('Ângulo normalizado')
                
                # Distribuição de velocidades
                axes[1, 1].hist(traj[:, 2], bins=10, alpha=0.7)
                axes[1, 1].set_title('Distribuição de Velocidades')
                axes[1, 1].set_xlabel('Velocidade')
                axes[1, 1].set_ylabel('Frequência')
                
                # Distribuição de acelerações
                axes[1, 2].hist(traj[:, 3], bins=10, alpha=0.7)
                axes[1, 2].set_title('Distribuição de Acelerações')
                axes[1, 2].set_xlabel('Aceleração')
                axes[1, 2].set_ylabel('Frequência')
                
                pbar.update(1)
        
        plt.tight_layout()
        return fig


def main(json_file_path: str, output_folder: str, output_filename: str):
    """
    Função principal para executar o processamento
    
    Args:
        json_file_path: Caminho para o arquivo JSON de tracking
        output_folder: Pasta onde salvar o arquivo processado
        output_filename: Nome do arquivo .npy (sem extensão)
    """
    
    # Configurações
    FRAME_WIDTH = 1920  # ajuste conforme seu vídeo
    FRAME_HEIGHT = 1080  # ajuste conforme seu vídeo
    TARGET_LENGTH = 20  # número de pontos por trajetória
    
    # Criar pasta de output se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    # Construir caminho completo do arquivo de saída
    if not output_filename.endswith('.npy'):
        output_filename += '.npy'
    output_path = os.path.join(output_folder, output_filename)
    
    # Inicializar processador
    processor = TrajectoryProcessor(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        target_length=TARGET_LENGTH
    )
    
    try:
        # Verificar se arquivo de entrada existe
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {json_file_path}")
        
        # Processar trajetórias
        print(f"🎯 Iniciando processamento de trajetórias...")
        print(f"📂 Arquivo de entrada: {json_file_path}")
        print(f"📁 Pasta de saída: {output_folder}")
        print(f"📄 Arquivo de saída: {output_filename}")
        print("=" * 60)
        
        trajectories, vehicle_ids = processor.process_all_trajectories(json_file_path)
        
        # Salvar resultado
        processor.save_trajectories(trajectories, output_path)
        
        # Estatísticas finais
        print("=" * 60)
        print(f"📊 ESTATÍSTICAS FINAIS:")
        print(f"✅ Total de trajetórias processadas: {len(trajectories)}")
        print(f"📐 Forma do array final: {trajectories.shape}")
        print(f"🔢 Features por ponto: 5 (x, y, velocidade, aceleração, direção)")
        print(f"⏱️  Pontos por trajetória: {TARGET_LENGTH}")
        
        # Visualizar amostras
        if len(trajectories) > 0:
            print("📊 Gerando visualizações...")
            fig = processor.visualize_sample_trajectories(trajectories)
            visualization_path = os.path.join(output_folder, f'sample_trajectories_{output_filename}.png')
            fig.savefig(visualization_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"📈 Visualizações salvas em: {visualization_path}")
        
        print("=" * 60)
        print(f"🎉 PROCESSAMENTO CONCLUÍDO! Arquivo salvo: {output_path}")
        
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Verificar se foram passados argumentos suficientes
    if len(sys.argv) != 4:
        print("Uso: python trajectoryProcessor.py <json_file_path> <output_folder> <output_filename>")
        print("Exemplo: python trajectoryProcessor.py data/1_tracking.json output trajectories_processed")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    output_folder = sys.argv[2]
    output_filename = sys.argv[3]
    
    main(json_file_path, output_folder, output_filename)
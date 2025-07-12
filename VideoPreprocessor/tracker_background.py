import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ultralytics import YOLO
import numpy as np
import os
import time
from collections import defaultdict, deque
import pandas as pd
from tqdm import tqdm  # <--- NOVO: Para barras de progresso
from sklearn.model_selection import train_test_split # <--- NOVO: Para dividir dados de treino/validação
from sklearn.metrics import f1_score, roc_curve, roc_auc_score # <--- NOVO: Para métricas
import matplotlib.pyplot as plt # <--- NOVO: Para plotar gráficos
import gc # <--- NOVO: Importa o Garbage Collector do Python
import cv2

# --- Configurações ---
CONFIG = {
    "MODEL_NAME": ["yolo11n.pt", "yolo11n-seg.pt"], # Modelos YOLOv8 são geralmente mais rápidos e precisos
    "CLASSES_TO_TRACK": [2, 7, 3],  # YOLO class IDs: 2=car, 7=truck, 3=motorcycle
    "CLASS_NAMES": {2: "car", 7: "truck", 3: "motorcycle"},
    "TRAIN_DATA_DIR": 'D:/UTFPR/TCC/AI-City Challenge/aic21-track4-train-data',
    "TEST_DATA_DIR": 'D:/UTFPR/TCC/AI-City Challenge/aic21-track4-test-data',
    "TRAIN_ANOMALY_RESULTS_FILE": 'D:/UTFPR/TCC/AI-City Challenge/train-anomaly-results.csv',
    "OUTPUT_SUBMISSION_FILE": 'track4.txt',
    "FPS": 30, # Usado como fallback, o código tentará obter o FPS real
    "MIN_STALL_FRAMES": 90,
    "MIN_STALL_MOVEMENT_PX": 5,
    "MAX_TRACK_HISTORY_LENGTH": 100,
    "FEATURE_WINDOW_SIZE": 15,
    "ANOMALY_THRESHOLD": 0.1, # ATENÇÃO: Este valor será atualizado AUTOMATICAMENTE
    "AUTOENCODER_EPOCHS": 50, # Reduzido para um treinamento mais rápido, ajuste se necessário
    "AUTOENCODER_HIDDEN_DIM": 64,
    "VALIDATION_SET_SIZE": 0.25, # Usaremos 25% dos dados de treino para validação/calibração
}

# --- Funções Auxiliares (sem alteração) ---
def calculate_speed_and_direction(history_pts, fps):
    if len(history_pts) < 2: return 0.0, 0.0, 0.0
    start_pos = history_pts[0]
    end_pos = history_pts[-1]
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    time_elapsed = len(history_pts) / fps
    if time_elapsed <= 0: return 0.0, 0.0, 0.0
    speed = np.sqrt(dx**2 + dy**2) / time_elapsed
    return speed, dx, dy

def extract_features_for_object(track_id, track_history_pts, object_states, fps):
    history_pts = track_history_pts[track_id]
    if len(history_pts) < 2: return None
    last_box = object_states[track_id]['last_box']
    if last_box is None: return None
    x1, y1, x2, y2 = last_box
    bbox_area = (x2 - x1) * (y2 - y1)
    speed, dx, dy = calculate_speed_and_direction(deque(list(history_pts)[-CONFIG["FEATURE_WINDOW_SIZE"]:]), fps)
    is_stalled = 1 if object_states[track_id]['stalled_frames'] > CONFIG["MIN_STALL_FRAMES"] else 0
    features = [speed, dx, dy, bbox_area, is_stalled]
    return np.array(features, dtype=np.float32)

def load_anomaly_ground_truth(file_path):
    anomalies_df = pd.read_csv(file_path, sep=' ', header=None, names=['video_id', 'start_time', 'end_time'])
    anomaly_map = defaultdict(list)
    for _, row in anomalies_df.iterrows():
        anomaly_map[row['video_id']].append({'start': row['start_time'], 'end': row['end_time']})
    print(f"Carregadas {len(anomalies_df)} entradas de anomalias do arquivo {file_path}.")
    return anomaly_map

# --- Modelo LSTM Autoencoder (com melhorias no treino) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        self.encoder_lstm = nn.LSTM(input_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.decoder_output_layer = nn.Linear(hidden_dim, input_feature_dim)
        print(f"LSTMAutoencoder: Inicializado (features: {input_feature_dim}, hidden: {hidden_dim}, seq_len: {seq_len}).")

    def forward(self, x):
        _, (hidden_state, _) = self.encoder_lstm(x)
        hidden_state = hidden_state.view(1, x.size(0), -1) # Concatenar hidden states bidirecionais
        decoder_input = hidden_state.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        output_decoder, _ = self.decoder_lstm(decoder_input)
        reconstructed_x = self.decoder_output_layer(output_decoder)
        return reconstructed_x

    def train_model(self, data_loader, epochs, device='cuda'):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.to(device)
        self.train()
        # <--- NOVO: Barra de progresso para as épocas
        epoch_iterator = tqdm(range(epochs), desc="Treinando Autoencoder", unit="epoch")
        for epoch in epoch_iterator:
            total_loss = 0
            for batch_features in data_loader:
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                reconstructions = self(batch_features)
                loss = criterion(reconstructions, batch_features)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            epoch_iterator.set_postfix(loss=f"{avg_loss:.6f}") # Atualiza a barra de progresso com a perda
        self.eval()

# --- Placeholder WGAN removido para focar nas solicitações atuais ---

# <--- NOVO: Função para extrair sequências de um conjunto de vídeos ---
def extract_sequences_from_videos(video_files, data_dir, yolo_model, anomaly_ground_truth, device):
    all_sequences = []
    all_labels = []
    
    # Barra de progresso externa para os VÍDEOS
    video_iterator = tqdm(video_files, desc="Extraindo Sequências (Vídeos)", unit="video", position=0)
    
    for video_file in video_iterator:
        video_id = int(video_file.split('.')[0])
        video_path = os.path.join(data_dir, video_file)
        
        track_history_pts = defaultdict(lambda: deque(maxlen=CONFIG["MAX_TRACK_HISTORY_LENGTH"]))
        object_states = {}
        
        try:
            # <--- NOVO: Usamos CV2 uma vez para obter o total de quadros para a barra de progresso
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG["FPS"]
            cap.release()
            
            results_iterator = yolo_model.track(source=video_path, classes=CONFIG["CLASSES_TO_TRACK"], device=device, persist=True, stream=True, verbose=False)
            
            # <--- NOVO: Barra de progresso interna para os QUADROS de cada vídeo
            frame_iterator = tqdm(results_iterator, total=total_frames, desc=f"Processando {video_file}", unit="frame", position=1, leave=False)

            for frame_count, result in enumerate(frame_iterator):
                current_frame_timestamp = frame_count / video_fps
                
                is_frame_anomaly_gt = any(
                    anom['start'] <= current_frame_timestamp <= anom['end']
                    for anom in anomaly_ground_truth.get(video_id, [])
                )

                if result.boxes is not None and result.boxes.id is not None:
                    # ... (O resto da lógica interna do loop permanece exatamente o mesmo) ...
                    for box, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()):
                        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        track_history_pts[track_id].append(center)

                        if track_id not in object_states:
                            object_states[track_id] = {'last_pos': center, 'stalled_frames': 0, 'last_box': box, 'feature_sequence_buffer': deque(maxlen=CONFIG["FEATURE_WINDOW_SIZE"])}
                        
                        distance_moved = np.sqrt((center[0] - object_states[track_id]['last_pos'][0])**2 + (center[1] - object_states[track_id]['last_pos'][1])**2)
                        object_states[track_id]['stalled_frames'] = object_states[track_id]['stalled_frames'] + 1 if distance_moved < CONFIG["MIN_STALL_MOVEMENT_PX"] else 0
                        object_states[track_id]['last_pos'] = center
                        object_states[track_id]['last_box'] = box

                        current_features = extract_features_for_object(track_id, track_history_pts, object_states, video_fps)
                        if current_features is not None:
                            object_states[track_id]['feature_sequence_buffer'].append(current_features)
                            if len(object_states[track_id]['feature_sequence_buffer']) == CONFIG["FEATURE_WINDOW_SIZE"]:
                                sequence = np.array(list(object_states[track_id]['feature_sequence_buffer']), dtype=np.float32)
                                all_sequences.append(sequence)
                                all_labels.append(1 if is_frame_anomaly_gt else 0)

        except Exception as e:
            print(f"Erro processando {video_file}: {e}")

    return np.array(all_sequences), np.array(all_labels)


# <--- NOVO: Função para encontrar o threshold ótimo via F1-Score ---
def find_optimal_threshold_f1(model, val_sequences, val_labels, device):
    print("\n--- Calibrando o Threshold Ótimo via F1-Score ---")
    
    # <--- MUDANÇA: Usamos um DataLoader para processar em lotes e evitar estouro de memória ---
    val_dataset = TensorDataset(torch.tensor(val_sequences, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.int32))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    model.eval()
    all_errors = []
    
    print("Calculando erros de reconstrução em lotes...")
    with torch.no_grad():
        for sequences_batch, _ in tqdm(val_loader, desc="Validando", unit="batch"):
            sequences_batch = sequences_batch.to(device)
            reconstructions = model(sequences_batch)
            errors = torch.mean((reconstructions - sequences_batch)**2, dim=[1, 2]).cpu().numpy()
            all_errors.extend(errors)

    errors = np.array(all_errors)
    best_f1 = 0
    best_threshold = 0
    
    # Testa thresholds de forma mais eficiente
    thresholds_to_test = np.linspace(np.percentile(errors, 1), np.percentile(errors, 99), 200)

    for threshold in tqdm(thresholds_to_test, desc="Buscando melhor threshold", unit="thr"):
        predictions = (errors > threshold).astype(int)
        current_f1 = f1_score(val_labels, predictions)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"Threshold ótimo encontrado: {best_threshold:.6f}")
    print(f"Melhor F1-Score no conjunto de validação: {best_f1:.4f}")
    return best_threshold, errors


# <--- NOVO: Função para plotar a Curva ROC ---
def plot_roc_curve(true_labels, errors):
    # Esta função agora recebe os 'errors' já calculados pela função anterior,
    # então não precisa de mudanças na lógica, pois não acessa mais o modelo.
    print("\n--- Gerando Gráfico da Curva ROC ---")
    fpr, tpr, thresholds = roc_curve(true_labels, errors)
    auc_score = roc_auc_score(true_labels, errors)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC para Detecção de Anomalias')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# --- Função de Detecção e Submissão (Otimizada) ---
def process_videos_for_submission(yolo_model, anomaly_model, device):
    print("\n--- Iniciando a Fase de Teste: Detectando Anomalias em Vídeos de Teste ---")
    if not os.path.exists(CONFIG["TEST_DATA_DIR"]):
        print(f"Erro: Diretório de teste não encontrado.")
        return

    test_video_files = sorted([f for f in os.listdir(CONFIG["TEST_DATA_DIR"]) if f.endswith('.mp4')])
    all_predictions = []
    
    # <--- NOVO: Barra de progresso para vídeos de teste
    test_iterator = tqdm(test_video_files, desc="Processando Vídeos de Teste", unit="video")
    
    for video_file in test_iterator:
        video_id = int(video_file.split('.')[0])
        video_path = os.path.join(CONFIG["TEST_DATA_DIR"], video_file)
        
        track_history_pts = defaultdict(lambda: deque(maxlen=CONFIG["MAX_TRACK_HISTORY_LENGTH"]))
        object_states = {}
        
        try:
            results_iterator = yolo_model.track(source=video_path, classes=CONFIG["CLASSES_TO_TRACK"], device=device, persist=True, stream=True, verbose=False)
            
            frame_count = 0
            video_fps = CONFIG["FPS"]

            for result in results_iterator:
                if frame_count == 0 and hasattr(result, 'fps'):
                     video_fps = result.fps() if result.fps() else CONFIG["FPS"]

                current_frame_timestamp = frame_count / video_fps
                
                sequences_to_predict = {}
                track_ids_in_frame = []

                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()):
                        # ... (mesma lógica de extração de features do treino) ...
                        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        track_history_pts[track_id].append(center)
                        if track_id not in object_states: object_states[track_id] = {'last_pos': center, 'stalled_frames': 0, 'last_box': box, 'feature_sequence_buffer': deque(maxlen=CONFIG["FEATURE_WINDOW_SIZE"])}
                        distance_moved = np.sqrt((center[0] - object_states[track_id]['last_pos'][0])**2 + (center[1] - object_states[track_id]['last_pos'][1])**2)
                        object_states[track_id]['stalled_frames'] = object_states[track_id]['stalled_frames'] + 1 if distance_moved < CONFIG["MIN_STALL_MOVEMENT_PX"] else 0
                        object_states[track_id]['last_pos'] = center
                        object_states[track_id]['last_box'] = box
                        current_features = extract_features_for_object(track_id, track_history_pts, object_states, video_fps)
                        
                        if current_features is not None:
                            object_states[track_id]['feature_sequence_buffer'].append(current_features)
                            if len(object_states[track_id]['feature_sequence_buffer']) == CONFIG["FEATURE_WINDOW_SIZE"]:
                                sequences_to_predict[track_id] = np.array(list(object_states[track_id]['feature_sequence_buffer']), dtype=np.float32)
                                track_ids_in_frame.append(track_id)
                
                if sequences_to_predict:
                    sequences_tensor = torch.tensor(list(sequences_to_predict.values()), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        reconstructions = anomaly_model(sequences_tensor)
                        errors = torch.mean((reconstructions - sequences_tensor)**2, dim=[1, 2]).cpu().numpy()
                    
                    for i, track_id in enumerate(track_ids_in_frame):
                        if errors[i] > CONFIG["ANOMALY_THRESHOLD"]:
                            all_predictions.append(f"{video_id} {current_frame_timestamp:.4f} {errors[i]:.4f}")
                frame_count += 1
        except Exception as e:
            print(f"Erro processando {video_file}: {e}")

    # --- Escrever Arquivo de Submissão ---
    print(f"\nEscrevendo {len(all_predictions)} predições brutas para {CONFIG['OUTPUT_SUBMISSION_FILE']}...")
    # A consolidação das anomalias (filtragem) pode ser feita aqui se necessário
    # Por simplicidade, estamos escrevendo todas as detecções acima do threshold
    try:
        with open(CONFIG['OUTPUT_SUBMISSION_FILE'], 'w') as f:
            f.write("\n".join(all_predictions))
        print(f"Arquivo de submissão '{CONFIG['OUTPUT_SUBMISSION_FILE']}' gerado com sucesso.")
    except Exception as e:
        print(f"Erro ao escrever arquivo de submissão: {e}")


# --- Função Principal do Script ---
def main():
    print("Iniciando o Sistema de Detecção de Anomalias...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    if device == 'cpu': print("AVISO CRÍTICO: CUDA não disponível. O processamento será EXTREMAMENTE lento.")

    yolo_model = YOLO(CONFIG["MODEL_NAME"][0])

    # --- 1. Divisão de Dados ---
    anomaly_ground_truth = load_anomaly_ground_truth(CONFIG["TRAIN_ANOMALY_RESULTS_FILE"])
    all_train_files = sorted([f for f in os.listdir(CONFIG["TRAIN_DATA_DIR"]) if f.endswith('.mp4')])
    
    train_files, val_files = train_test_split(
        all_train_files, 
        test_size=CONFIG["VALIDATION_SET_SIZE"], 
        random_state=42
    )
    print(f"Dados divididos: {len(train_files)} vídeos para treino, {len(val_files)} para validação.")

    # --- 2. Treinamento do Modelo (Etapa de Treino) ---
    print("\n--- FASE 1: Processando vídeos de TREINO para treinar o autoencoder ---")
    train_sequences, train_labels = extract_sequences_from_videos(train_files, CONFIG["TRAIN_DATA_DIR"], yolo_model, anomaly_ground_truth, device)
    normal_train_sequences = train_sequences[train_labels == 0]
    
    feature_dimension = normal_train_sequences.shape[2]
    anomaly_model = LSTMAutoencoder(feature_dimension, CONFIG["AUTOENCODER_HIDDEN_DIM"], CONFIG["FEATURE_WINDOW_SIZE"])
    
    if len(normal_train_sequences) > 0:
        train_loader = DataLoader(torch.tensor(normal_train_sequences, dtype=torch.float32), batch_size=256, shuffle=True)
        anomaly_model.train_model(train_loader, epochs=CONFIG["AUTOENCODER_EPOCHS"], device=device)
    else:
        print("Nenhuma sequência normal de treino coletada. Abortando.")
        return

    # <--- NOVO: Liberação explícita de memória ---
    print("\nLimpando memória dos dados de treino...")
    del train_sequences, train_labels, normal_train_sequences, train_loader
    gc.collect() # Pede ao Python para liberar a memória não utilizada
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Limpa o cache de memória da GPU

    # --- 3. Calibração do Threshold (Etapa de Validação) ---
    print("\n--- FASE 2: Processando vídeos de VALIDAÇÃO para calibrar o threshold ---")
    val_sequences, val_labels = extract_sequences_from_videos(val_files, CONFIG["TRAIN_DATA_DIR"], yolo_model, anomaly_ground_truth, device)
    if len(val_sequences) > 0:
        optimal_threshold, val_errors = find_optimal_threshold_f1(anomaly_model, val_sequences, val_labels, device)
        CONFIG["ANOMALY_THRESHOLD"] = optimal_threshold # ATUALIZA O THRESHOLD GLOBAL
        plot_roc_curve(val_labels, val_errors)
    else:
        print("Nenhuma sequência de validação coletada. Usando threshold padrão.")

    # <--- NOVO: Liberação explícita de memória ---
    print("\nLimpando memória dos dados de validação...")
    del val_sequences, val_labels, val_errors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 4. Processamento Final para Submissão ---
    process_videos_for_submission(yolo_model, anomaly_model, device)

    print("\n--- Sistema Finalizado ---")


if __name__ == "__main__":
    main()
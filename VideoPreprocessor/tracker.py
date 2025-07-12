import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from collections import defaultdict, deque
import pandas as pd

# --- Configurações ---
CONFIG = {
    "MODEL_NAME": ["yolo11n.pt", "yolo11n-seg.pt"], # Nome do modelo YOLO a ser usado (ajustado para string)
    "CLASSES_TO_TRACK": [2, 7, 3],  # YOLO class IDs: 2=car, 7=truck, 3=motorcycle
    "CLASS_NAMES": {2: "car", 7: "truck", 3: "motorcycle"},
    "TRAIN_DATA_DIR": 'D:/UTFPR/TCC/AI-City Challenge/aic21-track4-train-data',
    "TRAIN_ANOMALY_RESULTS_FILE": 'D:/UTFPR/TCC/AI-City Challenge/train-anomaly-results.csv', # CSV de anomalias (mantido .txt)
    "FPS": 30, # Assumed FPS from dataset description
    "MIN_STALL_FRAMES": 90, # 3 segundos a 30 FPS para considerar parado
    "MIN_STALL_MOVEMENT_PX": 5, # Movimento mínimo em pixels para não ser considerado parado
    "MAX_TRACK_HISTORY_LENGTH": 100, # Max length for object's center points history
    "FEATURE_WINDOW_SIZE": 15, # Número de frames para considerar na extração de features contextuais (sequência para LSTM)
    "VISUALIZE_TRAINING_PROCESS": False, # Flag para visualizar o processo de coleta de dados de treino
    "VISUALIZE_TEST_PROCESS": True, # Nova flag para visualizar o processo de teste/detecção
    "ANOMALY_THRESHOLD": 0.1, # Limiar para o erro de reconstrução do Autoencoder. Ajuste este valor!
    "AUTOENCODER_EPOCHS": 100, # Número de épocas para treinar o Autoencoder
    "AUTOENCODER_HIDDEN_DIM": 64, # Dimensão da camada oculta do LSTM Autoencoder
}

# --- Funções Auxiliares (mantidas as mesmas, exceto o nome do parâmetro para o CSV) ---

def calculate_speed_and_direction(history_pts, fps):
    """Calcula a velocidade e direção média de um objeto usando seu histórico de pontos."""
    if len(history_pts) < CONFIG["FEATURE_WINDOW_SIZE"]:
        return 0.0, 0.0, 0.0 # Velocidade, dx, dy

    recent_history = list(history_pts)[-CONFIG["FEATURE_WINDOW_SIZE"]:]
    start_pos = recent_history[0]
    end_pos = recent_history[-1]

    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    time_elapsed = (CONFIG["FEATURE_WINDOW_SIZE"] - 1) / fps
    if time_elapsed <= 0:
        return 0.0, 0.0, 0.0

    speed = np.sqrt(dx**2 + dy**2) / time_elapsed
    return speed, dx, dy

def extract_features_for_object(track_id, track_history_pts, object_states, current_frame_timestamp, fps):
    """
    Extrai um vetor de características para um objeto rastreado no frame atual.
    Estas são as features do *frame atual* que serão adicionadas à sequência.
    """
    history_pts = track_history_pts[track_id]
    
    if len(history_pts) < CONFIG["FEATURE_WINDOW_SIZE"]:
        return None 

    last_box = object_states[track_id]['last_box']
    if last_box is None:
        return None
    
    x1, y1, x2, y2 = last_box
    bbox_area = (x2 - x1) * (y2 - y1)

    speed, dx, dy = calculate_speed_and_direction(history_pts, fps)

    is_stalled = 1 if object_states[track_id]['stalled_frames'] > CONFIG["MIN_STALL_FRAMES"] else 0

    features = [
        speed,
        dx,
        dy,
        bbox_area,
        is_stalled,
        # TODO: Adicione mais features aqui para enriquecer seu modelo
    ]
    return np.array(features, dtype=np.float32)

def load_anomaly_ground_truth(file_path):
    """Carrega os dados de anomalias do arquivo train-anomaly-results.txt."""
    # O arquivo train-anomaly-results.txt usa espaço como separador. Mantido assim.
    anomalies_df = pd.read_csv(file_path, sep=' ', header=None, names=['video_id', 'start_time', 'end_time'])
    
    anomaly_map = defaultdict(list)
    for _, row in anomalies_df.iterrows():
        anomaly_map[row['video_id']].append({
            'start': row['start_time'],
            'end': row['end_time']
        })
    print(f"Carregadas {len(anomalies_df)} entradas de anomalias do arquivo {file_path}.")
    return anomaly_map

# --- Modelo LSTM Autoencoder para Detecção de Anomalias ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(input_feature_dim, hidden_dim, batch_first=True)
        
        # Decoder (recebe o hidden state do encoder e tenta reconstruir a sequência)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_output_layer = nn.Linear(hidden_dim, input_feature_dim)

        print(f"LSTMAutoencoder: Inicializado (features: {input_feature_dim}, hidden: {hidden_dim}, seq_len: {seq_len}).")

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_feature_dim)

        # Encoder
        # Passa a sequência através do encoder LSTM.
        # `output_encoder` contém a saída de cada passo de tempo do LSTM
        # `(hidden_state, cell_state)` contém o estado oculto e da célula final
        output_encoder, (hidden_state, cell_state) = self.encoder_lstm(x)
        
        # O estado oculto final do encoder é a representação comprimida da sequência.
        # Usamos hidden_state[-1] para pegar o último hidden state (assumindo 1 camada LSTM).
        
        # Decoder
        # O input para o decoder é o hidden state final do encoder, replicado para o comprimento da sequência.
        decoder_input = hidden_state[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Passa o input replicado através do decoder LSTM
        output_decoder, _ = self.decoder_lstm(decoder_input)
        
        # Aplica uma camada linear final para mapear a saída do decoder de volta para a dimensão original das features
        reconstructed_x = self.decoder_output_layer(output_decoder)
        
        return reconstructed_x

    def train_model(self, data_loader, epochs, device='cuda'):
        print(f"LSTMAutoencoder: Treinando por {epochs} épocas no dispositivo {device}...")
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss() # Mean Squared Error para o erro de reconstrução

        self.train() # Define o modelo em modo de treinamento
        for epoch in range(epochs):
            total_loss = 0
            for batch_features in data_loader:
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                reconstructions = self(batch_features)
                loss = criterion(reconstructions, batch_features) # Compara reconstrução com o original
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1: # Print a cada 10 épocas ou na última
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        self.eval() # Define o modelo em modo de avaliação após o treinamento
        print("Treinamento do Autoencoder finalizado.")

# --- Placeholder para o modelo WGAN ---
class WGAN:
    def __init__(self, feature_dim):
        print("WGAN: Placeholder inicializado. Para gerar dados anômalos, você precisará implementar o Gerador e o Crítico aqui.")
        self.feature_dim = feature_dim
        # Exemplo: self.generator = Generator(feature_dim, latent_dim)
        # Exemplo: self.critic = Critic(feature_dim)

    def train(self, normal_data_sequences, epochs=100):
        print(f"WGAN: Treinando placeholder com {len(normal_data_sequences)} sequências de dados normais por {epochs} épocas.")
        pass

    def generate_anomalous_data_sequences(self, num_sequences, seq_length):
        print(f"WGAN: Gerando {num_sequences} sequências de dados anômalos sintéticos de comprimento {seq_length}.")
        synthetic_anomalies = []
        for _ in range(num_sequences):
            normal_sequence = np.random.rand(seq_length, self.feature_dim) * 5 # Dummy normal data

            anomaly_start_frame = np.random.randint(seq_length // 4, seq_length // 2)
            anomaly_end_frame = anomaly_start_frame + np.random.randint(5, 20)

            for frame_idx in range(anomaly_start_frame, min(anomaly_end_frame, seq_length)):
                normal_sequence[frame_idx, 0] = np.random.uniform(0, 0.5)
                normal_sequence[frame_idx, 1] = np.random.uniform(-0.1, 0.1)
                normal_sequence[frame_idx, 2] = np.random.uniform(-0.1, 0.1)
                normal_sequence[frame_idx, 4] = 1 

            synthetic_anomalies.append(normal_sequence)
        
        return np.array(synthetic_anomalies)

# --- Função Principal de Preparação de Dados (sem alterações lógicas significativas) ---
def prepare_training_data(yolo_model):
    print("\n--- Iniciando a preparação dos dados de treinamento ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    anomaly_ground_truth = load_anomaly_ground_truth(CONFIG["TRAIN_ANOMALY_RESULTS_FILE"])

    train_video_files = sorted([f for f in os.listdir(CONFIG["TRAIN_DATA_DIR"]) if f.endswith('.mp4')])
    
    labeled_sequences = []

    print(f"Processando {len(train_video_files)} vídeos de treinamento para coletar features...")
    
    for i, video_file in enumerate(train_video_files):
        video_id = i + 1
        video_path = os.path.join(CONFIG["TRAIN_DATA_DIR"], video_file)
        
        print(f"Processando vídeo de treinamento {video_id}/{len(train_video_files)}: {video_file}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo {video_path}. Pulando.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Aviso: Não foi possível obter o FPS para {video_path}. Usando configurado: {CONFIG['FPS']}.")
            fps = CONFIG['FPS']

        track_history_pts = defaultdict(lambda: deque(maxlen=CONFIG["MAX_TRACK_HISTORY_LENGTH"]))
        object_states = {}

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_timestamp = frame_count / fps

            results = yolo_model.track(
                source=frame,
                classes=CONFIG["CLASSES_TO_TRACK"],
                device=device,
                persist=True,
                stream=False,
                show=False,
                verbose=False
            )

            is_frame_anomaly_gt = False
            if video_id in anomaly_ground_truth:
                for anom_interval in anomaly_ground_truth[video_id]:
                    if anom_interval['start'] <= current_frame_timestamp <= anom_interval['end']:
                        is_frame_anomaly_gt = True
                        break
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)

                        for box, track_id, class_id in zip(boxes, ids, class_ids):
                            if track_id is None:
                                continue

                            x1, y1, x2, y2 = map(int, box)
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                            track_history_pts[track_id].append(center)

                            if track_id not in object_states:
                                object_states[track_id] = {
                                    'last_pos': center,
                                    'stalled_frames': 0,
                                    'last_box': box,
                                    'class_id': class_id,
                                    'feature_sequence_buffer': deque(maxlen=CONFIG["FEATURE_WINDOW_SIZE"])
                                }
                            
                            prev_center = object_states[track_id]['last_pos']
                            distance_moved = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                            
                            if distance_moved < CONFIG["MIN_STALL_MOVEMENT_PX"]:
                                object_states[track_id]['stalled_frames'] += 1
                            else:
                                object_states[track_id]['stalled_frames'] = 0
                            
                            object_states[track_id]['last_pos'] = center
                            object_states[track_id]['last_box'] = box

                            current_object_features = extract_features_for_object(
                                track_id, track_history_pts, object_states, current_frame_timestamp, fps
                            )
                            
                            if current_object_features is not None:
                                object_states[track_id]['feature_sequence_buffer'].append(current_object_features)
                                
                                if len(object_states[track_id]['feature_sequence_buffer']) == CONFIG["FEATURE_WINDOW_SIZE"]:
                                    sequence = np.array(list(object_states[track_id]['feature_sequence_buffer']), dtype=np.float32)
                                    
                                    label = 1 if is_frame_anomaly_gt else 0
                                    
                                    labeled_sequences.append((sequence, label))

                            if CONFIG["VISUALIZE_TRAINING_PROCESS"]:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label_text = CONFIG["CLASS_NAMES"].get(class_id, "unknown")
                                cv2.putText(frame, f'{label_text} ID:{int(track_id)}', (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                pts = track_history_pts[track_id]
                                if len(pts) > 1:
                                    for j in range(1, len(pts)):
                                        cv2.line(frame, pts[j - 1], pts[j], (255, 0, 0), 2)
            
            if CONFIG["VISUALIZE_TRAINING_PROCESS"]:
                if is_frame_anomaly_gt:
                    cv2.putText(frame, "ANOMALIA (GT)", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
                cv2.putText(frame, f'Video: {video_id} | Frame: {frame_count} | Time: {current_frame_timestamp:.2f}s', (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(f'Coleta de Dados de Treinamento - Vídeo {video_id}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1

        cap.release()
        if CONFIG["VISUALIZE_TRAINING_PROCESS"]:
            cv2.destroyAllWindows()
    
    print(f"\nColeta de dados de treinamento finalizada. Total de {len(labeled_sequences)} sequências rotuladas coletadas.")
    
    normal_sequences = [seq for seq, label in labeled_sequences if label == 0]
    anomalous_sequences = [seq for seq, label in labeled_sequences if label == 1]
    
    print(f"Das sequências coletadas: {len(normal_sequences)} normais, {len(anomalous_sequences)} anômalas.")
    
    return normal_sequences, anomalous_sequences

# --- Funções para Processamento de Teste e Detecção ---

def detect_anomalies_in_frame(frame_sequences, current_frame_timestamp, anomaly_model, device='cuda'):
    """
    Usa o modelo LSTM Autoencoder para detectar anomalias com base no erro de reconstrução.
    Retorna uma lista de anomalias detectadas no formato (track_id, timestamp, confidence).
    """
    detected_anomalies_frame = []
    if not frame_sequences: # Se não houver sequências para analisar, retorne vazio
        return detected_anomalies_frame

    # Converte o dicionário de sequências para tensores PyTorch
    track_ids = list(frame_sequences.keys())
    sequences_tensor = torch.tensor(list(frame_sequences.values()), dtype=torch.float32).to(device)
    
    # Coloca o modelo em modo de avaliação
    anomaly_model.eval()
    with torch.no_grad(): # Desativa o cálculo de gradientes para inferência
        reconstructions = anomaly_model(sequences_tensor)
        
        # Calcula o erro de reconstrução (MSE) para cada sequência
        # Reduzindo a média por sequência, obtendo um score por objeto
        reconstruction_errors = torch.mean((reconstructions - sequences_tensor)**2, dim=[1, 2]).cpu().numpy()

    for i, track_id in enumerate(track_ids):
        error_score = reconstruction_errors[i]
        
        # Se o erro de reconstrução for maior que o limiar, é uma anomalia
        if error_score > CONFIG["ANOMALY_THRESHOLD"]:
            detected_anomalies_frame.append({
                'timestamp': current_frame_timestamp,
                'confidence': float(error_score), # O score de confiança é o erro de reconstrução
                'track_id': track_id
            })
            # print(f"Anomaly detected for ID {track_id} at {current_frame_timestamp:.2f}s with error {error_score:.4f}")
    
    return detected_anomalies_frame

def process_video_for_anomalies(video_path, video_id, yolo_model, anomaly_model, mode="test"):
    """Processa um único vídeo para detecção de anomalias."""
    print(f"Processando vídeo {video_path} (ID: {video_id}) em modo {mode}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Aviso: Não foi possível obter o FPS para {video_path}. Usando configurado: {CONFIG['FPS']}.")
        fps = CONFIG['FPS']
    else:
        print(f"FPS do vídeo: {fps}")

    track_history_pts = defaultdict(lambda: deque(maxlen=CONFIG["MAX_TRACK_HISTORY_LENGTH"]))
    object_states = {} # Stores last_pos, stalled_frames, last_box, e o feature_sequence_buffer
    all_detected_anomalies_raw = [] # Coleta todas as detecções antes da consolidação

    frame_count = 0
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_timestamp = frame_count / fps

        results = yolo_model.track(
            source=frame,
            classes=CONFIG["CLASSES_TO_TRACK"],
            device=device,
            persist=True,
            stream=False,
            show=False,
            verbose=False
        )

        current_frame_object_sequences = {} # Dicionário: track_id -> sequência_completa_de_features (para o Autoencoder)

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                if result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, class_id in zip(boxes, ids, class_ids):
                        if track_id is None:
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                        track_history_pts[track_id].append(center)

                        if track_id not in object_states:
                            object_states[track_id] = {
                                'last_pos': center,
                                'stalled_frames': 0,
                                'last_box': box,
                                'class_id': class_id,
                                'feature_sequence_buffer': deque(maxlen=CONFIG["FEATURE_WINDOW_SIZE"])
                            }
                        
                        prev_center = object_states[track_id]['last_pos']
                        distance_moved = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                        
                        if distance_moved < CONFIG["MIN_STALL_MOVEMENT_PX"]:
                            object_states[track_id]['stalled_frames'] += 1
                        else:
                            object_states[track_id]['stalled_frames'] = 0
                        
                        object_states[track_id]['last_pos'] = center
                        object_states[track_id]['last_box'] = box

                        current_object_features = extract_features_for_object(
                            track_id, track_history_pts, object_states, current_frame_timestamp, fps
                        )
                        
                        if current_object_features is not None:
                            object_states[track_id]['feature_sequence_buffer'].append(current_object_features)
                            
                            # Se temos uma sequência completa para este objeto, adicionamos para o Autoencoder analisar
                            if len(object_states[track_id]['feature_sequence_buffer']) == CONFIG["FEATURE_WINDOW_SIZE"]:
                                current_frame_object_sequences[track_id] = np.array(list(object_states[track_id]['feature_sequence_buffer']), dtype=np.float32)

                        # --- Visualização durante o Teste ---
                        if CONFIG["VISUALIZE_TEST_PROCESS"]:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_text = CONFIG["CLASS_NAMES"].get(class_id, "unknown")
                            cv2.putText(frame, f'{label_text} ID:{int(track_id)}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            pts = track_history_pts[track_id]
                            if len(pts) > 1:
                                for j in range(1, len(pts)):
                                    cv2.line(frame, pts[j - 1], pts[j], (255, 0, 0), 2)
        
        # Detecção de Anomalias Usando o Autoencoder
        if current_frame_object_sequences:
            anomalies_in_frame = detect_anomalies_in_frame(current_frame_object_sequences, current_frame_timestamp, anomaly_model, device)
            all_detected_anomalies_raw.extend(anomalies_in_frame)

            if CONFIG["VISUALIZE_TEST_PROCESS"]:
                for anom in anomalies_in_frame:
                    # Encontrar a bounding box do objeto anômalo para desenhar um aviso
                    # Isso é uma simplificação; idealmente você teria a posição exata
                    # do objeto anômalo no frame atual a partir do `current_frame_object_sequences`
                    # e dos resultados YOLO do frame.
                    for result in results: # Re-iterar os resultados do YOLO no frame atual
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            if result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
                                for box, obj_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy() if result.boxes.id is not None else []):
                                    if obj_id == anom['track_id']:
                                        x1, y1, x2, y2 = map(int, box)
                                        cv2.putText(frame, "ANOMALY!", (x1, y1 - 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                        break


        if CONFIG["VISUALIZE_TEST_PROCESS"]:
            cv2.putText(frame, f'Video: {video_id} | Frame: {frame_count} | Time: {current_frame_timestamp:.2f}s', (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(f'Detecção de Anomalias - Vídeo {video_id}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1

    end_time = time.time()
    print(f"Finalizado processamento do vídeo {video_id}. Total de frames: {frame_count}, Tempo: {end_time - start_time:.2f}s")
    cap.release()
    if CONFIG["VISUALIZE_TEST_PROCESS"]:
        cv2.destroyAllWindows()

    return consolidate_anomalies(all_detected_anomalies_raw, video_id) # Não precisa de FPS aqui

def consolidate_anomalies(raw_anomalies, video_id):
    """
    Consolida as detecções brutas de anomalias no formato final de submissão.
    Filtra e agrupa detecções próximas no tempo para um mesmo objeto.
    """
    if not raw_anomalies:
        return []

    # Sort por timestamp para processamento sequencial
    raw_anomalies.sort(key=lambda x: x['timestamp'])

    consolidated_predictions = []
    
    # Mapeia track_id para o último tempo reportado para evitar duplicação em curto período
    last_reported_time_for_track = defaultdict(lambda: -float('inf'))
    
    # Intervalo de tempo para considerar a mesma anomalia (em segundos)
    # Ex: se uma anomalia for detectada 2x em 2 segundos para o mesmo objeto, conte como 1
    COALESCE_TIME_WINDOW = 2.0 

    for anomaly in raw_anomalies:
        track_id = anomaly['track_id']
        timestamp = anomaly['timestamp']
        confidence = anomaly['confidence']

        # Verifica se esta anomalia para este track_id foi reportada recentemente
        if (timestamp - last_reported_time_for_track[track_id]) > COALESCE_TIME_WINDOW:
            consolidated_predictions.append({
                'video_id': video_id,
                'timestamp': timestamp,
                'confidence': confidence,
                'track_id': track_id # Mantém o track_id para depuração, não para submissão
            })
            last_reported_time_for_track[track_id] = timestamp
        else:
            # Se for uma detecção próxima, atualiza a confiança se for maior
            # Encontre a última anomalia consolidada para este track_id
            for i in range(len(consolidated_predictions) - 1, -1, -1):
                if consolidated_predictions[i]['track_id'] == track_id:
                    if confidence > consolidated_predictions[i]['confidence']:
                        consolidated_predictions[i]['confidence'] = confidence
                        # O timestamp poderia ser atualizado para o mais recente ou o do pico de confiança
                        consolidated_predictions[i]['timestamp'] = timestamp 
                    break

    # Após consolidar, filtre os 100 com maior confiança e remova track_id
    final_submission_anomalies = []
    for anom in consolidated_predictions:
        final_submission_anomalies.append({
            'video_id': anom['video_id'],
            'timestamp': anom['timestamp'],
            'confidence': anom['confidence']
        })

    # Classifica por confiança e pega os top 100
    final_submission_anomalies.sort(key=lambda x: x['confidence'], reverse=True)
    return final_submission_anomalies[:100]


# --- Função Principal do Script ---
def main():
    print("Iniciando o Sistema de Detecção de Anomalias...")
    print(f"Modelo YOLO: {CONFIG['MODEL_NAME'][0]}")
    print(f"Visualização do treinamento: {'Ativada' if CONFIG['VISUALIZE_TRAINING_PROCESS'] else 'Desativada'}")
    print(f"Visualização da detecção: {'Ativada' if CONFIG['VISUALIZE_TEST_PROCESS'] else 'Desativada'}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    if device == 'cpu':
        print("Aviso: CUDA não disponível. O processamento será muito lento no CPU.")

    try:
        yolo_model = YOLO(CONFIG["MODEL_NAME"][0])
        print(f"Modelo YOLO '{CONFIG['MODEL_NAME'][0]}' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}. Verifique o caminho e a instalação.")
        return

    # --- 1. Preparação dos Dados de Treinamento e Coleta de Features ---
    normal_sequences, anomalous_sequences = prepare_training_data(yolo_model)

    # --- 2. Treinamento do LSTM Autoencoder ---
    feature_dimension = 5 # speed, dx, dy, bbox_area, is_stalled
    seq_len = CONFIG["FEATURE_WINDOW_SIZE"]
    hidden_dim = CONFIG["AUTOENCODER_HIDDEN_DIM"]

    anomaly_model = LSTMAutoencoder(feature_dimension, hidden_dim, seq_len).to(device)

    if normal_sequences:
        # Converte as sequências normais para um TensorDataset e DataLoader
        normal_data_tensor = torch.tensor(np.array(normal_sequences), dtype=torch.float32)
        train_loader = DataLoader(normal_data_tensor, batch_size=64, shuffle=True) # Batch size ajustável
        
        print("\n--- Iniciando o treinamento do LSTM Autoencoder com dados normais ---")
        anomaly_model.train_model(train_loader, epochs=CONFIG["AUTOENCODER_EPOCHS"], device=device)
    else:
        print("Nenhuma sequência normal coletada. O LSTM Autoencoder NÃO será treinado.")
        print("Certifique-se de que seus vídeos de treinamento estão acessíveis e o rastreamento está funcionando.")
        return # Não pode continuar sem o modelo treinado

    # --- 3. Integração com WGAN (Placeholder) ---
    wgan = WGAN(feature_dimension)
    if normal_sequences: # WGAN só treina se houver dados normais
        wgan.train(normal_data_tensor, epochs=50) # WGAN treina com tensores PyTorch
        # Geração de dados anômalos sintéticos (para futuro uso no treinamento/avaliação de classificadores)
        num_synthetic_anomalies = 5000
        synthetic_anomalies_data = wgan.generate_anomalous_data_sequences(num_synthetic_anomalies, seq_len)
        print(f"Geradas {len(synthetic_anomalies_data)} sequências anômalas sintéticas (Placeholder).")
    else:
        print("Pulando treinamento do WGAN e geração de dados sintéticos.")

    # --- 4. Fase de Teste e Detecção de Anomalias ---
    print("\n--- Iniciando a Fase de Teste: Detectando Anomalias em Vídeos de Teste ---")
    test_data_dir = 'D:/UTFPR/TCC/AI-City Challenge/aic21-track4-test-data' # Adicione ao CONFIG ou defina aqui
    if not os.path.exists(test_data_dir):
        print(f"Erro: Diretório de dados de teste não encontrado em {test_data_dir}. Não é possível prosseguir com o teste.")
        return

    test_video_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.mp4')])
    
    all_predictions = []
    for i, video_file in enumerate(test_video_files):
        video_id = i + 1
        video_path = os.path.join(test_data_dir, video_file)
        
        # O modelo de anomalias (Autoencoder treinado) é passado para a função
        video_predictions = process_video_for_anomalies(video_path, video_id, yolo_model, anomaly_model, mode="test")
        all_predictions.extend(video_predictions)

    # --- 5. Escrever Arquivo de Submissão ---
    output_submission_file = 'track4.txt' # Adicione ao CONFIG ou defina aqui
    print(f"\nEscrevendo previsões para {output_submission_file}...")
    try:
        with open(output_submission_file, 'w') as f:
            for pred in all_predictions:
                f.write(f"{pred['video_id']} {pred['timestamp']:.4f} {pred['confidence']:.4f}\n")
        print(f"Arquivo de submissão '{output_submission_file}' gerado com sucesso. Total de anomalias reportadas: {len(all_predictions)}.")
    except Exception as e:
        print(f"Erro ao escrever arquivo de submissão: {e}")

    print("\n--- Sistema Finalizado ---")

if __name__ == "__main__":
    main()
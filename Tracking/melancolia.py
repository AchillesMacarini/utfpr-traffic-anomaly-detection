import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
from .trajectoryManager import TrajectoryManager

def run_tracking(file_path, json_out_path, csv_out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = file_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    model = YOLO('yolov8m.pt')
    model.to(device)
    car_class_ids = [2, 3, 5, 7]

    trajectories = {}
    missing_frames = {}

    frame_idx = 0
    for _ in tqdm(range(total_frames), desc=f"Processing {file_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=car_class_ids,
            conf=0.10,  # Diminua para pegar mais detecções
            verbose=False,
            device=device,
            imgsz=960,
            tracker='bytetrack.yaml'
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            track_ids = results[0].boxes.id.int().detach().cpu().numpy()
            confs = results[0].boxes.conf.detach().cpu().numpy()
            classes = results[0].boxes.cls.int().detach().cpu().numpy()

            detected_car_ids = set()
            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                if cls in car_class_ids and conf > 0.4:
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    car_id = f"car_{track_id}"

                    if car_id not in trajectories:
                        trajectories[car_id] = []
                    # Salva a posição com timestamp em segundos
                    timestamp = frame_idx / fps
                    trajectories[car_id].append((timestamp, cx, cy))
                    detected_car_ids.add(car_id)

            for car_id in trajectories.keys():
                if car_id not in detected_car_ids:
                    missing_frames[car_id] = missing_frames.get(car_id, 0) + 1
                else:
                    missing_frames[car_id] = 0

        frame_idx += 1

    cap.release()

    # Salvar as trajetórias em JSON e CSV
    tm = TrajectoryManager(json_out_path)
    for car_id, traj in trajectories.items():
        tm.add_trajectory(car_id, traj)
    tm.save_trajectories()
    tm.save_trajectories_csv(int(fps), csv_out_path)
    print(f"Trajetórias salvas em {csv_out_path}.")

# To run directly if this file is executed
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Uso: python melancolia.py <video_path> <json_out_path> <csv_out_path>")
    else:
        run_tracking(sys.argv[1], sys.argv[2], sys.argv[3])

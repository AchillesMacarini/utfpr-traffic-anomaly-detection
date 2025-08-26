import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import sys

from Tracking.trajectoryManager import TrajectoryManager  # Adicione este import

def run_tracking(file_path):
    print("cv2 importado")
    print("torch importado")
    print("tqdm importado")
    print("ultralytics importado")
    print("Path exists")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"File path: {file_path}")

    video_path = file_path
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    class_names = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    model = YOLO('yolov8m.pt')
    model.to(device)
    car_class_ids = [2, 3, 5, 7]

    trajectories = {}
    missing_frames = {}

    frame_idx = 0
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = torch.from_numpy(rgb_frame).to(device).float()
            tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0)

            results = model.track(
                frame,
                persist=True,
                classes=car_class_ids,
                conf=0.25,
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
                        trajectories[car_id].append((cx, cy))
                        detected_car_ids.add(car_id)

                        label = f"{class_names.get(cls, 'Desconhecido')}_{track_id}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                for car_id in trajectories.keys():
                    if car_id not in detected_car_ids:
                        missing_frames[car_id] = missing_frames.get(car_id, 0) + 1
                    else:
                        missing_frames[car_id] = 0

                car_ids_to_remove = [car_id for car_id, count in missing_frames.items() if count > 30]
                for car_id in car_ids_to_remove:
                    del trajectories[car_id]
                    del missing_frames[car_id]

            for car_id, traj in trajectories.items():
                for i in range(1, len(traj)):
                    cv2.line(frame, traj[i-1], traj[i], (255,0,0), 2)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time tracking finished.")

    # Salvar as trajetórias em CSV
    tm = TrajectoryManager("trajectories.json")
    for car_id, traj in trajectories.items():
        tm.add_trajectory(car_id, traj)
    tm.save_trajectories_csv(fps, "trajectories.csv")
    print("Trajetórias salvas em trajectories.csv.")

# To run directly if this file is executed
if __name__ == "__main__":
    run_tracking()

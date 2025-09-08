import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import os
from google.colab import drive
import sys

drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/ai-city-challenge"
tracking_path = os.path.join(drive_path, "tracking") # Define tracking_path

if os.path.exists(drive_path):
  print("Path exists")
  # Add the directory to the Python path
  sys.path.append(tracking_path) # Use tracking_path here
  print(f"Added {tracking_path} to sys.path") # Print the path being added

  # Changed import to directly import the module
  try:
      import trajectoryManager # Attempt to import the module directly
      print("Successfully imported trajectoryManager") # Indicate successful import
  except ModuleNotFoundError:
      print(f"Error: Could not import 'trajectoryManager'. Make sure 'trajectoryManager.py' exists in {tracking_path}")
      sys.exit(1) # Exit if import fails

import json  # Add this import

def run_tracking(file_path, json_out_path, csv_out_path, show_stream=False):
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
    car_frame_data = {}  # {car_id: [ {frame_idx, timestamp, x, y, speed}, ... ] }

    frame_idx = 0
    prev_positions = {}

    # --- Add trackbar for video seeking in seconds ---
    if show_stream:
        window_name = "Tracking Stream"
        cv2.namedWindow(window_name)
        total_seconds = int(total_frames / fps)
        seek_request = [False]  # Use a mutable type to allow modification in callback

        def on_trackbar(val):
            seek_request[0] = True

        cv2.createTrackbar('Time (s)', window_name, 0, total_seconds, on_trackbar)

    # --- Progress bar setup ---
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    while frame_idx < total_frames:
        # Only seek if requested by the user
        if show_stream and seek_request[0]:
            pos = cv2.getTrackbarPos('Time (s)', window_name)
            frame_idx = int(pos * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            seek_request[0] = False

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=car_class_ids,
            conf=0.10,
            verbose=False,
            device=device,
            imgsz=960,
            tracker='botsort.yaml'  # Use BOTSort (supported by Ultralytics)
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
                    timestamp = frame_idx / fps
                    trajectories[car_id].append((timestamp, cx, cy))
                    detected_car_ids.add(car_id)

                    # Calculate speed in px/s
                    speed = 0.0
                    if car_id in prev_positions:
                        prev_timestamp, prev_cx, prev_cy = prev_positions[car_id]
                        dt = timestamp - prev_timestamp
                        if dt > 0:
                            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                            speed = dist / dt
                    prev_positions[car_id] = (timestamp, cx, cy)

                    # --- Collect per-frame data for JSON ---
                    if car_id not in car_frame_data:
                        car_frame_data[car_id] = []
                    car_frame_data[car_id].append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "x": cx,
                        "y": cy,
                        "speed": speed
                    })

                    # Draw bounding box and label if streaming
                    if show_stream:
                        label = f"{class_names.get(cls, 'Vehicle')} {track_id}"
                        coords = f"X:{cx} Y:{cy}"
                        speed_str = f"{speed:.1f}px/s"
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        # Draw label, coordinates, and speed
                        cv2.putText(frame, label, (int(x1), int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, coords, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, speed_str, (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for car_id in trajectories.keys():
                if car_id not in detected_car_ids:
                    missing_frames[car_id] = missing_frames.get(car_id, 0) + 1
                else:
                    missing_frames[car_id] = 0

        if show_stream:
            cv2.imshow(window_name, frame)
            # Update trackbar position in seconds
            cv2.setTrackbarPos('Time (s)', window_name, int(frame_idx / fps))
            key = cv2.waitKey(1)
            # ESC or Q to quit, P to pause
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("Streaming interrupted by user.")
                break
            elif key == ord('p') or key == ord('P'):
                print("Paused. Press P to resume.")
                while True:
                    key2 = cv2.waitKey(0)
                    pos = cv2.getTrackbarPos('Time (s)', window_name)
                    if int(pos * fps) != frame_idx:
                        frame_idx = int(pos * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        break
                    if key2 == ord('p') or key2 == ord('P'):
                        print("Resumed.")
                        break
                    elif key2 == 27 or key2 == ord('q') or key2 == ord('Q'):
                        print("Streaming interrupted by user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        frame_idx += 1
        pbar.update(1)  # Update progress bar

    pbar.close()  # Close progress bar

    cap.release()
    if show_stream:
        cv2.destroyAllWindows()

    # --- Save car_frame_data to JSON ---
    with open(json_out_path, "w") as f:
        json.dump(car_frame_data, f, indent=2)

# To run directly if this file is executed
if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [4, 5]:
        print("Uso: python melancolia.py <video_path> <json_out_path> <csv_out_path> [stream]")
    else:
        show_stream = len(sys.argv) == 5 and sys.argv[4].lower() == "stream"
        run_tracking(sys.argv[1], sys.argv[2], sys.argv[3], show_stream)

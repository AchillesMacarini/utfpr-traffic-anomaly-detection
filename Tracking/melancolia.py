import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os

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
    # Use YOLOv8x for detection only (no built-in tracking)
    model = YOLO('yolov8x.pt')
    model.to(device)
    car_class_ids = [2, 3, 5, 7]

    # Custom tracker class
    class CustomTracker:
        def __init__(self, max_disappeared=30, max_distance=150):
            self.next_id = 1
            self.tracks = {}  # {track_id: track_info}
            self.max_disappeared = max_disappeared
            self.max_distance = max_distance
            
        def calculate_iou(self, box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        def calculate_cost_matrix(self, detections):
            if not self.tracks:
                return np.array([]), [], []
                
            # Get active tracks (not disappeared for too long)
            active_tracks = {tid: track for tid, track in self.tracks.items() 
                           if track['disappeared'] < self.max_disappeared}
            
            if not active_tracks:
                return np.array([]), [], []
            
            track_ids = list(active_tracks.keys())
            
            # Create cost matrix based on multiple factors
            cost_matrix = np.full((len(detections), len(track_ids)), 1000.0)
            
            for i, detection in enumerate(detections):
                det_center = detection['center']
                det_bbox = detection['bbox']
                det_class = detection['class_id']
                
                for j, track_id in enumerate(track_ids):
                    track = active_tracks[track_id]
                    track_center = track['center']
                    track_bbox = track['bbox']
                    track_class = track['class_id']
                    
                    # Skip if different vehicle classes
                    if det_class != track_class:
                        continue
                    
                    # Distance cost
                    distance = np.sqrt((det_center[0] - track_center[0])**2 + 
                                     (det_center[1] - track_center[1])**2)
                    
                    # IoU similarity
                    iou = self.calculate_iou(det_bbox, track_bbox)
                    
                    # Size similarity
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    size_ratio = min(det_area, track_area) / max(det_area, track_area)
                    
                    # Combined cost (lower is better)
                    if distance < self.max_distance and iou > 0.1:
                        cost = distance * 0.5 + (1 - iou) * 100 + (1 - size_ratio) * 50
                        cost_matrix[i, j] = cost
            
            return cost_matrix, track_ids, active_tracks
        
        def update(self, detections, frame_idx):
            if not detections:
                # Mark all tracks as disappeared
                for track in self.tracks.values():
                    track['disappeared'] += 1
                return {}
            
            # Calculate cost matrix and perform assignment
            cost_matrix, track_ids, active_tracks = self.calculate_cost_matrix(detections)
            
            assignments = {}
            used_detection_indices = set()
            used_track_indices = set()
            
            if cost_matrix.size > 0:
                # Use Hungarian algorithm for optimal assignment
                det_indices, track_indices = linear_sum_assignment(cost_matrix)
                
                for det_idx, track_idx in zip(det_indices, track_indices):
                    if cost_matrix[det_idx, track_idx] < 500:  # Reasonable threshold
                        track_id = track_ids[track_idx]
                        assignments[det_idx] = track_id
                        used_detection_indices.add(det_idx)
                        used_track_indices.add(track_idx)
                        
                        # Update track
                        detection = detections[det_idx]
                        self.tracks[track_id].update({
                            'center': detection['center'],
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'class_id': detection['class_id'],
                            'disappeared': 0,
                            'last_seen': frame_idx
                        })
            
            # Mark unmatched tracks as disappeared
            for j, track_id in enumerate(track_ids):
                if j not in used_track_indices:
                    self.tracks[track_id]['disappeared'] += 1
            
            # Create new tracks for unmatched detections
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    # Only create new track if detection has good confidence
                    if detection['confidence'] > 0.5:
                        new_track_id = self.next_id
                        self.next_id += 1
                        
                        self.tracks[new_track_id] = {
                            'center': detection['center'],
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'class_id': detection['class_id'],
                            'disappeared': 0,
                            'last_seen': frame_idx,
                            'created': frame_idx
                        }
                        assignments[i] = new_track_id
            
            # Remove tracks that have been gone too long
            tracks_to_remove = [tid for tid, track in self.tracks.items() 
                              if track['disappeared'] >= self.max_disappeared]
            for tid in tracks_to_remove:
                del self.tracks[tid]
            
            return assignments

    # Initialize custom tracker
    tracker = CustomTracker(max_disappeared=60, max_distance=200)
    
    # Enhanced tracking data structures
    trajectories = {}
    car_frame_data = defaultdict(list)
    prev_positions = {}
    track_confidence_history = defaultdict(list)
    min_track_length = 5

    frame_idx = 0

    # --- Add trackbar for video seeking in seconds ---
    if show_stream:
        window_name = "Enhanced Tracking Stream"
        cv2.namedWindow(window_name)
        total_seconds = int(total_frames / fps)
        seek_request = [False]

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

        # Run detection only (no tracking)
        results = model(
            frame,
            classes=car_class_ids,
            conf=0.3,
            iou=0.7,
            verbose=False,
            device=device,
            imgsz=1280,
            half=False
        )

        detections = []
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            confs = results[0].boxes.conf.detach().cpu().numpy()
            classes = results[0].boxes.cls.int().detach().cpu().numpy()

            # Prepare detections for custom tracker
            for box, conf, cls in zip(boxes, confs, classes):
                if cls in car_class_ids and conf > 0.4:
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [cx, cy],
                        'confidence': float(conf),
                        'class_id': int(cls)
                    })

            # Update tracker with current detections
            assignments = tracker.update(detections, frame_idx)
            
            # Process assigned detections
            for det_idx, track_id in assignments.items():
                detection = detections[det_idx]
                x1, y1, x2, y2 = detection['bbox']
                cx, cy = detection['center']
                conf = detection['confidence']
                cls = detection['class_id']
                
                bottom_center = (cx, int(y2))
                car_id = f"car_{track_id}"
                
                # Initialize trajectory if new
                if car_id not in trajectories:
                    trajectories[car_id] = []
                
                timestamp = frame_idx / fps
                trajectories[car_id].append((timestamp, cx, cy))
                
                # Track confidence history
                track_confidence_history[car_id].append(conf)
                avg_confidence = np.mean(track_confidence_history[car_id][-10:])
                
                # Calculate speed
                speed = 0.0
                if car_id in prev_positions:
                    prev_frame, prev_cx, prev_cy = prev_positions[car_id]
                    frame_diff = frame_idx - prev_frame
                    if frame_diff > 0:
                        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                        speed = dist / frame_diff
                
                prev_positions[car_id] = (frame_idx, cx, cy)

                # Frame data for JSON
                frame_data = {
                    "track_id": track_id,
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [cx, cy],
                    "bottom_center": list(bottom_center),
                    "speed": float(speed),
                    "confidence": float(conf),
                    "avg_confidence": float(avg_confidence),
                    "class": class_names.get(cls, 'Vehicle'),
                    "class_id": int(cls)
                }
                
                car_frame_data[car_id].append(frame_data)

                # Enhanced visualization
                if show_stream:
                    # Color based on track age and confidence
                    track_age = frame_idx - tracker.tracks[track_id]['created']
                    if avg_confidence > 0.7 and track_age > 10:
                        color = (0, 255, 0)  # Green for stable, high confidence
                    elif avg_confidence > 0.5:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Labels
                    label = f"{class_names.get(cls, 'Vehicle')} ID:{track_id}"
                    conf_text = f"Conf:{conf:.2f} Avg:{avg_confidence:.2f}"
                    speed_text = f"Speed:{speed:.1f}px/f Age:{track_age}"
                    
                    # Text background
                    cv2.rectangle(frame, (int(x1), int(y1)-75), (int(x1)+320, int(y1)), (0,0,0), -1)
                    cv2.putText(frame, label, (int(x1), int(y1)-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, conf_text, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, speed_text, (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw center and bottom center
                    cv2.circle(frame, (cx, cy), 3, color, -1)
                    cv2.circle(frame, bottom_center, 3, (255, 0, 255), -1)

        if show_stream:
            # Add info overlay
            active_tracks = len([t for t in tracker.tracks.values() if t['disappeared'] < 30])
            info_text = f"Frame: {frame_idx}/{total_frames} | Active tracks: {active_tracks} | Total tracks: {len(tracker.tracks)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            cv2.setTrackbarPos('Time (s)', window_name, int(frame_idx / fps))
            
            key = cv2.waitKey(1)
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
        pbar.update(1)

    pbar.close()
    cap.release()
    if show_stream:
        cv2.destroyAllWindows()

    # Filter out short tracks
    filtered_car_frame_data = {}
    for car_id, frames in car_frame_data.items():
        if len(frames) >= min_track_length:
            filtered_car_frame_data[car_id] = frames
        else:
            print(f"Filtered out short track {car_id} with only {len(frames)} frames")

    # Save enhanced data to JSON
    output_data = {
        "metadata": {
            "total_frames": total_frames,
            "fps": fps,
            "video_path": video_path,
            "total_tracks": len(filtered_car_frame_data),
            "min_track_length": min_track_length
        },
        "tracks": filtered_car_frame_data
    }
    
    # Before the JSON writing section, add this check:
    if os.path.isdir(json_out_path):
        # If it's a directory, create a filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_out_path = os.path.join(json_out_path, f"{video_name}_tracking.json")

    with open(json_out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Tracking completed. Found {len(filtered_car_frame_data)} valid tracks.")
    print(f"Results saved to: {json_out_path}")

# To run directly if this file is executed
if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [4, 5]:
        print("Uso: python melancolia.py <video_path> <json_out_path> <csv_out_path> [stream]")
    else:
        show_stream = len(sys.argv) == 5 and sys.argv[4].lower() == "stream"
        run_tracking(sys.argv[1], sys.argv[2], sys.argv[3], show_stream)

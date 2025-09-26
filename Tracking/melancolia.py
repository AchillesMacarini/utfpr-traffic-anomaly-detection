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
    model = YOLO('yolov8x.pt')
    model.to(device)
    car_class_ids = [2, 3, 5, 7]

    # Enhanced Custom tracker class
    class EnhancedTracker:
        def __init__(self, max_disappeared=90, max_distance=250, min_confidence=0.3):
            self.next_id = 1
            self.tracks = {}  # {track_id: track_info}
            self.inactive_tracks = {}  # Tracks that disappeared but might return
            self.retired_tracks = set()  # IDs that will never be reused
            self.max_disappeared = max_disappeared
            self.max_distance = max_distance
            self.min_confidence = min_confidence
            self.kalman_filters = {}
            self.track_history_buffer = defaultdict(list)  # Store recent positions for each track
            self.appearance_features = {}  # Store appearance features for re-identification
            
        def init_kalman_filter(self, track_id, initial_pos):
            """Initialize Kalman filter for motion prediction"""
            kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            kf.measurementNoiseCov = 5.0 * np.eye(2, dtype=np.float32)
            kf.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
            kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
            self.kalman_filters[track_id] = kf
            
        def predict_position(self, track_id):
            """Predict next position using Kalman filter"""
            if track_id in self.kalman_filters:
                prediction = self.kalman_filters[track_id].predict()
                return [int(prediction[0]), int(prediction[1])]
            return None
            
        def update_kalman(self, track_id, measurement):
            """Update Kalman filter with new measurement"""
            if track_id in self.kalman_filters:
                measurement_np = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
                self.kalman_filters[track_id].correct(measurement_np)
                
        def extract_appearance_features(self, frame, bbox):
            """Extract simple appearance features from vehicle region"""
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            if roi.size == 0:
                return np.zeros(10)
            
            # Extract color histogram and size features
            hist_b = cv2.calcHist([roi], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [8], [0, 256])
            hist_r = cv2.calcHist([roi], [2], None, [8], [0, 256])
            
            # Normalize histograms
            features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            features = features / (np.sum(features) + 1e-6)
            
            # Add size feature
            area = (x2 - x1) * (y2 - y1)
            aspect_ratio = (x2 - x1) / max(1, y2 - y1)
            
            return np.concatenate([features[:10], [area / 10000.0, aspect_ratio]])
            
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
            
        def calculate_motion_consistency(self, track_id, new_center):
            """Calculate motion consistency score"""
            if track_id not in self.track_history_buffer or len(self.track_history_buffer[track_id]) < 3:
                return 1.0
            
            recent_positions = self.track_history_buffer[track_id][-5:]
            
            # Calculate average velocity from recent positions
            velocities = []
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                velocities.append([dx, dy])
            
            if not velocities:
                return 1.0
            
            avg_velocity = np.mean(velocities, axis=0)
            expected_pos = [recent_positions[-1][0] + avg_velocity[0], 
                           recent_positions[-1][1] + avg_velocity[1]]
            
            distance_from_expected = np.sqrt((new_center[0] - expected_pos[0])**2 + 
                                           (new_center[1] - expected_pos[1])**2)
            
            # Normalize to 0-1 score (lower distance = higher score)
            motion_score = 1.0 / (1.0 + distance_from_expected / 50.0)
            return motion_score
            
        def calculate_enhanced_cost_matrix(self, detections, frame):
            """Enhanced cost calculation with multiple factors"""
            all_tracks = {**self.tracks, **self.inactive_tracks}
            
            if not all_tracks:
                return np.array([]), [], {}
            
            # Separate active and inactive tracks for different treatment
            active_tracks = {tid: track for tid, track in self.tracks.items()}
            inactive_tracks = {tid: track for tid, track in self.inactive_tracks.items() 
                             if track['disappeared'] < self.max_disappeared * 2}
            
            all_track_dict = {**active_tracks, **inactive_tracks}
            track_ids = list(all_track_dict.keys())
            
            if not track_ids:
                return np.array([]), [], {}
            
            cost_matrix = np.full((len(detections), len(track_ids)), 10000.0)
            
            for i, detection in enumerate(detections):
                det_center = detection['center']
                det_bbox = detection['bbox']
                det_class = detection['class_id']
                det_confidence = detection['confidence']
                det_features = self.extract_appearance_features(frame, det_bbox)
                
                for j, track_id in enumerate(track_ids):
                    track = all_track_dict[track_id]
                    track_class = track['class_id']
                    
                    # Skip if different vehicle classes
                    if det_class != track_class:
                        continue
                    
                    # Use predicted position for active tracks
                    if track_id in active_tracks:
                        predicted_pos = self.predict_position(track_id)
                        if predicted_pos:
                            track_center = predicted_pos
                        else:
                            track_center = track['center']
                    else:
                        track_center = track['center']
                    
                    track_bbox = track['bbox']
                    
                    # Distance cost with prediction
                    distance = np.sqrt((det_center[0] - track_center[0])**2 + 
                                     (det_center[1] - track_center[1])**2)
                    
                    # Skip if too far away
                    max_dist = self.max_distance * (2 if track_id in inactive_tracks else 1)
                    if distance > max_dist:
                        continue
                    
                    # IoU similarity
                    iou = self.calculate_iou(det_bbox, track_bbox)
                    
                    # Size similarity
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    size_ratio = min(det_area, track_area) / max(det_area, track_area)
                    
                    # Motion consistency
                    motion_score = self.calculate_motion_consistency(track_id, det_center)
                    
                    # Appearance similarity
                    appearance_score = 0.5
                    if track_id in self.appearance_features:
                        feature_distance = np.linalg.norm(det_features - self.appearance_features[track_id])
                        appearance_score = 1.0 / (1.0 + feature_distance)
                    
                    # Time penalty for inactive tracks
                    time_penalty = 0
                    if track_id in inactive_tracks:
                        time_penalty = track['disappeared'] * 2
                    
                    # Confidence bonus for high-confidence detections
                    confidence_bonus = max(0, (det_confidence - 0.5) * 100)
                    
                    # Combined cost calculation (lower is better)
                    if iou > 0.05 or (distance < max_dist and appearance_score > 0.3):
                        cost = (distance * 0.4 + 
                               (1 - iou) * 150 + 
                               (1 - size_ratio) * 80 + 
                               (1 - motion_score) * 100 +
                               (1 - appearance_score) * 120 +
                               time_penalty - 
                               confidence_bonus)
                        
                        cost_matrix[i, j] = max(0, cost)
            
            return cost_matrix, track_ids, all_track_dict
        
        def update(self, detections, frame_idx, frame=None):
            if not detections:
                # Mark all active tracks as disappeared
                for track in self.tracks.values():
                    track['disappeared'] += 1
                
                # Move long-disappeared tracks to inactive
                tracks_to_move = []
                for tid, track in self.tracks.items():
                    if track['disappeared'] >= self.max_disappeared // 3:
                        tracks_to_move.append(tid)
                
                for tid in tracks_to_move:
                    self.inactive_tracks[tid] = self.tracks[tid]
                    del self.tracks[tid]
                
                return {}
            
            # Calculate enhanced cost matrix
            cost_matrix, track_ids, all_tracks = self.calculate_enhanced_cost_matrix(detections, frame)
            
            assignments = {}
            used_detection_indices = set()
            used_track_indices = set()
            
            if cost_matrix.size > 0:
                # Use Hungarian algorithm with stricter threshold
                det_indices, track_indices = linear_sum_assignment(cost_matrix)
                
                for det_idx, track_idx in zip(det_indices, track_indices):
                    if cost_matrix[det_idx, track_idx] < 800:  # Stricter threshold
                        track_id = track_ids[track_idx]
                        assignments[det_idx] = track_id
                        used_detection_indices.add(det_idx)
                        used_track_indices.add(track_idx)
                        
                        detection = detections[det_idx]
                        
                        # Update Kalman filter
                        self.update_kalman(track_id, detection['center'])
                        
                        # Update appearance features
                        if frame is not None:
                            new_features = self.extract_appearance_features(frame, detection['bbox'])
                            if track_id in self.appearance_features:
                                # Running average of appearance features
                                self.appearance_features[track_id] = (0.8 * self.appearance_features[track_id] + 
                                                                    0.2 * new_features)
                            else:
                                self.appearance_features[track_id] = new_features
                        
                        # Update track history
                        self.track_history_buffer[track_id].append(detection['center'])
                        if len(self.track_history_buffer[track_id]) > 10:
                            self.track_history_buffer[track_id].pop(0)
                        
                        # Move back to active if it was inactive
                        if track_id in self.inactive_tracks:
                            self.tracks[track_id] = self.inactive_tracks[track_id]
                            del self.inactive_tracks[track_id]
                        
                        # Update track information
                        self.tracks[track_id].update({
                            'center': detection['center'],
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'class_id': detection['class_id'],
                            'disappeared': 0,
                            'last_seen': frame_idx,
                            'stability_score': self.tracks[track_id].get('stability_score', 0) + 1
                        })
            
            # Handle unmatched tracks
            for j, track_id in enumerate(track_ids):
                if j not in used_track_indices:
                    if track_id in self.tracks:
                        self.tracks[track_id]['disappeared'] += 1
                        
                        # Move to inactive if disappeared too long
                        if self.tracks[track_id]['disappeared'] >= self.max_disappeared // 3:
                            self.inactive_tracks[track_id] = self.tracks[track_id]
                            del self.tracks[track_id]
                    elif track_id in self.inactive_tracks:
                        self.inactive_tracks[track_id]['disappeared'] += 1
            
            # Create new tracks for high-confidence unmatched detections
            for i, detection in enumerate(detections):
                if i not in used_detection_indices and detection['confidence'] > 0.6:
                    # Check if this might be a returning vehicle by comparing with recently retired tracks
                    is_new_vehicle = True
                    
                    # Only create new track if we're confident it's truly new
                    if is_new_vehicle:
                        new_track_id = self.next_id
                        self.next_id += 1
                        
                        # Initialize Kalman filter
                        self.init_kalman_filter(new_track_id, detection['center'])
                        
                        # Initialize appearance features
                        if frame is not None:
                            self.appearance_features[new_track_id] = self.extract_appearance_features(
                                frame, detection['bbox'])
                        
                        # Initialize track history
                        self.track_history_buffer[new_track_id] = [detection['center']]
                        
                        self.tracks[new_track_id] = {
                            'center': detection['center'],
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'class_id': detection['class_id'],
                            'disappeared': 0,
                            'last_seen': frame_idx,
                            'created': frame_idx,
                            'stability_score': 1
                        }
                        assignments[i] = new_track_id
            
            # Remove tracks that have been gone too long and retire their IDs
            tracks_to_remove = []
            for tid, track in self.inactive_tracks.items():
                if track['disappeared'] >= self.max_disappeared:
                    tracks_to_remove.append(tid)
            
            for tid in tracks_to_remove:
                del self.inactive_tracks[tid]
                self.retired_tracks.add(tid)
                if tid in self.kalman_filters:
                    del self.kalman_filters[tid]
                if tid in self.appearance_features:
                    del self.appearance_features[tid]
                if tid in self.track_history_buffer:
                    del self.track_history_buffer[tid]
            
            return assignments

    # Initialize enhanced tracker with more conservative parameters
    tracker = EnhancedTracker(max_disappeared=120, max_distance=300, min_confidence=0.4)
    
    # Enhanced tracking data structures
    trajectories = {}
    car_frame_data = defaultdict(list)
    prev_positions = {}
    track_confidence_history = defaultdict(list)
    min_track_length = 8  # Increased minimum track length

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

        # Run detection with more conservative parameters
        results = model(
            frame,
            classes=car_class_ids,
            conf=0.25,  # Lower confidence threshold for detection
            iou=0.6,    # More permissive IoU for NMS
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

            # Prepare detections for enhanced tracker
            for box, conf, cls in zip(boxes, confs, classes):
                if cls in car_class_ids and conf > 0.3:  # Lower threshold for tracking
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [cx, cy],
                        'confidence': float(conf),
                        'class_id': int(cls)
                    })

            # Update tracker with current detections and frame
            assignments = tracker.update(detections, frame_idx, frame)
            
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
                
                # Track confidence history with more samples
                track_confidence_history[car_id].append(conf)
                avg_confidence = np.mean(track_confidence_history[car_id][-15:])  # More samples for average
                
                # Enhanced speed calculation
                speed = 0.0
                if car_id in prev_positions:
                    prev_frame, prev_cx, prev_cy = prev_positions[car_id]
                    frame_diff = frame_idx - prev_frame
                    if frame_diff > 0:
                        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                        speed = dist / frame_diff
                
                prev_positions[car_id] = (frame_idx, cx, cy)

                # Enhanced frame data for JSON
                stability_score = tracker.tracks.get(track_id, {}).get('stability_score', 0)
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
                    "stability_score": int(stability_score),
                    "class": class_names.get(cls, 'Vehicle'),
                    "class_id": int(cls)
                }
                
                car_frame_data[car_id].append(frame_data)

                # Enhanced visualization
                if show_stream:
                    # Color based on track stability and confidence
                    if stability_score > 20 and avg_confidence > 0.7:
                        color = (0, 255, 0)  # Green for very stable tracks
                    elif stability_score > 10 and avg_confidence > 0.5:
                        color = (0, 255, 255)  # Yellow for stable tracks
                    elif track_id in tracker.inactive_tracks:
                        color = (255, 0, 255)  # Magenta for inactive but tracked
                    else:
                        color = (0, 0, 255)  # Red for new/unstable tracks
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Labels with enhanced info
                    label = f"{class_names.get(cls, 'Vehicle')} ID:{track_id}"
                    conf_text = f"Conf:{conf:.2f} Avg:{avg_confidence:.2f}"
                    stability_text = f"Stability:{stability_score} Speed:{speed:.1f}"
                    
                    # Text background
                    cv2.rectangle(frame, (int(x1), int(y1)-85), (int(x1)+350, int(y1)), (0,0,0), -1)
                    cv2.putText(frame, label, (int(x1), int(y1)-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, conf_text, (int(x1), int(y1)-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, stability_text, (int(x1), int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw center and predicted position
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    cv2.circle(frame, bottom_center, 3, (255, 0, 255), -1)
                    
                    # Draw predicted position for active tracks
                    predicted_pos = tracker.predict_position(track_id)
                    if predicted_pos:
                        cv2.circle(frame, tuple(predicted_pos), 6, (255, 255, 0), 2)

        if show_stream:
            # Enhanced info overlay
            active_tracks = len(tracker.tracks)
            inactive_tracks = len(tracker.inactive_tracks)
            retired_tracks = len(tracker.retired_tracks)
            info_text = f"Frame: {frame_idx}/{total_frames} | Active: {active_tracks} | Inactive: {inactive_tracks} | Retired: {retired_tracks}"
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

    # Enhanced filtering for track quality
    filtered_car_frame_data = {}
    for car_id, frames in car_frame_data.items():
        # More sophisticated filtering based on track quality
        avg_confidence = np.mean([f['avg_confidence'] for f in frames])
        max_stability = max([f['stability_score'] for f in frames])
        
        if len(frames) >= min_track_length and avg_confidence > 0.4 and max_stability > 5:
            filtered_car_frame_data[car_id] = frames
        else:
            print(f"Filtered out track {car_id}: {len(frames)} frames, avg_conf: {avg_confidence:.2f}, max_stability: {max_stability}")

    # Enhanced metadata
    output_data = {
        "metadata": {
            "total_frames": total_frames,
            "fps": fps,
            "video_path": video_path,
            "total_tracks": len(filtered_car_frame_data),
            "min_track_length": min_track_length,
            "total_detections": sum(len(frames) for frames in filtered_car_frame_data.values()),
            "unique_track_ids": len(set(int(car_id.split('_')[1]) for car_id in filtered_car_frame_data.keys())),
            "retired_track_ids": len(tracker.retired_tracks)
        },
        "tracks": filtered_car_frame_data
    }
    
    # Before the JSON writing section, add this check:
    if os.path.isdir(json_out_path):
        # If it's a directory, create a filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_out_path = os.path.join(json_out_path, f"{video_name}_enhanced_tracking.json")

    with open(json_out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Enhanced tracking completed. Found {len(filtered_car_frame_data)} valid tracks.")
    print(f"Unique track IDs used: {output_data['metadata']['unique_track_ids']}")
    print(f"Results saved to: {json_out_path}")

# To run directly if this file is executed
if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [4, 5]:
        print("Uso: python melancolia.py <video_path> <json_out_path> <csv_out_path> [stream]")
    else:
        show_stream = len(sys.argv) == 5 and sys.argv[4].lower() == "stream"
        run_tracking(sys.argv[1], sys.argv[2], sys.argv[3], show_stream)

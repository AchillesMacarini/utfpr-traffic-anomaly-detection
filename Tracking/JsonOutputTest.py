import sys
import os
import json
import cv2
import time
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VehicleDetectionTracker import VehicleDetectionTracker

class VehicleDetectionLogger:
    def __init__(self, output_file="detection_results.json", total_frames=None):
        self.output_file = output_file
        self.results = []
        self.frame_count = 0
        self.total_frames = total_frames
        self.start_time = time.time()
        self.last_progress_update = 0
        self.last_progress_time = 0
        
    def format_detection_result(self, result):
        """Format detection result as clean JSON structure"""
        self.frame_count += 1
        timestamp = datetime.now().isoformat()
        
        # Extract vehicle data in clean format
        vehicles = []
        for vehicle in result.get('detected_vehicles', []):
            speed_info = vehicle.get('speed_info', {})
            
            # Parse color and model info if available
            color_info = []
            model_info = []
            try:
                if vehicle.get('color_info'):
                    color_data = json.loads(vehicle['color_info'])
                    color_info = [{"color": c.get("color", "unknown"), "probability": float(c.get("prob", 0))} for c in color_data[:3]]
            except:
                pass
                
            try:
                if vehicle.get('model_info'):
                    model_data = json.loads(vehicle['model_info'])
                    model_info = [{"make": m.get("make", "unknown"), "model": m.get("model", "unknown"), "probability": float(m.get("prob", 0))} for m in model_data[:3]]
            except:
                pass
            
            vehicle_data = {
                "id": vehicle.get("vehicle_id"),
                "type": vehicle.get("vehicle_type", "unknown"),
                "confidence": round(vehicle.get("detection_confidence", 0), 3),
                "coordinates": {
                    "x": round(vehicle.get("vehicle_coordinates", {}).get("x", 0), 1),
                    "y": round(vehicle.get("vehicle_coordinates", {}).get("y", 0), 1),
                    "width": round(vehicle.get("vehicle_coordinates", {}).get("width", 0), 1),
                    "height": round(vehicle.get("vehicle_coordinates", {}).get("height", 0), 1)
                },
                "speed": {
                    "kph": round(speed_info.get("kph", 0), 1) if speed_info.get("kph") else None,
                    "reliability": speed_info.get("reliability", 0),
                    "direction": speed_info.get("direction_label", "Unknown"),
                    "direction_radians": round(speed_info.get("direction", 0), 3) if speed_info.get("direction") else None
                },
                "color_predictions": color_info,
                "model_predictions": model_info
            }
            vehicles.append(vehicle_data)
        
        # Create clean output structure
        output = {
            "frame": self.frame_count,
            "timestamp": timestamp,
            "summary": {
                "total_vehicles_detected": result.get("number_of_vehicles_detected", 0),
                "active_vehicles": len(vehicles)
            },
            "vehicles": vehicles
        }
        
        return output

    def update_progress_bar(self):
        """Update and display progress bar (updates in place, not too frequently)"""
        current_time = time.time()
        
        # Only update progress bar every 1 second to avoid issues
        if current_time - self.last_progress_time < 1.0 and self.frame_count > 1:
            return
        
        self.last_progress_time = current_time
        
        if self.total_frames is None or self.total_frames <= 0:
            # If no total frames, show simple counter
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            # Clear the entire line and rewrite
            sys.stdout.write('\r' + ' ' * 100)  # Clear line
            sys.stdout.write(f"\r🎬 Processing frame {self.frame_count:4d} | ⚡{fps:4.1f} FPS")
            sys.stdout.flush()
            return
            
        # Calculate progress
        progress = self.frame_count / self.total_frames
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        # Calculate timing
        elapsed_time = current_time - self.start_time
        if self.frame_count > 0:
            avg_time_per_frame = elapsed_time / self.frame_count
            remaining_frames = self.total_frames - self.frame_count
            eta_seconds = remaining_frames * avg_time_per_frame
            
            # Format ETA
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            
            if eta_hours > 0:
                eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
            else:
                eta_str = f"{eta_minutes:02d}:{eta_secs:02d}"
        else:
            eta_str = "--:--"
        
        # Create progress bar
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percentage = progress * 100
        
        # Calculate FPS
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Format elapsed time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)
        elapsed_str = f"{elapsed_minutes:02d}:{elapsed_secs:02d}"
        
        # Create the progress line
        progress_line = f"[{bar}] {percentage:5.1f}% | Frame {self.frame_count:4d}/{self.total_frames} | ⚡{fps:4.1f} FPS | ⏱️{elapsed_str} | ETA: {eta_str}"
        
        # Clear the entire line and rewrite to ensure proper overwriting
        sys.stdout.write('\r' + ' ' * 120)  # Clear line with spaces
        sys.stdout.write('\r' + progress_line)  # Write new progress
        sys.stdout.flush()

    def result_callback(self, result):
        """Process and store detection results"""
        try:
            formatted_result = self.format_detection_result(result)
            self.results.append(formatted_result)
            
            # Update progress bar (this will update in place)
            self.update_progress_bar()
            
        except Exception as e:
            print(f"\n❌ Error in result callback: {e}")
            print(f"Frame: {self.frame_count}")
            # Continue processing even if there's an error

    def save_results(self):
        """Save all results to JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Final progress update - clear the progress bar line
        sys.stdout.write('\r' + ' ' * 120)  # Clear the progress bar line
        if self.total_frames:
            progress_bar = '█' * 50
            sys.stdout.write(f"\r[{progress_bar}] 100.0% | Completed!\n")
        else:
            sys.stdout.write("\r🎬 Processing completed!\n")
        sys.stdout.flush()
        
        # Calculate final statistics
        total_elapsed = time.time() - self.start_time
        avg_fps = len(self.results) / total_elapsed if total_elapsed > 0 else 0
        
        elapsed_hours = int(total_elapsed // 3600)
        elapsed_minutes = int((total_elapsed % 3600) // 60)
        elapsed_secs = int(total_elapsed % 60)
        
        if elapsed_hours > 0:
            time_str = f"{elapsed_hours}h {elapsed_minutes}m {elapsed_secs}s"
        else:
            time_str = f"{elapsed_minutes}m {elapsed_secs}s"
        
        print()
        print("=" * 90)
        print("🎉 PROCESSING COMPLETE!")
        print(f"📁 Results saved to: {self.output_file}")
        print(f"🎬 Total frames processed: {len(self.results):,}")
        print(f"⏱️  Total processing time: {time_str}")
        print(f"⚡ Average processing speed: {avg_fps:.1f} FPS")
        print("=" * 90)

def get_video_info(video_path):
    """Get video information including frame count, FPS, and duration"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return total_frames, fps, duration_seconds

def process_video_headless(vehicle_tracker, video_path, result_callback):
    """Process video without displaying OpenCV windows"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_count += 1
                timestamp = datetime.now()
                
                # Process frame without window display
                response = vehicle_tracker.process_frame(frame, timestamp)
                
                # Call the callback with the response
                result_callback(response)
                
            else:
                # Break the loop if the end of the video is reached
                break
                
    except Exception as e:
        print(f"\n❌ Error processing frame {frame_count}: {e}")
    finally:
        # Release the video capture object
        cap.release()
        print(f"\n📹 Video processing completed. Processed {frame_count} frames.")

# Main execution
def main():
    # Directory containing the videos
    video_dir = r"D:/UTFPR/TCC/AI-City Challenge/aic21-track4-train-data"
    video_template = os.path.join(video_dir, "{}.mp4")

    print("🎬 Vehicle Detection and Tracking Analysis")
    print("=" * 90)

    for i in range(1, 51):
        video_path = video_template.format(i)
        if not os.path.exists(video_path):
            print(f"❌ Skipping: {os.path.basename(video_path)} (file not found)")
            continue

        print(f"\n📹 Video: {os.path.basename(video_path)}")
        print("🔍 Analyzing video file...")
        total_frames, fps, duration = get_video_info(video_path)

        if total_frames is None:
            print("❌ Error: Could not open video file!")
            print(f"   Path: {video_path}")
            continue

        # Format duration
        duration_hours = int(duration // 3600)
        duration_minutes = int((duration % 3600) // 60)
        duration_secs = int(duration % 60)

        if duration_hours > 0:
            duration_str = f"{duration_hours}h {duration_minutes}m {duration_secs}s"
        else:
            duration_str = f"{duration_minutes}m {duration_secs}s"

        print(f"📊 Video Info:")
        print(f"   • Total frames: {total_frames:,}")
        print(f"   • Frame rate: {fps:.1f} FPS")
        print(f"   • Duration: {duration_str}")

        # Create logger instance with frame count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = r"D:/UTFPR/TCC/AI-City Challenge"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"vehicle_detection_results_{i:03d}.json")
        logger = VehicleDetectionLogger(output_file, total_frames)

        print(f"💾 Output file: {output_file}")
        print()
        print("🚀 Starting processing... (Press Ctrl+C to stop)")
        print("📌 Note: Processing without window display for better performance")
        print("=" * 90)

        try:
            # Initialize vehicle detection
            vehicle_detection = VehicleDetectionTracker()

            # Process video with a custom processor to avoid window display issues
            process_video_headless(vehicle_detection, video_path, logger.result_callback)

        except KeyboardInterrupt:
            print("\n\n⏹️  Processing interrupted by user!")
            break
        except Exception as e:
            print(f"\n\n❌ Error during processing: {str(e)}")
        finally:
            # Save results
            logger.save_results()

if __name__ == "__main__":
    main()
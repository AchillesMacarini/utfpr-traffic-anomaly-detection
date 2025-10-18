import os
import subprocess
import sys

def run_melancolia_batch():
    # Define paths
    video_directory = r"D:\UTFPR\TCC\AI-City Challenge\aic21-track4-train-data"
    melancolia_script = r"D:\UTFPR\TCC\utfpr-traffic-anomaly-detection\Tracking\melancolia.py"
    output_directory = r"D:\UTFPR\TCC\AI-City Challenge\tracking_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process videos 1.mp4 to 66.mp4
    for i in range(5, 6):
        video_file = f"{i}.mp4"
        video_path = os.path.join(video_directory, video_file)
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found, skipping...")
            continue
        
        # Define output paths
        json_output = os.path.join(output_directory, f"{i}_tracking.json")
        csv_output = os.path.join(output_directory, f"{i}_tracking.csv")
        
        print(f"Processing video {i}/66: {video_file}")
        
        try:
            # Run melancolia.py with the current video
            subprocess.run([
                sys.executable, melancolia_script,
                video_path,
                json_output,
                csv_output
            ], check=True)
            
            print(f"Successfully processed {video_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_file}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {video_file}: {e}")
            continue
    
    print("Batch processing completed!")

if __name__ == "__main__":
    run_melancolia_batch()
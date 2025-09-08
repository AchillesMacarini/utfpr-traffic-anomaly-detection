import cv2
import os
import torch
from tqdm import tqdm
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
      import melancolia_colab # Attempt to import the module directly
      print("Successfully imported melancolia_colab") # Indicate successful import
  except ModuleNotFoundError:
      print(f"Error: Could not import 'melancolia_colab'. Make sure 'melancolia_colab.py' exists in {tracking_path}")
      sys.exit(1) # Exit if import fails


  def main():
      # Assuming the video directory is also within your Google Drive
      video_dir = os.path.join(drive_path, "aic21-track4-train-data")
      num_videos = 100

      for i in range(1, num_videos + 1):
        video_path = f"{video_dir}/{i}.mp4"
        print(f"Processando vídeo: {video_path}")
        melancolia_colab.run_tracking(video_path, f"trajectories_{i}.json", f"trajectories_{i}.csv", show_stream=False)

  if __name__ == "__main__":
        main()

else:
  print("Path does not exist")
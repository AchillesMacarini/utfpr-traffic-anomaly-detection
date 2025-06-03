import cv2
import numpy as np
import os
import argparse
import shutil
from glob import glob

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def stabilize_video(frames):
    stabilized_frames = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((len(frames)-1, 3), np.float32)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        if m is None:
            m = np.eye(2, 3)
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0])
        transforms[i-1] = [dx, dy, da]
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    h, w = frames[0].shape[:2]
    stabilized_frames.append(frames[0])
    for i in range(1, len(frames)):
        dx = transforms_smooth[i-1,0]
        dy = transforms_smooth[i-1,1]
        da = transforms_smooth[i-1,2]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ])
        stabilized_frame = cv2.warpAffine(frames[i], m, (w, h))
        stabilized_frames.append(stabilized_frame)
    return stabilized_frames

def smooth_trajectory(trajectory, radius=30):
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:,i] = moving_average(trajectory[:,i], radius)
    return smoothed

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    filter = np.ones(window_size)/window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, filter, mode='same')
    return curve_smoothed[radius:-radius]

def enhance_frame(frame):
    # Convert to YUV and apply histogram equalization on the Y channel
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def preprocess_video(input_path, output_path):
    frames = read_video(input_path)
    print(f"Read {len(frames)} frames from {input_path}")
    stabilized_frames = stabilize_video(frames)
    print("Video stabilization completed.")
    enhanced_frames = [enhance_frame(f) for f in stabilized_frames]
    print("Image enhancement completed.")
    save_video(enhanced_frames, output_path)
    print(f"Preprocessed video saved to {output_path}")

def save_video(frames, output_path, fps=30):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess videos.")
    parser.add_argument('--src_folder', type=str, required=True, help='Source folder containing videos')
    parser.add_argument('--dst_folder', type=str, required=True, help='Destination folder for processed videos')
    parser.add_argument('--num_videos', type=int, default=10, help='Number of videos to process')
    args = parser.parse_args()

    os.makedirs(args.dst_folder, exist_ok=True)
    video_files = sorted(glob(os.path.join(args.src_folder, '*.mp4')))[:args.num_videos]

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_path = os.path.join(args.dst_folder, f"preprocessed_{filename}")
        preprocess_video(video_path, output_path)
        shutil.move(video_path, os.path.join(args.dst_folder, filename))
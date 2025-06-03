import cv2
import numpy as np
import os
import argparse
import shutil
from glob import glob
from multiprocessing import Pool

def preprocess_video(input_path, output_path):
    def stabilize_video(input_path):
        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read the first frame
        ret, prev = cap.read()
        if not ret:
            cap.release()
            return input_path  # fallback: return original if failed

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        transforms = []

        # Reduce number of points and increase minDistance for speed
        max_corners = 50
        min_distance = 50
        smoothing_radius = 10  # less smoothing for speed

        for _ in range(n_frames - 1):
            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_corners, qualityLevel=0.03, minDistance=min_distance, blockSize=3
            )
            if prev_pts is None:
                transforms.append([0, 0, 0])
                prev_gray = curr_gray
                continue

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            if curr_pts is None or status is None:
                transforms.append([0, 0, 0])
                prev_gray = curr_gray
                continue

            valid_prev_pts = prev_pts[status.flatten() == 1]
            valid_curr_pts = curr_pts[status.flatten() == 1]

            if len(valid_prev_pts) < 6:
                transforms.append([0, 0, 0])
                prev_gray = curr_gray
                continue

            m, _ = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)
            if m is None:
                transforms.append([0, 0, 0])
                prev_gray = curr_gray
                continue

            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])

            transforms.append([dx, dy, da])
            prev_gray = curr_gray

        transforms = np.array(transforms)
        if len(transforms) == 0:
            cap.release()
            return input_path

        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(
                trajectory[:, i], np.ones(smoothing_radius) / smoothing_radius, mode='same'
            )
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        stabilized_frames = []

        # Only process every 2nd frame for speed (downsample)
        frame_idx = 0
        for i in range(len(transforms_smooth)):
            success, frame = cap.read()
            if not success or frame is None:
                break
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue  # skip every other frame

            dx = transforms_smooth[i][0]
            dy = transforms_smooth[i][1]
            da = transforms_smooth[i][2]

            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            stabilized_frame = cv2.warpAffine(frame, m, (w, h))
            stabilized_frames.append(stabilized_frame)
            frame_idx += 1

        cap.release()

        # Save stabilized video (with reduced fps if frames were skipped)
        out_fps = fps / 2 if fps > 2 else fps
        stabilized_path = input_path.replace('.mp4', '_stabilized.mp4')
        save_video(stabilized_frames, stabilized_path, out_fps)

        return stabilized_path

    stabilized_path = stabilize_video(input_path)


def save_video(frames, output_path, fps=30):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    args = argparse.Namespace(
        src_folder='D:/UTFPR/TCC/AI-City Challenge/aic21-track4-test-data',
        dst_folder='D:/UTFPR/TCC/test',
        num_videos=1
    )

    os.makedirs(args.dst_folder, exist_ok=True)
    video_files = sorted(glob(os.path.join(args.src_folder, '*.mp4')))[:args.num_videos]

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_path = os.path.join(args.dst_folder, f"preprocessed_{filename}")
        preprocess_video(video_path, output_path)
        shutil.move(video_path, os.path.join(args.dst_folder, filename))
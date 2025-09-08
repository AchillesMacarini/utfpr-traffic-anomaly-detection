from Tracking import melancolia

def main():
    video_dir = r"Train_data"
    num_videos = 100

    for i in range(1, num_videos + 1):
        video_path = f"{video_dir}/{i}.mp4"
        print(f"Processando vídeo: {video_path}")
        melancolia.run_tracking(video_path, f"trajectories_{i}.json", f"trajectories_{i}.csv", show_stream=False)

if __name__ == "__main__":
    main()
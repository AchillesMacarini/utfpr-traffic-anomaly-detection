from Tracking import melancolia

def main():
    video_dir = "{...}/AI-City Challenge/aic21-track4-train-data"
    num_videos = 100

    for i in range(1, num_videos + 1):
        video_path = f"{video_dir}/{i}.mp4"
        print(f"Processando vídeo: {video_path}")
        melancolia.run_tracking(video_path, f"trajectories_{i}.json", f"trajectories_{i}.csv")

if __name__ == "__main__":
    main()
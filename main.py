from Tracking import melancolia
from DataEngineering.anomalyProcessing import load_anomaly_annotations, extract_sequences
from Models.wgan_gp import WGAN_GP

def main():
    video_dir = r"Train_data"
    num_videos = 100
    extract_videos = False

    print("Starting WGAN-GP training...")
    video_dir = r"D:/UTFPR/TCC/AI-City Challenge/newDataExtracted"
    annotation_file = "D:/UTFPR/TCC/AI-City Challenge/train-anomaly-results.csv"

    print("Loading and processing data...")
    # Load anomaly annotations
    annotation_dict = load_anomaly_annotations(annotation_file)
    
    print("Extracting sequences...")
    # Extract sequences
    X, y = extract_sequences(video_dir, annotation_dict)
    
    print(f"Initializing WGAN-GP with data shape: {X.shape}")
    # Initialize WGAN-GP model
    wgan_gp = WGAN_GP(input_shape=X.shape[1:], latent_dim=100)
    
    print("Training WGAN-GP model...")
    # Train the WGAN-GP model
    wgan_gp.train(X, epochs=10000, batch_size=64)

if __name__ == "__main__":
    main()
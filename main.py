from DataEngineering.anomalyProcessing import load_anomaly_annotations, extract_and_save_sequences
from Models import wgan_gp
from Models.wgan_gp import WGAN_GP
import numpy as np
import os

def main():
    video_dir = r"extractedData"
    annotation_file = r"train-anomaly-results.csv"

    print("Loading and processing data...")
    # Load anomaly annotations
    annotation_dict = load_anomaly_annotations(annotation_file)
    
    print("Extracting sequences...")
    # Extract sequences and save to disk, also get DataFrame for inspection
    df_sequences = extract_and_save_sequences(video_dir, annotation_dict, sequence_length=3, return_df=True)

    # Print detailed DataFrame information
    print("\nSequence distribution across videos:")
    print(df_sequences['video_id'].value_counts())
    
    print("\nSample sequences from each video:")
    for vid in sorted(df_sequences['video_id'].unique()):
        print(f"\nVideo {vid}:")
        print(df_sequences[df_sequences['video_id'] == vid].head(2))
    
    print("\nDataFrame info:")
    print(df_sequences.info())

    # Load all sequence files for training
    sequences_dir = "sequences"
    sequence_files = [f for f in os.listdir(sequences_dir) if f.endswith(".npz")]
    if not sequence_files:
        raise FileNotFoundError("No sequence files found in the 'sequences' directory.")
    
    # Combine data from all sequence files
    X_list = []
    for seq_file in sequence_files:
        data = np.load(os.path.join(sequences_dir, seq_file))
        X = data["X"]
        if X.size > 0:  # Only append if not empty
            X_list.append(X)
    
    if not X_list:
        raise ValueError("No valid sequences found in any file")
    
    X = np.concatenate(X_list, axis=0)
    
    # Ensure proper shape (samples, sequence_length, features)
    if len(X.shape) == 2:  # (samples, sequence_length)
        X = X[..., np.newaxis]  # Add feature dimension
    
    X = X.astype(np.float32)

    print(f"Final data shape: {X.shape}")
    
    # Initialize WGAN-GP model with correct input shape
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)
    wgan_gp = WGAN_GP(input_shape=input_shape, latent_dim=100)
    
    print("Training WGAN-GP model...")
    # Train the WGAN-GP model
    wgan_gp.train(X, epochs=1, batch_size=128)  # Use larger batch size for GPU

if __name__ == "__main__":
    main()
   
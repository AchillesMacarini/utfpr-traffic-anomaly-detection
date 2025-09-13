import numpy as np
import tensorflow as tf
import os
from tqdm import trange


class WGAN_GP:
    def __init__(self, input_shape, latent_dim):
        # input_shape should be (seq_len, features, 1)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.compile_models()

    def build_generator(self):
        # Calculate intermediate shapes
        sequence_length = self.input_shape[0]  # 3
        features = self.input_shape[1]  # 1
        total_dims = sequence_length * features
        
        model = tf.keras.Sequential([
            # Start with Input layer
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            # Dense layer to get enough units
            tf.keras.layers.Dense(total_dims * 16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            # Reshape to prepare for Conv1D
            tf.keras.layers.Reshape((sequence_length, features * 16)),
            
            # Conv1D layers
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            # Final Conv1D to get desired feature dimension
            tf.keras.layers.Conv1D(features, kernel_size=3, padding='same', activation='tanh'),
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            # Input layer with correct shape
            tf.keras.layers.Input(shape=self.input_shape),
            
            # Conv1D layers
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            
            # Flatten and Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def compile_models(self):
        # Compile discriminator
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='mse'
        )

    def train(self, X_train, epochs, batch_size, log_file='training_log.txt'):
        # Prepare log file
        with open(log_file, 'w') as f:
            f.write('epoch,d_loss,g_loss\n')

        steps_per_epoch = X_train.shape[0] // batch_size

        for epoch in trange(epochs, desc="Epochs"):
            d_losses = []
            g_losses = []

            for step in trange(steps_per_epoch, desc="Batches", leave=False):
                # Train discriminator
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_samples = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_samples = self.generator.predict(noise, verbose=0)

                d_loss_real = self.discriminator.train_on_batch(real_samples, -np.ones((batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.ones((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Append scalar loss value
                d_losses.append(float(d_loss))

                # Train generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_samples = self.generator.predict(noise, verbose=0)
                g_loss = self.discriminator.train_on_batch(fake_samples, -np.ones((batch_size, 1)))
                
                # Append scalar loss value
                g_losses.append(float(g_loss))

            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)

            # Logging
            log_msg = f"Epoch {epoch+1}/{epochs} [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_d_loss},{avg_g_loss}\n")
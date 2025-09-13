import numpy as np
import tensorflow as tf
import os
from tqdm import trange


class WGAN_GP:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=self.latent_dim))
        model.add(tf.keras.layers.Reshape((1, 1, 128)))
        model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=self.input_shape))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        return model

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
                d_losses.append(d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss)

                # Train generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.discriminator.train_on_batch(self.generator.predict(noise, verbose=0), -np.ones((batch_size, 1)))
                g_losses.append(g_loss[0] if isinstance(g_loss, (list, np.ndarray)) else g_loss)

            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)

            # Logging
            log_msg = f"Epoch {epoch+1}/{epochs} [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_d_loss},{avg_g_loss}\n")
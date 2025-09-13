import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

class WGAN_GP:
    def __init__(self, input_shape, latent_dim, n_critic=5, gp_weight=0.1):  # Lowered from 2.0 to 0.1
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight

        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0, beta_2=0.9)  # Lower critic LR

    def build_generator(self):
        sequence_length = self.input_shape[0]
        features = self.input_shape[1]
        total_dims = sequence_length * features

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            tf.keras.layers.Dense(total_dims * 16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Reshape((sequence_length, features * 16)),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(features, kernel_size=3, padding='same', activation='tanh'),
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolates)
            d_interpolates = self.critic(interpolates)
        gradients = gp_tape.gradient(d_interpolates, interpolates)  # FIX: use gp_tape.gradient, not tf.gradients
        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_norm = tf.norm(gradients, axis=1)
        gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        d_real = d_fake = gp = 0.0  # For logging
        # Train critic
        for _ in tf.range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as tape:
                fake_samples = self.generator(noise, training=True)
                fake_output = self.critic(fake_samples, training=True)
                real_output = self.critic(real_samples, training=True)
                d_fake = tf.reduce_mean(fake_output)
                d_real = tf.reduce_mean(real_output)
                c_loss = d_fake - d_real
                gp = self.gradient_penalty(real_samples, fake_samples)
                c_loss = c_loss + self.gp_weight * gp
            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_samples = self.generator(noise, training=True)
            fake_output = self.critic(fake_samples, training=True)
            g_loss = -tf.reduce_mean(fake_output)
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return c_loss, g_loss, d_real, d_fake, gp

    def train(self, X_train, epochs, batch_size=64, log_file='training_log.txt'):
        dataset = (
            tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32))
            .shuffle(50000)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        steps_per_epoch = len(X_train) // batch_size

        for epoch in trange(epochs, desc="Epochs"):
            d_losses = []
            g_losses = []
            d_real_vals = []
            d_fake_vals = []
            gp_vals = []
            for real_batch in tqdm(dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}"):
                c_loss, g_loss, d_real, d_fake, gp = self.train_step(real_batch)
                d_losses.append(float(c_loss))
                g_losses.append(float(g_loss))
                d_real_vals.append(float(d_real))
                d_fake_vals.append(float(d_fake))
                gp_vals.append(float(gp))
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            avg_d_real = np.mean(d_real_vals)
            avg_d_fake = np.mean(d_fake_vals)
            avg_gp = np.mean(gp_vals)
            print(f"Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, D(real)={avg_d_real:.4f}, D(fake)={avg_d_fake:.4f}, GP={avg_gp:.4f}")
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_d_loss},{avg_g_loss},{avg_d_real},{avg_d_fake},{avg_gp}\n")
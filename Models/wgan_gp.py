import numpy as np
import tensorflow as tf
import os
from tqdm import trange

# Configure CUDA and TensorFlow GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set TensorFlow to use mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Optional: Set which GPU to use if you have multiple
        # tf.config.set_visible_devices(gpus[0], 'GPU')
        
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu.device_type}: {gpu.name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU devices found. Running on CPU")

def print_cuda_info():
    print("\nCUDA Configuration:")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
    print(f"GPU Device: {tf.test.gpu_device_name()}")
    print(f"Eager Execution: {tf.executing_eagerly()}")
    # Add version information
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")

class WGAN_GP:
    def __init__(self, input_shape, latent_dim, n_critic=5, gp_weight=10.0):
        # Move to GPU strategy scope
        self.strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {self.strategy.num_replicas_in_sync}')
        
        with self.strategy.scope():
            self.input_shape = input_shape
            self.latent_dim = latent_dim
            self.n_critic = n_critic
            self.gp_weight = gp_weight
            self.generator = self.build_generator()
            self.critic = self.build_critic()
            
            # Use mixed precision for faster training
            self.g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
            )
            self.c_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
            )

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

    def build_critic(self):
        # Similar to your discriminator but without BatchNormalization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # No activation function for Wasserstein loss
        ])
        return model

    @tf.function
    def gradient_penalty(self, real_samples, fake_samples):
        # Ensure computation happens on GPU
        with tf.device('/GPU:0'):
            batch_size = tf.shape(real_samples)[0]
            alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
            diff = fake_samples - real_samples
            interpolated = real_samples + alpha * diff

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic(interpolated, training=True)

            grads = gp_tape.gradient(pred, interpolated)[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            return gp

    @tf.function
    def train_step(self, real_samples):
        # Ensure computation happens on GPU
        with tf.device('/GPU:0'):
            batch_size = tf.shape(real_samples)[0]
            
            # Train critic
            for _ in range(self.n_critic):
                noise = tf.random.normal([batch_size, self.latent_dim])
                with tf.GradientTape() as tape:
                    fake_samples = self.generator(noise, training=True)
                    fake_output = self.critic(fake_samples, training=True)
                    real_output = self.critic(real_samples, training=True)
                    
                    c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
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
            
            return c_loss, g_loss

    def train(self, X_train, epochs, batch_size=64, log_file='training_log.txt'):  # increased batch_size
        # Optimize dataset pipeline for GPU
        dataset = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .cache()  # Cache the data in memory
            .shuffle(50000)  # Increased buffer size
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        steps_per_epoch = len(X_train) // batch_size

        for epoch in trange(epochs, desc="Epochs"):
            d_losses = []
            g_losses = []

            for real_batch in trange(dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}"):
                c_loss, g_loss = self.train_step(real_batch)
                d_losses.append(float(c_loss))
                g_losses.append(float(g_loss))

            # Log results
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_d_loss},{avg_g_loss}\n")
import tensorflow as tf
from tensorflow.keras import layers, models, losses

class MemoryModule(layers.Layer):
    """Revised memory module that handles batch processing correctly"""
    def __init__(self, mem_dim=100, mem_feat=256, alpha=0.1, **kwargs):
        super(MemoryModule, self).__init__(**kwargs)
        self.mem_dim = mem_dim
        self.mem_feat = mem_feat
        self.alpha = alpha
        self.memory = self.add_weight(
            name='memory',
            shape=(self.mem_dim, self.mem_feat),
            initializer='glorot_uniform',
            trainable=True)
        
    def call(self, inputs):
        # inputs shape: [batch_size, mem_feat]
        # memory shape: [mem_dim, mem_feat]
        
        # Normalize inputs and memory
        inputs_norm = tf.math.l2_normalize(inputs, axis=1)  # [batch_size, mem_feat]
        memory_norm = tf.math.l2_normalize(self.memory, axis=1)  # [mem_dim, mem_feat]
        
        # Calculate cosine similarity
        cos_sim = tf.matmul(inputs_norm, memory_norm, transpose_b=True)  # [batch_size, mem_dim]
        
        # Get attention weights
        attn = tf.nn.softmax(cos_sim * 10, axis=1)  # [batch_size, mem_dim]
        
        # Get memory updates
        mem_update = tf.matmul(attn, memory_norm)  # [batch_size, mem_feat]
        
        # Update memory (only update closest memory items)
        closest_idx = tf.argmax(attn, axis=1)  # [batch_size]
        one_hot = tf.one_hot(closest_idx, depth=self.mem_dim)  # [batch_size, mem_dim]
        memory_updates = tf.matmul(one_hot, inputs_norm, transpose_a=True)  # [mem_dim, mem_feat]
        
        # Apply updates with learning rate alpha
        updated_memory = self.memory * (1 - self.alpha) + memory_updates * self.alpha
        self.memory.assign(updated_memory)
        
        return mem_update

class SpectralNormalization(layers.Wrapper):
    """Spectral normalization for stability"""
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration
        
    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.u = self.add_weight(
                shape=(1, self.layer.kernel.shape[-1]),
                initializer='random_normal',
                trainable=False,
                name='sn_u')
        
    def call(self, inputs):
        self._compute_weights()
        output = self.layer(inputs)
        return output
        
    def _compute_weights(self):
        w = self.layer.kernel
        w_shape = w.shape.as_list()
        w_reshaped = tf.reshape(w, [-1, w_shape[-1]])
        
        u = self.u
        for _ in range(self.iteration):
            v = tf.math.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
            
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        self.u.assign(u)
        self.layer.kernel.assign(w / sigma)
        
class SMAAE(tf.keras.Model):
    """Revised SMAAE with proper batch handling"""
    def __init__(self, input_dim=153, latent_dim=32, mem_dim=100, **kwargs):
        super(SMAAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mem_dim = mem_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='leaky_relu'),
            layers.Dense(64, activation='leaky_relu'),
            layers.Dense(latent_dim)
        ])
        
        # Memory module (now with correct batch handling)
        self.memory = MemoryModule(mem_dim=mem_dim, mem_feat=latent_dim)
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='leaky_relu'),
            layers.Dense(128, activation='leaky_relu'),
            layers.Dense(input_dim)
        ])
        
        # Discriminator
        self.discriminator = tf.keras.Sequential([
            layers.Dense(64, activation='leaky_relu'),
            layers.Dense(32, activation='leaky_relu'),
            layers.Dense(1)
        ])
        
        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.memory_loss_tracker = tf.keras.metrics.Mean(name="memory_loss")
        
    def call(self, inputs):
        z = self.encoder(inputs)
        z_hat = self.memory(z)
        reconstructed = self.decoder(z_hat)
        return reconstructed
    
    def compile(self, ae_optimizer, d_optimizer, **kwargs):
        super(SMAAE, self).compile(**kwargs)
        self.ae_optimizer = ae_optimizer
        self.d_optimizer = d_optimizer
        
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            z = self.encoder(data)
            z_hat = self.memory(z)
            reconstructed = self.decoder(z_hat)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MeanSquaredError()(data, reconstructed)
            )
            
            # Memory loss
            memory_loss = tf.reduce_mean(
                tf.keras.losses.MeanSquaredError()(z, z_hat)
            )
            
            # Discriminator loss
            real_output = self.discriminator(z)
            fake_output = self.discriminator(z_hat)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            # Total loss
            total_loss = reconstruction_loss + memory_loss + 0.1 * d_loss
            
        # Train encoder+decoder+memory
        ae_grads = tape.gradient(
            total_loss,
            self.encoder.trainable_variables + 
            self.decoder.trainable_variables + 
            self.memory.trainable_variables
        )
        self.ae_optimizer.apply_gradients(
            zip(
                ae_grads,
                self.encoder.trainable_variables + 
                self.decoder.trainable_variables + 
                self.memory.trainable_variables
            )
        )
        
        # Train discriminator
        d_grads = tape.gradient(
            d_loss,
            self.discriminator.trainable_variables
        )
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.discriminator_loss_tracker.update_state(d_loss)
        self.memory_loss_tracker.update_state(memory_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "memory_loss": self.memory_loss_tracker.result(),
        }

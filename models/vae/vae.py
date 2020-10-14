import tensorflow as tf
import os
import time
import numpy as np
import glob

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_data, inner_shape):
        '''
        Initialise the convolutional autoencoder
        Parameters
        ----------
            latent_dim : dimension of the latent space
            input_shape : shape of the input array
            inner_shape : shape of the array before entering the laternt space
        '''
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_data = input_data
        self.inner_shape = inner_shape
        self.inference_net = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=input_data),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=4, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=4, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
          ) 

        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=inner_shape[0]*inner_shape[1]
                               *32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(inner_shape[0], inner_shape[1], 32)),
            tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=4,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
            tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=4,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]  
            )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def run_inference(model, x, sigmoid=False):
       mean, logvar = model.encode(x)
       z = model.reparameterize(mean, logvar)
       return model.decode(z, apply_sigmoid=sigmoid)

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
      super(VAE, self).__init__()
      self.latent_dim = latent_dim
      self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(40, 400, 1)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          #tf.keras.layers.dense(256, activation=tf.nn.relu),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
      )

      self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          tf.keras.layers.Dense(40*400, activation=tf.nn.sigmoid),
          tf.keras.layers.Reshape(target_shape=(40, 400, 1)),
        ]
      )  

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits
   
    def run_inference(model, x, sigmoid=False):
       mean, logvar = model.encode(x)
       z = model.reparameterize(mean, logvar)
       return model.decode(z, apply_sigmoid=sigmoid)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, x, y, sigmoid=False):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z, apply_sigmoid=sigmoid)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                               logits=x_logit, labels=y)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, y, optimizer, sigmoid=False):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y, sigmoid=sigmoid)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

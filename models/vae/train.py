import numpy as np
import tensorflow as tf
import time
from vae import CVAE
from vae import compute_apply_gradients, compute_loss
tf.config.experimental_run_functions_eagerly(True)

def norm_im(im):
    maxval = np.max(im)
    minval = np.min(im)
    return (im - minval)/(maxval - minval)
#############################
split = 3500
datapatha = '<datadir>/dimer/brille/simulate_dimer_newer.npy'
datapathb = '<datadir>/goodenough/brille/simulate_goodenough_new.npy'
epochs = 100
latent_dims = [20]
input_data = (240, 400, 1)
inner_shape = (60, 100)
optimizer = tf.keras.optimizers.Adam(1e-4)
load_weights_file = False
save_weights_file = './vae-denoise-pairs-'
##############################

train_images = np.concatenate((np.load(datapatha), np.load(datapathb)))
np.random.shuffle(train_images)
Xo = np.zeros(shape=(train_images.shape))
for i, t in enumerate(train_images):
    Xo[i] = norm_im(t)
Xo = np.float32(Xo)
Xo = np.expand_dims(Xo, axis=3)
X = Xo[:split]
xtest = Xo[split:split+100]

labels = X
ltest = xtest


for latent_dim in latent_dims:
    model = CVAE(latent_dim, input_data, inner_shape)
    if load_weights_file:
        model.load_weights(load_weights_file)

    for epoch in range(1, epochs + 1):
          start_time = time.time()
          for index, train_x in enumerate(X):
              trx = np.expand_dims(train_x, axis=0)
              trl = np.expand_dims(labels[index], axis=0)
              compute_apply_gradients(model, trx, trl, optimizer, sigmoid=False)
          end_time = time.time()

          if epoch % 1 == 0:
              loss = tf.keras.metrics.Mean()
              for index, test_x in enumerate(xtest):
                  test_x = np.expand_dims(test_x, axis=0)
                  test_y = np.expand_dims(ltest[index], axis=0)
                  loss(compute_loss(model, test_x, test_y, sigmoid=False))
              elbo = -loss.result()
              print('Epoch: {0:5d}, Test set ELBO: {1:10.4f}, '
              'time elapse for current epoch {2:8.3f}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
    if save_weights_file:
        model.save_weights(save_weights_file + '%s.h5' % str(latent_dim))
    del model 

#outputs = np.zeros(shape=(xtest.shape))
#for i, ex in enumerate(xtest):
#    outputs[i] = model.run_inference(np.expand_dims(ex, axis=0), sigmoid=True)

#np.save('outputs.npy', outputs)

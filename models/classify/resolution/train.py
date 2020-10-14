import sys
import os
import numpy as np
import random
from utils import norm_data
from models import cnn_global
import tensorflow.keras.optimizers as opts
from tensorflow.keras.callbacks import  ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import pickle
import tensorflow as tf

# INPUTS
datapath = ('<datadir>')
ge_data_file = datapath + '/goodenough/resolution/simulated.npy'
di_data_file = datapath + '/dimer/resolution/simulated.npy'
weights_file = 'weights.hdf5' 
load_prev = False
epochs = 200
batch_size = 128
rate = 0.001
cutoff = 6000
null_hypothesis = False
train_data_file = 'training_data.pickle'
### NO FURTHER EDITING REQUIRED
def norm_im(im):
    maxval = np.max(im)
    minval = np.min(im)
    return (im - minval)/(maxval - minval)

if os.path.exists(ge_data_file):
    ge_data = np.load(ge_data_file)[:2000]
else:
    print('Goodenough data not found.')
if os.path.exists(di_data_file):
    di_data = np.load(di_data_file)[:2000]
else:
    print('Dimer data not found.')

print(len(ge_data), len(di_data))
labels = np.zeros((len(ge_data) + len(di_data), 1))
labels[:len(ge_data)] = 1.

X, y = shuffle(np.concatenate((ge_data, di_data)), labels)
y = np.array([int(b[0]) for b in y])
X = np.clip(X, 0, 120)
d = [norm_im(i) for i in X]
X = np.array(d)
np.nan_to_num(X, copy = False, nan=0)

Xtrain = X[:3000]
ytrain = y[:3000]
Xtest = X[3000:]
ytest = y[3000:]

print(Xtest.shape, ytest.shape)

stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=40)
checkpoint = ModelCheckpoint('h5-model-6layer.h5',
                             verbose=1, monitor='val_loss',
                             save_best_only=True, mode='auto')

if null_hypothesis:
    random.shuffle(Xtrain)
new_model = cnn_global(nlayers=6)
print(new_model.summary())

if os.path.exists(weights_file) and not null_hypothesis and load_prev:
    new_model.load_weights(weights_file)
opt = opts.Adam(lr=rate)
new_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = new_model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=epochs, batch_size=batch_size, 
             callbacks=[stopping_callback, checkpoint])
pickle.dump(history.history, open( "train_history.pkl", "wb" ) )
# serialize weights to HDF5

if not null_hypothesis:
    new_model.save_weights(weights_file)

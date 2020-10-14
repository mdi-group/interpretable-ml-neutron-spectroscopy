import sys
import os
import numpy as np
import random
from utils import norm_data
from models import cnn_global
import tensorflow.keras.optimizers as opts
from tensorflow.keras.callbacks import  ModelCheckpoint
from sklearn.utils import shuffle
import pickle
import tensorflow as tf

# INPUTS

datapath = ('<datadir>')
ge_data_file = datapath + '/goodenough/brille/simulate_goodenough_new.npy'
di_data_file = datapath + '/dimer/brille/simulated_dimer_newer.npy'
weights_file = 'weights.hdf5' 
load_prev = False
epochs = 500
batch_size = 128
rate = 0.001
cutoff = 6000
null_hypothesis = False
### NO FURTHER EDITING REQUIRED

if os.path.exists(ge_data_file):
    ge_data = np.load(ge_data_file)[:3344]
else:
    print('Goodenough data not found.')
if os.path.exists(di_data_file):
    di_data = np.load(di_data_file)[:3344]
else:
    print('Dimer data not found.')

stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
checkpoint = ModelCheckpoint('h5-model-6layer.h5',
                             verbose=1, monitor='val_loss',
                             save_best_only=True, mode='auto')

labels = np.zeros((len(ge_data) + len(di_data), 2))
labels[:len(ge_data)] = [0., 1.]
labels[len(ge_data):] = [1., 0.]

print('labels:', len(labels), len(ge_data))

X, y = shuffle(np.concatenate((ge_data, di_data)), labels)
X = np.expand_dims(X, axis=3)
X = np.clip(X, 0, 120)
d = [norm_data(i) for i in X]
X = np.array(d)
np.nan_to_num(X, copy = False, nan=0)


Xtrain = X[:cutoff]
Xtest = X[cutoff:] 
ytrain = y[:cutoff]
ytest = y[cutoff:]

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

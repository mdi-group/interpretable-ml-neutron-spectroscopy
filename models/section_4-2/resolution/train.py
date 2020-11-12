import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
import time
import os  
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from models_large import CNN_DUQ
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.style.use('sciml-style')
from tqdm.notebook import tqdm

def norm_im(im):
    maxval = np.max(im)
    minval = np.min(im)
    return (im - minval)/(maxval - minval)

data_path = ('<datadir>')
ge_data_file = data_path + 'goodenough/resolution/simulated.npy'
di_data_file = data_path + 'dimer/resolution/simulated.npy'
#train_data_file = 'training_data.pickle'

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
#X = np.concatenate((ge_data, di_data))
#y = labels
#X = np.expand_dims(X, axis=3)
X = np.moveaxis(X, -1,1)
X = np.clip(X, 0, 120)
d = [norm_im(i) for i in X]
X = np.array(d)
np.nan_to_num(X, copy = False, nan=0)

#print(y[1], encoded_Y[1])

batch_size = 64
X_train = X[:3000]
y_train = y[:3000]
X_test = X[3000:]
y_test = y[3000:]
print(X.shape, y.shape)

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

model = CNN_DUQ(input_size=(240, 400, 1), num_classes=2, embedding_size=64,
               learnable_length_scale=False, length_scale=1., gamma=1.)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
# One sided penalty - down
#     gradient_penalty = F.relu(grad_norm - 1).mean()
    return gradient_penalty


def output_transform_acc(output):
    y_pred, y, x, z = output
    y = torch.argmax(y, dim=1)
    return y_pred, y

def output_transform_bce(output):
    y_pred, y, x, z = output

    return y_pred, y

def output_transform_gp(output):
    y_pred, y, x, z = output

    return x, y_pred


def step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x.requires_grad_(True)
    z, y_pred = model(x)
    loss1 =  F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item()

def eval_step(engine, batch):
    model.eval()
    x, y = batch
    x.requires_grad_(True)
    z, y_pred = model(x)

    return y_pred, y, x, z

trainer = Engine(step)
evaluator = Engine(eval_step)

metric = Accuracy(output_transform=output_transform_acc)
metric.attach(evaluator, "accuracy")

metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
metric.attach(evaluator, "bce")

metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
metric.attach(evaluator, "gp")


ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

t_start = time.time()
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(trainer):
    evaluator.run(dl_test)
    metrics = evaluator.state.metrics
    
    print("Test Results - Epoch: {} Acc: {:.4f} BCE: {:.2f} GP {:.2f} Tot Time: {:.2f} s"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['gp'],
                 time.time() - t_start))

trainer.run(dl_train, max_epochs=100)
torch.save(model.state_dict(), './uq-discrim-resolution.pt')

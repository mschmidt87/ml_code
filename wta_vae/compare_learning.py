from func import train_autoencoder
import matplotlib.pyplot as pl
import numpy as np
import os
import tensorflow as tf
from utils import load_data

"""
This script produces an overview figure to compare the
performance of the three different models.

It creates a figures with the losses on the validation data
for the three models:
default
deterministic lwta
probabilistic lwta
.
"""
try:
    os.mkdir('saved_models/')
except:
    pass
try:
    os.mkdir('saved_models/compare_learning')
except:
    pass

models = ['default', 'lwta', 'p-lwta']
layer_structures = ['none', 'lwta', 'p-lwta']

paths = ['saved_models/compare_learning/default/',
         'saved_models/compare_learning/lwta/',
         'saved_models/compare_learning/p-lwta/']
scripts = ['train_default.py',
           'train_LWTA.py',
           'train_pLWTA.py']

learning_rate = 0.0001


batch_size = 100
steps = 2000
train_examples = 10000
validation_examples = 1000

(mnist,
 (train_data, train_labels),
 (validation_data, validation_labels)) = load_data(train_examples, validation_examples)

fig = pl.figure(figsize=(6., 3.5))
ax = pl.axes()
for model, layer_structure, path in zip(models,
                                        layer_structures,
                                        paths):
    try:
        os.mkdir(path)
    except:
        pass
    session = tf.Session()
    VAEmodel, loss_list, validation_loss_list = train_autoencoder(session,
                                                                  train_data,
                                                                  validation_data,
                                                                  layer_structure,
                                                                  batch_size,
                                                                  steps,
                                                                  learning_rate,
                                                                  path,
                                                                  return_loss=True)
    ax.plot(validation_loss_list, label=model)
ax.legend()
ax.set_xticks(np.arange(0., steps / 10. + 1, 40.))
ax.set_xticklabels(np.arange(0., steps+1, 400, dtype=np.int))
ax.set_xlabel('Step')
ax.set_ylabel('Validation loss')
ax.set_yscale('Log')
pl.savefig('compare_learning.eps')
pl.savefig('compare_learning.png')

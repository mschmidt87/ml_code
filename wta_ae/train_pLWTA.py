import os
import tensorflow as tf
from func import train_autoencoder, reconstruct
from utils import load_data

learning_rate = 0.0001
session = tf.Session()
layer_structure = 'p-lwta'
batch_size = 1000
steps = 2000
model_path = 'saved_models/p-lwta/p-lwta'
try:
    os.mkdir('saved_models/')
except:
    pass
try:
    os.mkdir(model_path)
except:
    pass
train_examples = 10000
validation_examples = 1000
(mnist,
 (train_data, train_labels),
 (validation_data, validation_labels)) = load_data(train_examples, validation_examples)

model = train_autoencoder(session,
                          train_data,
                          validation_data,
                          layer_structure,
                          batch_size,
                          steps,
                          learning_rate,
                          model_path)

model_path = 'saved_models/p-lwta/p-lwta-{}'.format(steps - 1)
figure_path = 'saved_models/p-lwta/results'
try:
    os.mkdir(figure_path)
except:
    pass
for i in range(10):
    out_path = os.path.join(figure_path,
                            'mnist_{}'.format(i))
    reconstruct(i,
                model_path,
                validation_data,
                out_path,
                layer_structure)

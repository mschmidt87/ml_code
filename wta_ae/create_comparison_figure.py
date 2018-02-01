import os
from PIL import Image as pimg
from utils import load_data
import subprocess
import sys

"""
This script produces an overview figure to compare the
performance of the three different models.

It creates a figure with four rows which show:
- The input data images (first 10 figures from the validation data)
and the reconstructed figures from the three models in this order:
default
deterministic lwta
probabilistic lwta
.

First, the validation data is loaded and plotted in the top row.
Then, the script looks for the reconstructed images and if they
have not been reproduced yet, it runs the respective scripts.
"""

models = ['default', 'lwta', 'p-lwta']
paths = ['saved_models/default/results/',
         'saved_models/lwta/results/',
         'saved_models/p-lwta/results/']
scripts = ['train_default.py',
           'train_LWTA.py',
           'train_pLWTA.py']

N_samples = 10
fn = ['mnist_{}.png'.format(i) for i in range(N_samples)]
image = pimg.new('L', (N_samples * 28,
                       4 * 28))

"""
Load the validation data and produce the top row.
"""
train_examples = 10000
validation_examples = 1000
(mnist,
 (train_data, train_labels),
 (validation_data, validation_labels)) = load_data(train_examples, validation_examples)

data = validation_data[:N_samples]

for i in range(N_samples):
    img = pimg.new('L', (28, 28))
    img.putdata(data[i] * 255)
    image.paste(img, box=(i*28, 0, (i+1)*28, 28))

"""
Load the images from the three models.
If necessary, execute the scripts to
train the models first.
"""

for j, (model, path, script) in enumerate(zip(models,
                                              paths,
                                              scripts),
                                          start=1):
    print(j, model, path)
    if not os.path.exists(path) or not all([f in os.listdir(path) for f in fn]):
        print(model)
        subprocess.call([sys.executable,
                         os.path.join(os.getcwd(), script)])

    for i in range(N_samples):
        img = pimg.open(os.path.join(path,
                                     'mnist_{}.png'.format(i)))
        image.paste(img, box=(i*28, j*28, (i+1)*28, (j+1)*28))

image.save('comparison.png')

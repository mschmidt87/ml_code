import argparse
from func import train_autoencoder, reconstruct
import tensorflow as tf
import time
from utils import load_data

parser = argparse.ArgumentParser(description='Winner-take-all VAE')
parser.add_argument('mode',
                    choices=['train', 'reconstruct'],
                    help='run mode: can be "train" to train a model or'
                    '"reconstruct" to reconstruct an MNIST figure with a trained model')
parser.add_argument('--batch-size', type=int, default=100, metavar='batch-size',
                    help='input batch size for training (default: 100)')
parser.add_argument('--steps', type=int, default=1000, metavar='steps',
                    help='number of mini-batches to run the training algorithm on (default: 100)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='learning-rate',
                    help='learning rate for the Adam optimizer (default: 0.001)')
parser.add_argument('--layer-structure', type=str, default='none', metavar='layer-structure',
                    choices=['none', 'lwta', 'p-lwta'],
                    help='structure of the network layers. '
                    '"none": standard structure of single neurons without dropout,'
                    '"lwta": deterministic local winner-take-all,'
                    '"p-lwta": probabilistic softmax dropout during training')
parser.add_argument('--seed', type=int, default=1, metavar='seed',
                    help='random seed (default: 1)')
parser.add_argument('--output-path', type=str, metavar='output_path',
                    default=str(time.time()),
                    help='path to store the reconstructed figure (reconstruct mode)')
parser.add_argument('--mnist_index', type=int, default=0, metavar='mnist_index',
                    help='index of the figure in the MNIST dataset to be reconstructed')
parser.add_argument('--model-path', type=str, metavar='model_path',
                    help='path to store the trained model (train mode) or'
                    'to retrieve the model for reconstruction')

args = parser.parse_args()


train_examples = 10000
validation_examples = 1000
(mnist,
 (train_data, train_labels),
 (validation_data, validation_labels)) = load_data(train_examples, validation_examples)

tf.set_random_seed(args.seed)

session = tf.Session()
if args.mode == 'train':
    model = train_autoencoder(session,
                              train_data,
                              validation_data,
                              args.layer_structure,
                              args.batch_size,
                              args.steps,
                              args.learning_rate,
                              args.model_path)
elif args.mode == 'reconstruct':
    reconstruct(args.mnist_index,
                args.model_path,
                validation_data,
                args.output_path,
                args.layer_structure)

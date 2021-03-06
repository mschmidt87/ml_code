import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data(train_examples, validation_examples):
    """
    Load MNIST data and subsample to the specified number of
    training and validation examples.

    Parameters
    ----------
    train_examples : int
        Number of training examples.
    validation_examples : int
        Number of validation examples.

    Returns
    -------
    mnist : tensorflow dataset
        The complete MNIST dataset
    train_images, train_labels : tuple of numpy.ndarrays
        Images and labels for training
    validation_images, validation_labels : tuple of numpy.ndarrays
        Images and labels for validation
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_images, train_labels = (mnist.train.images[:train_examples],
                                  mnist.train.labels[:train_examples])
    validation_images, validation_labels = (mnist.validation.images[:validation_examples],
                                            mnist.validation.labels[:validation_examples])
    return mnist, (train_images, train_labels), (validation_images, validation_labels)


def loss_function(x_recon, x, mean, logvar, batch_size):
    """
    Loss function of the Variational Autoencoder.
    Uses the mean-squared error for the reconstruction error.

    Parameters
    ----------
    x_recon : tensorflow.Variable
        The reconstructed input from the autoencoder.
    x : tensorflow.Variable
        The input data to the autoencoder
    mean : tensorflow.Variable
        The mean values of the latent variables
    logvar : tensorflow.Variable
        The logarithmic variance of the latent variables
    batch_size : int
        The size of the batch

    Returns
    -------
    loss : tensorflow.Variable
        The value of the loss function
    """

    MSE = tf.losses.mean_squared_error(x_recon, x)
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    KLD /= batch_size * 784
    loss = MSE + KLD
    return loss

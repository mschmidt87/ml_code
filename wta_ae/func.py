from network import Basic_AE, LWTA_AE
import numpy as np
import os
from PIL import Image as pimg
import tensorflow as tf
from utils import loss_function


def train_autoencoder(session, training_data, validation_data,
                      layer_structure, batch_size, steps, learning_rate,
                      output_path, return_loss=False):
    """
    Train an autoencoder with the given data and the given layer_structure.

    Parameters
    ----------
    session: tensorflow.Session
    training_data: numpy.ndarray
        An array containing the data to train the autoencoder on. Each row is
        assumed to be a sample, and the number of columns the dimension of the
        data. In the case of MNIST, the number of columns will be 784 (28 * 28).
    validation_data:
        An array containing validation data. The format is the same as for the
        training_data.
    layer_structure: None or string
        None, "lwta" or "p-lwta" depending on the kind of layer structure to use
        during training.
    output_path : str
        Path to store the model after training.
    return_loss : bool
        Whether to return the trajectory of the loss function. Defaults to False.

    Returns
    -------
    autoencoder: callable
        A function that has the `autoencoder(data)` signature, where data has the
        same format as training_data and validation_data in train_autoencoder
    """
    if layer_structure == 'none':
        autoencoder = Basic_AE()
    else:
        autoencoder = LWTA_AE(layer_structure=layer_structure)

    num_epochs = int(np.ceil((steps * batch_size) / len(training_data)))
    data_placeholder = tf.placeholder(training_data.dtype, training_data.shape)
    dataset = tf.data.Dataset.from_tensor_slices((data_placeholder))
    dataset = dataset.repeat(num_epochs)
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Loss function
    x = tf.placeholder(tf.float32, shape=[None, 784], name="training_data")
    x_recon, z = autoencoder.forward(x)
    loss = loss_function(x_recon, x)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    if return_loss:
        loss_list = []
        validation_loss_list = []
    with session:
        session.run(tf.global_variables_initializer())
        session.run(iterator.initializer, feed_dict={data_placeholder: training_data})

        for step in range(steps):
            batch = session.run(next_element)
            train_step.run(feed_dict={x: batch})
            l = loss.eval(feed_dict={x: batch})

            if step % 10 == 0:
                test_l = loss.eval(feed_dict={x: validation_data})
                print("Step: {}, training loss: {}, test loss: {}".format(step, l, test_l))
                if return_loss:
                    loss_list.append(l)
                    validation_loss_list.append(test_l)

        saver = tf.train.Saver()
        saver.save(session, os.path.join(output_path), global_step=step)
    if return_loss:
        return autoencoder, loss_list, validation_loss_list
    else:
        return autoencoder


def reconstruct(mnist_index, model_path, validation_data, output_path, layer_structure):
    """
    Reconstruct a MNIST figure using a trained tensorflow model.

    Parameters
    ----------
    mnist_index : int
        Index of the figure in the validation dataset.
    model_path : str
        Path to the trained model. This has to include the directory
        of the model files and the filename of the meta datafile
        without the file suffix '.meta'.
    validation_data : numpy.ndarray
        The validation dataset.
    output_path : str
        Path to store the output images.
    layer_structure : str
        Layer structure of the trained model.
        Can be 'none', 'lwta' or 'p-lwta'.
        See the README for further explanation.
    """

    mnist_figure = validation_data[mnist_index]

    with tf.Session() as session:
        x = tf.Variable(np.array([mnist_figure]))
        session.run(tf.global_variables_initializer())

        checkpoint = os.path.split(model_path)[0]
        saver = tf.train.import_meta_graph('.'.join((model_path, 'meta')))
        saver.restore(session, tf.train.latest_checkpoint(checkpoint))
        graph = tf.get_default_graph()
        if layer_structure == 'none':
            autoencoder = Basic_AE(load=True, graph=graph)
        else:
            autoencoder = LWTA_AE(layer_structure=layer_structure,
                                  load=True, graph=graph)
        autoencoder.mode = 'eval'
        x_recon, z = autoencoder.forward(x)

        img = pimg.new('L', (28, 28))
        img.putdata(x_recon.eval()[0]*255)
        if not output_path.endswith('.png'):
            output_path = '.'.join((output_path, 'png'))
        img.save(output_path)

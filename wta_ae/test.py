from network import Basic_AE, p_lwta_unit
import numpy as np
import os
import pytest
import tensorflow as tf
from utils import load_data


def test_p_lwta():
    """
    Test it the probabilistic lwta unit
    produces correct statistics.
    """
    with tf.Session() as sess:
        num_samples = 1000
        x = tf.Variable(np.repeat(np.random.rand(2).reshape(1, 1, -1), num_samples, axis=0))
        sess.run(tf.global_variables_initializer())
        softmax = tf.nn.softmax(x).eval()[0][0]
        l = p_lwta_unit(x).eval()
        a = [np.where(l[:, 0, ii] > 0.)[0].size for ii in range(2)]
        assert(softmax[0] == pytest.approx(float(a[0]) / num_samples, abs=2.))
        assert(softmax[1] == pytest.approx(float(a[1]) / num_samples, abs=2.))


def test_load_model():
    """
    Explicitely tests the save and load mechanism used in
    the `train_autoencoder` and `reconstruct` functions.
    """
    session = tf.Session()
    model_path = 'saved_models/test/test_load'
    try:
        os.mkdir('saved_models')
    except:
        pass
    try:
        os.mkdir(model_path)
    except:
        pass
    train_examples = 10
    validation_examples = 10
    (mnist,
     (train_data, train_labels),
     (validation_data, validation_labels)) = load_data(train_examples, validation_examples)

    # Store a random model (Same code as func.py -> train_autoencoder)
    with session:
        model = Basic_AE()
        session.run(tf.global_variables_initializer())
        W4 = model.W4.eval()
        saver = tf.train.Saver()
        saver.save(session, os.path.join(model_path), global_step=1)

    # Retrieve the stored model (Same code as func.py -> reconstruct)
    model_path = 'saved_models/test/test_load-1'
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        checkpoint = os.path.split(model_path)[0]
        saver = tf.train.import_meta_graph('.'.join((model_path, 'meta')))
        saver.restore(session, tf.train.latest_checkpoint(checkpoint))
        graph = tf.get_default_graph()
        autoencoder = Basic_AE(load=True, graph=graph)
        W4_loaded = autoencoder.W4.eval()

    assert(np.all(W4 == W4_loaded))

import tensorflow as tf


def bias_variable(shape, name, load=False, graph=None):
    """
    Construct a variable for biases.

    Parameters
    ----------
    shape : iterable of int
        Shape of the bias vector.
    name : str
        Name of the tensorflow variable.
    load : bool
        Whether to initilize the variable with random values (False)
        or load from the specified graph (True).
        Defaults to False.
    graph :
        Tensorflow graph that holds a variable with the specified if
        load is set to True. Defaults to None.
    """
    if load:
        try:
            var = graph.get_tensor_by_name(':'.join((name, '0')))
        except KeyError:
            raise ValueError("Variable of the model does not exist in graph.")
    else:
        var = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    return var


def weight_variable(shape, name, load=False, graph=None):
    """
    Construct a variable for weight tensors.

    Parameters
    ----------
    shape : iterable of int
        Shape of the weight tensor.
    name : str
        Name of the tensorflow variable.
    load : bool
        Whether to initilize the variable with random values (False)
        or load from the specified graph (True).
        Defaults to False.
    graph :
        Tensorflow graph that holds a variable with the specified if
        load is set to True. Defaults to None.
    """
    if load:
        try:
            var = graph.get_tensor_by_name(':'.join((name, '0')))
        except KeyError:
            raise ValueError("Variable {} does not exist in graph.".format(name))
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial, name=name)
    return var


def lwta_unit(h, mode=None):
    """
    A local winner-take-all unit that sets all but
    the maximal value in each row of the given tensor to 0.

    Parameters
    ----------
    h : tensorflow.Variable
        Activation tensor of the unit.
    mode : None
        This variable is just a placeholder to enable using
        the function similar to p_lwta_unit.
    Returns
    -------
    transform : tensorflow.Variable
        Transformed activity of the LWTA unit.
    """
    ind = tf.equal(h, tf.reduce_max(h, axis=2, keep_dims=True))
    return tf.multiply(h, tf.cast(ind, h.dtype))


def p_lwta_unit(h, mode='train'):
    """
    A probabilistic local winner-take-all unit with two modes:
    In training mode, it sets all but a randomly drawn value
    to 0 in each row of the given tensor to 0. The dropout
    probabilities are determined by computing the softmax
    over the activations in each row.
    In evaluation mode, it scale all activations by the value
    of the softmax operation.

    Parameters
    ----------
    h : tensorflow.Variable
        Activation tensor of the unit with shape (-1, -1, 2).
    mode : str
        Mode of the unit. Set to 'train' for training mode
        and 'eval' for evaluation mode.
        Defaults to 'train'.

    Returns
    -------
    transform : tensorflow.Variable
        Transformed activity of the LWTA unit.
    """
    softmax = tf.nn.softmax(h, dim=2)
    if mode == 'train':
        dec = tf.less(tf.random_uniform(tf.shape(h)[:-1], dtype=softmax.dtype), softmax[:, :, 0])
        ind = tf.stack([dec, tf.logical_not(dec)], axis=2)
        transform = tf.multiply(h, tf.cast(ind, h.dtype))
    else:
        transform = tf.multiply(h, softmax)
    return transform


class VAE:
    """
    Base class for variational autoencoder.
    """
    def __init__(self, mode='train'):
        """
        Class constructor.

        Parameters
        ----------
        mode : str
            Mode of the network, can be either 'train'
            for training mode or 'eval' for evaluation mode.
        """
        self.mode = 'train'
        self.variables = []

    def __call__(self, inp):
        return self.forward(inp)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick. This function maps
        a random variable drawn from the standard normal
        distribution to a normal distribution with mean value
        mu and logarithmic variance logvar.

        Parameters
        ----------
        mu : tensorflow.Variable
            Mean value of the normal distribution.
        logvar: tensorflow.Variable
            Logarithmic variance of the normal distribution.

        Returns
        -------
        x : tensorflow.Variable
            Transformed normally distributed variable.
        """
        std = tf.exp(tf.multiply(logvar, 0.5))
        eps = tf.random_normal(tf.shape(std))
        x = tf.add(tf.multiply(eps, std), mu)
        return x

    def forward(self, inp):
        """
        Forward path of the autoencoder.
        Calls the encode and decode functions that
        have to implemented by child classes.

        Parameters
        ----------
        inp : tensorflow.Variable
            Input to the autoencoder.

        Returns
        -------
        inp_recon : tensorflow.Variable
            Reconstructed input after encoding and
            decoding.
        mean : tensorflow.Variable
            Encoded mean value of the latent variables.
        logvar: tensorflow.Variable
            Encoded logarithmic variance of the latent variables.
        """
        mean, logvar = self.encode(inp)
        z = self.reparameterize(mean, logvar)
        inp_recon = self.decode(z)
        return inp_recon, mean, logvar


class BasicVae(VAE):
    """
    A basic variational auto-encoder with four layers and relu
    activation functions in the hidden layers and sigmoid activation
    in the output layer.
    """
    def __init__(self, load=False, graph=None):
        """
        Class constructor.

        Parameters
        ----------
        load : bool
           Whether to load the values of the network parameters
           from the specified graph (True) or initialize parameters
           randomly (False). Defaults to False.
        graph : tensorflow Graph
           Tensorflow Graph holding the values for the network
           parameters if load is set to True.
           Defaults to None.
        """
        super(BasicVae, self).__init__()
        # 1st layer
        self.W1 = weight_variable([784, 500], 'W1', load=load, graph=graph)
        self.b1 = bias_variable([500], 'b1', load=load, graph=graph)

        # 2nd layer for mean
        self.W2_mean = weight_variable([500, 100], 'W2-mean', load=load, graph=graph)
        self.b2_mean = bias_variable([100], 'b2-mean', load=load, graph=graph)

        # 2nd layer for logarithmic variance
        self.W2_logvar = weight_variable([500, 100], 'W2-logvar', load=load, graph=graph)
        self.b2_logvar = bias_variable([100], 'b2-logvar', load=load, graph=graph)

        # 3nd layer
        self.W3 = weight_variable([100, 500], 'W3', load=load, graph=graph)
        self.b3 = bias_variable([500], 'b3', load=load, graph=graph)

        # 4nd layer
        self.W4 = weight_variable([500, 784], 'W4', load=load, graph=graph)
        self.b4 = bias_variable([784], 'b4', load=load, graph=graph)

        self.variables = [self.W1, self.b1,
                          self.W2_mean, self.b2_mean,
                          self.W2_logvar, self.b2_logvar,
                          self.W3, self.b3,
                          self.W4, self.b4]

    def encode(self, inp):
        """
        Encoding function of the network.

        Parameters
        ----------
        inp : tensorflow.Variable
            Input to the autoencoder.

        Returns
        -------
        mean : tensorflow.Variable
            Encoded mean value of the latent variables.
        logvar: tensorflow.Variable
            Encoded logarithmic variance of the latent variables.
        """
        x = tf.nn.relu(tf.matmul(inp, self.W1) + self.b1)
        mean = tf.matmul(x, self.W2_mean) + self.b2_mean
        logvar = tf.matmul(x, self.W2_logvar) + self.b2_logvar
        return mean, logvar

    def decode(self, z):
        """
        Decoding function of the network.

        Parameters
        ----------
        z : tensorflow.Variable
            Sampled values of the latent variables.

        Returns
        -------
        inp_recon : tensorflow.Variable
            Reconstructed input after encoding and
            decoding.
        """
        h = tf.nn.relu(tf.matmul(z, self.W3) + self.b3)
        inp_recon = tf.nn.sigmoid(tf.matmul(h, self.W4) + self.b4)
        return inp_recon


class LWTAVae(VAE):
    """
    A variational auto-encoder with four layers and LWTA units in all
    hidden layers.
    """
    def __init__(self, units_per_block=2, layer_structure='lwta',
                 load=False, graph=None):
        """
        Class constructor.

        Parameters
        ----------
        units_per_block : int
           Number of neurons per LWTA unit.
           Defaults to 2.
           Note: Currently, creating more neurons per unit is not implemented.
        layer_structure : str
           Type of LWTA unit. Can be set to 'lwta' to use determinist LWTA or
           'p-lwta' to use probabilistic LWTA. Defaults to 'lwta'.
        load : bool
           Whether to load the values of the network parameters
           from the specified graph (True) or initialize parameters
           randomly (False). Defaults to False.
        graph : tensorflow Graph
           Tensorflow Graph holding the values for the network
           parameters if load is set to True.
           Defaults to None.
        """

        super(LWTAVae, self).__init__()

        self.u_ = units_per_block
        if layer_structure == 'lwta':
            self.act = lwta_unit
        elif layer_structure == 'p-lwta':
            self.act = p_lwta_unit
        else:
            raise NotImplementedError()

        # 1st layer
        self.W1 = weight_variable([784, 250, self.u_], 'W1', load=load, graph=graph)
        self.b1 = bias_variable([250, self.u_], 'b1', load=load, graph=graph)

        # 2nd layer for mean
        self.W2_mean = weight_variable([self.u_, 250, 50, self.u_],
                                       'W2-mean', load=load, graph=graph)
        self.b2_mean = bias_variable([50, self.u_], 'b2-mean', load=load, graph=graph)

        # 2nd layer for logarithmic variance
        self.W2_logvar = weight_variable([self.u_, 250, 50, self.u_],
                                         'W2-logvar', load=load, graph=graph)
        self.b2_logvar = bias_variable([50, self.u_], 'b2-logvar', load=load, graph=graph)

        # 3nd layer
        self.W3 = weight_variable([self.u_, 50, 250, self.u_], 'W3', load=load, graph=graph)
        self.b3 = bias_variable([250, self.u_], 'b3', load=load, graph=graph)

        # 4nd layer
        self.W4 = weight_variable([self.u_, 250, 784], 'W4', load=load, graph=graph)
        self.b4 = bias_variable([784], 'b4', load=load, graph=graph)

        self.variables = [self.W1, self.b1,
                          self.W2_mean, self.b2_mean,
                          self.W2_logvar, self.b2_logvar,
                          self.W3, self.b3,
                          self.W4, self.b4]

    def encode(self, inp):
        """
        Encoding function of the network.

        Parameters
        ----------
        inp : tensorflow.Variable
            Input to the autoencoder.

        Returns
        -------
        mean : tensorflow.Variable
            Encoded mean value of the latent variables.
        logvar: tensorflow.Variable
            Encoded logarithmic variance of the latent variables.
        """
        x = tf.matmul(inp, tf.reshape(self.W1, [784, 500])) + tf.reshape(self.b1, [500])
        x = self.act(tf.reshape(x, [-1, 250, self.u_]), mode=self.mode)

        mean = (tf.matmul(tf.reshape(x, [-1, 500]), tf.reshape(self.W2_mean, [500, 100]))
                + tf.reshape(self.b2_mean, [100]))
        mean = self.act(tf.reshape(mean, [-1, 50, self.u_]), mode=self.mode)
        logvar = (tf.matmul(tf.reshape(x, [-1, 500]), tf.reshape(self.W2_logvar, [500, 100]))
                  + tf.reshape(self.b2_logvar, [100]))
        logvar = self.act(tf.reshape(logvar, [-1, 50, self.u_]), mode=self.mode)
        return mean, logvar

    def decode(self, z):
        """
        Decoding function of the network.

        Parameters
        ----------
        z : tensorflow.Variable
            Sampled values of the latent variables.

        Returns
        -------
        inp_recon : tensorflow.Variable
            Reconstructed input after encoding and
            decoding.
        """
        h = (tf.matmul(tf.reshape(z, [-1, 100]), tf.reshape(self.W3, [100, 500]))
             + tf.reshape(self.b3, [500]))
        h = self.act(tf.reshape(h, [-1, 250, self.u_]), mode=self.mode)

        inp_recon = tf.nn.sigmoid((tf.matmul(tf.reshape(h, [-1, 500]),
                                             tf.reshape(self.W4, [500, 784]))
                                   + tf.reshape(self.b4, [784])))
        return inp_recon

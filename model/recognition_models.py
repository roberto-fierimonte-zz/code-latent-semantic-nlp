import theano
import numpy as np
import theano.tensor as T
from lasagne.layers import DenseLayer, get_all_layers, get_all_param_values, get_all_params, get_output, InputLayer, \
    LSTMLayer, set_all_param_values
from lasagne.nonlinearities import linear
from nn.nonlinearities import elu_plus_one

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams(1234)


class RecModel(object):
    """ Implementation of a Recognition model """

    def __init__(self, z_dim, max_length, vocab_size, dist_z):
        """ Initialise the recognition model

        :param z_dim:           # dimension of the latent space (Z)
        :param max_length:      # maximum length of all the sentences (max(L))
        :param vocab_size:      # number of distinct tokens in the text (V) + 1
        :param dist_z:          # distribution for the latents in the recognition model
        """

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.dist_z = dist_z()

        self.mean_nn, self.cov_nn = self.nn_fn()    # variational mean and covariance for the latents

    def nn_fn(self):
        raise NotImplementedError()

    @staticmethod
    def get_meaningful_words(X, meaningful_mask=None):
        """ Transform a batch of N sentences into their restricted version by removing non-significant words

        :param X:               (N x max(L)) matrix representing the text
        :param meaningful_mask: (N x max(L)) matrix of 1s and -1s representing whether each word is significant or not

        :return:                (N x max(L)) matrix of restricted sentences
        """

        # If a mask is provided, transform all the non-significant words into the <EOS> token and move them to the end
        if meaningful_mask is not None:
            X_tilde = X * meaningful_mask

            for i in range(X_tilde.shape[0]):
                X_tilde[i] = np.concatenate((X_tilde[i, X_tilde[i] >= 0], X_tilde[i, X_tilde[i] < 0]))

            X_tilde[X_tilde < 0] = -1
            return X_tilde
        else:
            return X

    # def get_meaningful_words_symbolic(self, meaningful_mask):
    #
    #     X = T.imatrix('X')
    #     f =  X * meaningful_mask
    #
    #     return theano.function(inputs=[X, meaningful_mask], outputs=f, allow_input_downcast=True)

    def get_means_and_covs(self, X, X_embedded):
        """ Get the mean and the covariance for the distribution for the code z

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text

        :return:                variational mean and covariance for the latents given a sentence
        """

        # If x is less or equal than 0 then return 0, else 1 (used to filter out words)
        mask = T.switch(T.lt(X, 0), 0, 1)                                       # N x max(L)

        # Reshape the embedding of X adding a singleton dimension on the right
        X_embedded *= T.shape_padright(mask)                                    # N x max(L) x E x 1 (broadcastable)

        means = get_output(self.mean_nn, X_embedded)                            # N x Z
        covs = get_output(self.cov_nn, X_embedded)                              # N x Z

        return means, covs

    def get_samples(self, X, X_embedded, num_samples, means_only=False):
        """ Return S samples for z given N sentences

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text
        :param num_samples:     int (S)
        :param means_only:      boolean

        :return:                ((S * N) x Z) matrix of samples (S samples per sentence)
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))       # (S * N) x Z
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])       # (S * N) x Z

        return samples

    def log_q_z(self, z, X, X_embedded):
        """ Compute the logarithm of the code distribution given N sentences

        :param z:               ((S * N) x Z) matrix of code samples
        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text

        :return:                logarithm of the variational density calculated in z (distribution object)
        """

        N = X.shape[0]                                                          # Number of sentences
        S = T.cast(z.shape[0] / N, 'int32')                                     # Number of samples

        means, covs = self.get_means_and_covs(X, X_embedded)                    # (N x Z) and (N x Z)

        means = T.tile(means, [S] + [1]*(means.ndim - 1))                       # (S * N) x Z
        covs = T.tile(covs, [S] + [1]*(means.ndim - 1))                         # (S * N) x Z

        return self.dist_z.log_density(z, [means, covs])

    def kl_std_gaussian(self, X, X_embedded):
        """ Compute the KL divergence between the prior and posterior distribution for the code z given N sentences

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text

        :return:                N -dimensional vector of KL divergences, one per sentence
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))  # N x 1

        return kl

    def get_samples_and_kl_std_gaussian(self, X, X_embedded, num_samples, means_only=False):
        """ Return S samples from z given N sentences and the KL divergences for every sentence

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text
        :param num_samples:     int (S)
        :param means_only:      boolean

        :return:                ((S * N) x Z) samples from the code + N -dimensional vector of KL divergences, one per sentence
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))       # (S * N) x Z
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])       # (S * N) x Z

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def summarise_z(self, all_x, all_x_embedded):
        """

        :param all_x:
        :param all_x_embedded:

        :return:
        """

        def step(all_x_s, all_x_embedded_s):

            means, covs = self.get_means_and_covs(all_x_s, all_x_embedded_s)    # (N x Z) and (N x Z)

            precs = 1. / covs                                                   # N x Z
            weighted_means = means * precs                                      # N x Z

            return precs, weighted_means

        ([all_precs, all_weighted_means], _) = theano.scan(step,
                                                           sequences=[all_x.dimshuffle((1, 0, 2)),
                                                                      all_x_embedded.dimshuffle((1, 0, 2, 3))],
                                                           )

        total_precision = T.sum(all_precs, axis=0)                              # N x Z
        summary_mean = T.sum(all_weighted_means, axis=0) / total_precision      # N x Z

        return summary_mean

    def get_params(self):
        """

        :return:
        """

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params

    def get_param_values(self):
        """

        :return:
        """

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [nn_params_vals]

    def set_param_values(self, param_values):
        """

        :param param_values:
        :return:
        """

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNN(RecModel):
    """ Implementation of a RNN-MLP Recognition model """

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):
        """ Initialise the recognition model

        :param z_dim:           # dimension of the latent space (Z)
        :param max_length:      # maximum length of all the sentences (max(L))
        :param vocab_size:      # number of distinct tokens in the text (V) + 1
        :param dist_z:          # distribution for the latents in the recognition model
        :param nn_kwargs:       # additional parameters for the model (dict)
        """

        self.nn_rnn_depth = nn_kwargs['rnn_depth']                          # Number of RNNs to stack
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']                  # Dimensionality of the hidden layers of the RNNs
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']    # Non-linearity function of the RNNs

        self.nn_nn_depth = nn_kwargs['nn_depth']                            # Number of MLPs to stack
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']                    # Dimensionality of the hidden layers of the MLPs
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']      # Non-linearity function of the MLPs

        super().__init__(z_dim, max_length, vocab_size, dist_z)             # Initialise super-class

        self.rnn = self.rnn_fn()                                            # Initialise RNN

    def rnn_fn(self):
        """ Initialise recognition RNN. The network is structured as an Input Layer, followed by a variable number of
        LSTM layers. The input can be masked

        :return: a list of Lasagne layers
        """

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_mask = InputLayer((None, self.max_length))

        l_prev = l_in

        all_layers = []

        for h in range(self.nn_rnn_depth):

            l_prev = LSTMLayer(l_prev, num_units=self.nn_rnn_hid_units, mask_input=l_mask)

            all_layers.append(l_prev)

        return all_layers

    def nn_fn(self):
        """ Initialise recognition MLP to compute the mean and the covariance of the code z

        :return: One Lasagne layer for the mean and one for the covariance
        """

        l_in = InputLayer((None, self.nn_rnn_depth * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, X, X_embedded):
        """ Get the hidden layers for the recognition RNN given a batch of N sentences

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text

        :return:                rnn_depth -dimensional list of hidden states for the recognition RNN
        """

        # If x is less or equal than 0 then return 0, else 1 (exclude unused words)
        mask = T.switch(T.lt(X, 0), 0, 1)                                           # N x max(L)

        h_prev = X_embedded                                                         # N x max(L) x E

        all_h = []

        for h in range(len(self.rnn)):

            h_prev = self.rnn[h].get_output_for([h_prev, mask])                     # N x max(L) x dim(hid)

            all_h.append(h_prev[:, -1])

        hid = T.concatenate(all_h, axis=-1)

        return hid

    def get_means_and_covs(self, X, X_embedded):
        """ Get the mean and the covariance for the distribution for the code z for the RNN-MLP model

        :param X:               (N x max(L)) matrix representing the text
        :param X_embedded:      (N x max(L) x E) tensor representing the embedded text
        :return:                ((S * N) x Z) matrix of samples (S samples per sentence)
        """

        hid = self.get_hid(X, X_embedded)                                           # N x (depth * dim(hid))

        means = get_output(self.mean_nn, hid)                                       # N x Z
        covs = get_output(self.cov_nn, hid)                                         # N x Z

        return means, covs

    def get_params(self):
        """

        :return:
        """

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):
        """

        :return:
        """

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):
        """

        :param param_values:
        :return:
        """

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)

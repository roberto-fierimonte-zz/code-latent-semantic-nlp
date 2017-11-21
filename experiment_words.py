from model.distributions import GaussianDiagonal, Categorical
from model.generative_models import GenStanfordWords as GenerativeModel
from model.recognition_models import RecRNN as RecognitionModel

import json
import os
import sys

import lasagne

from trainers.sgvb import SGVBWords as SGVB
from run import RunWords as Run

import numpy as np
np.set_printoptions(threshold=3000000)


sys.setrecursionlimit(5000000)


main_dir = '.'
out_dir = './output'

dataset = 'BookCorpus/processed_vocab20000_1to60_limunk0.0'             # dataset folder

with open(os.path.join(main_dir, 'data/', dataset, 'valid_vocab.txt'), 'r') as f:
    valid_vocab = json.loads(f.read())

eos_ind = valid_vocab.index('<EOS>')                                    # index of the EOS token


solver = SGVB                                                           # solver


vocab_size = len(valid_vocab)                                           # size of the vocabulary
restrict_min_length = 5                                                 # minimum length of the sentence
restrict_max_length = 30                                                # maximum length of the sentence
train_prop = 0.9                                                        # fraction of data to use as the training set

d_z = 50                                                                # dimension of the latent space
d_emb = 300                                                             # dimension of the embedding

gen_nn_kwargs = {
    'rnn_depth': 1,                                                     # depth of the generative RNN
    'rnn_hid_units': 500,                                               # number of hidden units in the generative RNN
    'rnn_hid_nonlinearity': lasagne.nonlinearities.tanh,                # type of generative RNN non-linearity
    'rnn_time_steps': 40,                                               # number of generative RNN time steps (maximum length of the sentence)
    'read_attention_nn_depth': 2,                                       # depth of the Attention mechanism
    'read_attention_nn_hid_units': 500,                                 # number of hidden units in the Attention mechanism
    'read_attention_nn_hid_nonlinearity': lasagne.nonlinearities.tanh,  # type of Attention mechanism non-linearity
}

rec_nn_kwargs = {
    'rnn_depth': 1,                                                     # depth of the recognition RNN
    'rnn_hid_units': 500,                                               # number of Hidden units for the recognition RNN
    'rnn_hid_nonlinearity': lasagne.nonlinearities.tanh,                # type of recognition RNN non-linearity
    'nn_depth': 2,                                                      # depth of the recognition MLP
    'nn_hid_units': 500,                                                # number of Hidden units for the recognition MLP
    'nn_hid_nonlinearity': lasagne.nonlinearities.tanh,                 # type of recognition MLP non-linearity
}

solver_kwargs = {'generative_model': GenerativeModel,                   # generative model to use
                 'recognition_model': RecognitionModel,                 # recognition model to use
                 'vocab_size': vocab_size,                              # size of vocabulary
                 'gen_nn_kwargs': gen_nn_kwargs,                        # params for the generative model
                 'rec_nn_kwargs': rec_nn_kwargs,                        # params for the recognition model
                 'z_dim': d_z,                                          # dimension of the latent variable
                 'embedding_dim': d_emb,                                # dimension of the embedding
                 'dist_z_gen': GaussianDiagonal,                        # distribution for the latents in the generative model
                 'dist_x_gen': Categorical,                             # distribution for the observed in the generative model
                 'dist_z_rec': GaussianDiagonal,                        # distribution for the latents in the observed model
                 'eos_ind': eos_ind,                                    # index of the EOS token
                 }

pre_trained = False                                                     # if we already have the parameters
load_param_dir = 'code_outputs/2017_08_23_09_11_02'                     # directory with the saved parameters

train = True                                                            # do not train if we already have the parameters

training_iterations = 10000                                             # number of training iterations
training_batch_size = 5                                                 # number of sentences per training iteration
training_num_samples = 1                                                # number of samples per sentence per training iteration
warm_up = 20000                                                         # number of KL annealing iterations
word_drop = 0.3                                                         # percentage of words to drop to prevent vanishing KL term

grad_norm_constraint = None                                             # to prevent vanishing/exploding gradient
update = lasagne.updates.adam                                           # the optimisation algorithm
update_kwargs = {'learning_rate': 0.0001}                               # parameters for the optimisation algorithm

val_freq = 1000                                                         # how often to perform evaluation
val_batch_size = 100                                                    # number of sentences per per evaluation iteration
val_num_samples = 1                                                     # number of samples per sentence per evaluation iteration

save_params_every = 50000                                               # check-point for parameters saving

# The second part of the settings is task-specific
generate_output_prior = False                                           # ???
generate_output_posterior = False                                       # ???
generate_canvases = False                                               # ???

impute_missing_words = False                                            # ???
drop_rate = 0.2                                                         # ???
missing_chars_iterations = 500                                          # ???

find_best_matches = False                                               # ???
num_best_matches = 20                                                   # ???
best_matches_batch_size = 100                                           # ???

follow_latent_trajectory = False                                        # ???
latent_trajectory_steps = 10                                            # ???

num_outputs = 100                                                       # ???


test = True                                                             # ???
test_batch_size = 500                                                   # ???
test_num_samples = 100                                                  # ???
test_sub_sample_size = 10                                               # ???


if __name__ == '__main__':

    """
    Regardless of the task we need:
    """

    run = Run(dataset=dataset,                                          # a folder where to read the dataset from
              valid_vocab=valid_vocab,                                  # a valid vocabulary
              solver=solver,                                            # a solver
              solver_kwargs=solver_kwargs,                              # params for the solver
              main_dir=main_dir,                                        # the directory for the project
              out_dir=out_dir,                                          # the directory where to export the output
              pre_trained=pre_trained,                                  # to know whether the model is already trained
              load_param_dir=load_param_dir,                            # the directory where to load the parameters from
              restrict_min_length=restrict_min_length,                  # the minimum length of the sentence
              restrict_max_length=restrict_max_length,                  # the maximum length of the sentence
              train_prop=train_prop,                                    # the proportion of data to use for training
              )                                    # additional params for the simulation

    if train:

        """
        In case we want to train the model we need:
        """

        run.train(n_iter=training_iterations,                           # the number of training iterations
                  batch_size=training_batch_size,                       # number of sentences per batch
                  num_samples=training_num_samples,                     # number of samples per sentence per training iteration
                  grad_norm_constraint=grad_norm_constraint,            # to prevent vanishing/exploding gradient
                  update=update,                                        # the optimisation algorithm
                  update_kwargs=update_kwargs,                          # parameters for the optimisation algorithm
                  val_freq=val_freq,                                    # how often to perform evaluation
                  val_batch_size=val_batch_size,                        # number of sentences per per evaluation iteration
                  val_num_samples=val_num_samples,                      # number of samples per sentence per evaluation iteration
                  warm_up=warm_up,                                      # number of KL annealing iterations
                  save_params_every=save_params_every,                  # check-point for parameters saving
                  word_drop=word_drop)                                  # percentage of words to drop to prevent vanishing KL term

    if generate_output_prior or generate_output_posterior:
        run.generate_output(prior=generate_output_prior,
                            posterior=generate_output_posterior,
                            num_outputs=num_outputs)

    if generate_canvases:
        run.generate_canvases(num_outputs=num_outputs)

    if impute_missing_words:
        run.impute_missing_words(num_outputs=num_outputs,
                                 drop_rate=drop_rate,
                                 num_iterations=missing_chars_iterations)

    if find_best_matches:
        run.find_best_matches(num_outputs=num_outputs,
                              num_matches=num_best_matches,
                              batch_size=best_matches_batch_size)

    if follow_latent_trajectory:
        run.follow_latent_trajectory(num_samples=num_outputs,
                                     num_steps=latent_trajectory_steps)

    if test:
        run.test(test_batch_size, test_num_samples, test_sub_sample_size)

from model.generative_models import GenStanfordWords, GenAUTRWords, GenLatentGateWords
from model.recognition_models import RecRNN, RecMLP
from model.distributions import GaussianDiagonal, Categorical
from trainers.sgvb import SGVBWords as SGVB
from run import RunWords as Run

import numpy as np
import argparse
import lasagne
import json
import sys
import os

np.set_printoptions(threshold=3000000)
sys.setrecursionlimit(5000000)


def check_percentage(value):
    fvalue = float(value)
    if fvalue > 1.:
        fvalue /= 100
    if fvalue < 0. or fvalue > 1.:
        raise argparse.ArgumentTypeError("%s is an invalid percentage value" % value)
    return fvalue


def check_natural(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid natural number" % value)
    return ivalue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Params of the model
    parser.add_argument('--vocab-size', default=20000, dest='vocab_size', type=int)
    parser.add_argument('--lim-unk', default=0.0, type=check_percentage)
    parser.add_argument('--min', '--min-length', default=11, dest='restrict_min_length', type=int)
    parser.add_argument('--max', '--max-length', default=30, dest='restrict_max_length', type=int)
    parser.add_argument('--dataset', default='BookCorpus', type=str)
    parser.add_argument('--generative-model', '--gen-model', '--decoder', default='Stanford', dest='generative_model')
    parser.add_argument('--generative-net', '--gen-net', default='LSTM', dest='generative_network',
                        choices=['LSTM', 'GRU'])
    parser.add_argument('--recognition-model', '--rec-model', '--encoder', default='LSTM', dest='recognition_model')

    # Params of the training
    parser.add_argument('--most-common', default=-1, dest='most_common', type=int)
    parser.add_argument('--max-iter', default=100000, type=int, dest='max_iter')
    # parser.add_argument('--optimal-beta', action='store_true', dest='optimal_ratio')
    parser.add_argument('-m', default=0., type=float)
    # parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--no-word-drop', action='store_true', dest='no_word_drop')
    parser.add_argument('--no-annealing', action='store_true', dest='no_annealing')
    parser.add_argument('--no-teacher-forcing', action='store_true', dest='no_teacher_forcing')

    # Params of the task
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pre-trained', action='store_true', dest='pre_trained')
    parser.add_argument('--gen-prior', action='store_true', dest='generate_prior')
    parser.add_argument('--gen-posterior', action='store_true', dest='generate_posterior')
    parser.add_argument('--interpolate-latents', action='store_true', dest='interpolate_latents')
    parser.add_argument('--interpolate-latents-posterior', action='store_true', dest='interpolate_latents_posterior')
    parser.add_argument('--impute-missing', '--missing_words', '--missing', '--impute', action='store_true',
                        dest='impute_missing_words')
    parser.add_argument('--find-best', '--find-matches', '--best-matches', action='store_true',
                        dest='find_best_matches')

    # Other params
    parser.add_argument('-v', '--verbose', default=True)
    parser.add_argument('--save-arrays', action='store_true', dest='save_arrays')

    args = parser.parse_args()

    if args.generative_model == 'Stanford':
        GenerativeModel = GenStanfordWords
    elif args.generative_model in ['AUTR', 'Canvas']:
        GenerativeModel = GenAUTRWords
    elif args.generative_model == 'TopicRNN':
        GenerativeModel = GenLatentGateWords
    else:
        raise ValueError('{} is not a valid generative model'.format(args.generative_model))

    if args.recognition_model == 'LSTM':
        RecognitionModel = RecRNN
    elif args.recognition_model in ['MLP', 'FNN', 'NN']:
        RecognitionModel = RecMLP
    else:
        raise ValueError('{} is not a valid recognition model'.format(args.recognition_model))

    main_dir = '.'

    if args.generative_model != 'Stanford' or args.generative_network == 'LSTM':
        out_dir = './output/{}/{}_{}_{}_{}to{}'.format(args.dataset, GenerativeModel.__name__,
                                                       RecognitionModel.__name__,
                                                       args.vocab_size, args.restrict_min_length,
                                                       args.restrict_max_length)
    else:
        out_dir = './output/{}/{}_{}{}_{}_{}to{}'.format(args.dataset, GenerativeModel.__name__,
                                                         args.generative_network, RecognitionModel.__name__,
                                                         args.vocab_size, args.restrict_min_length,
                                                         args.restrict_max_length)
    if args.most_common > 0:
        out_dir = out_dir + 'Top{}'.format(args.most_common)
    else:
        out_dir = out_dir + 'All'

    # if args.optimal_ratio:
    #     out_dir = out_dir + '_OptBeta'

    if args.m != 0.:
        out_dir = out_dir + '_m{}'.format(args.m)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    dataset = '{}/processed_vocab{}_1to60_limunk{}'.format(args.dataset, args.vocab_size, args.lim_unk)  # dataset folder

    with open(os.path.join(main_dir, 'data/', dataset, 'valid_vocab.txt'), 'r') as f:
        valid_vocab = json.loads(f.read())

    eos_ind = valid_vocab.index('<EOS>')                                    # index of the EOS token

    solver = SGVB                                                           # solver
    vocab_size = len(valid_vocab)                                           # size of the vocabulary
    train_prop = 0.9                                                        # fraction of data to use as the training set

    d_z = 50                                                                # dimension of the latent space
    d_emb = 300                                                             # dimension of the embedding

    gen_nn_kwargs = {
        'rnn_depth': 1,                                                     # depth of the generative RNN
        'rnn_type': args.generative_network,                                # which RNN type to use
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
                     'm': args.m,                                           # balancing factor for AutoGen training
                     'teacher_forcing': not args.no_teacher_forcing         #
                     }

    # if we already have the parameters
    pre_trained = args.pre_trained                                          # use existing parameters
    load_param_dir = out_dir                                                # directory with the saved parameters

    train = args.train                                                      # do not train if we already have the parameters

    training_iterations = args.max_iter                                     # number of training iterations
    training_batch_size = 200                                               # number of sentences per training iteration
    training_num_samples = 1                                                # number of samples per sentence per training iteration

    if args.no_annealing:
        warm_up = None
    else:
        warm_up = 20000                                                     # number of KL annealing iterations

    if args.no_word_drop:
        word_drop = None
    else:
        word_drop = 0.3                                                     # percentage of words to drop to prevent vanishing KL term

    grad_norm_constraint = None                                             # to prevent vanishing/exploding gradient
    update = lasagne.updates.adam                                           # the optimisation algorithm
    update_kwargs = {'learning_rate': 0.0001}                               # parameters for the optimisation algorithm

    val_freq = 10000                                                        # how often to perform evaluation
    val_batch_size = 100                                                    # number of sentences per per evaluation iteration
    val_num_samples = 1                                                     # number of samples per sentence per evaluation iteration

    save_params_every = 100                                                 # check-point for parameters saving

    # The second part of the settings is task-specific
    generate_output_prior = args.generate_prior                                           # ???
    generate_output_posterior = args.generate_posterior                                   # ???
    generate_canvases = False                                                             # ???

    impute_missing_words = args.impute_missing_words                                      # ???
    drop_rate = 0.2                                                                       # ???
    missing_chars_iterations = 500                                                        # ???

    find_best_matches = args.find_best_matches                                            # ???
    num_best_matches = 20                                                                 # ???
    best_matches_batch_size = 100                                                         # ???

    follow_latent_trajectory = args.interpolate_latents                                   # ???
    follow_latent_trajectory_posterior = args.interpolate_latents_posterior
    latent_trajectory_steps = 5                                                           # ???

    num_outputs = 100                                                                     # ???

    test = args.test                                                                      # ???
    test_batch_size = 500                                                                 # ???
    test_num_samples = 100                                                                # ???
    test_sub_sample_size = 10                                                             # ???

    run_kwargs = {'most_common': args.most_common,
                  'exclude_eos': False,
                  'save_arrays': args.save_arrays}

    # Initialise the model
    run = Run(dataset=dataset,                                              # a folder where to read the dataset from
              valid_vocab=valid_vocab,                                      # a valid vocabulary
              solver=solver,                                                # a solver
              solver_kwargs=solver_kwargs,                                  # params for the solver
              main_dir=main_dir,                                            # the directory for the project
              out_dir=out_dir,                                              # the directory where to export the output
              pre_trained=pre_trained,                                      # to know whether the model is already trained
              load_param_dir=load_param_dir,                                # the directory where to load the parameters from
              restrict_min_length=args.restrict_min_length,                 # the minimum length of the sentence
              restrict_max_length=args.restrict_max_length,                 # the maximum length of the sentence
              train_prop=train_prop,                                        # the proportion of data to use for training
              **run_kwargs)                                                 # additional params for the simulation

    if train:
        run.train(n_iter=training_iterations,                                   # the number of training iterations
                  batch_size=training_batch_size,                               # number of sentences per batch
                  num_samples=training_num_samples,                             # number of samples per sentence per training iteration
                  grad_norm_constraint=grad_norm_constraint,                    # to prevent vanishing/exploding gradient
                  update=update,                                                # the optimisation algorithm
                  update_kwargs=update_kwargs,                                  # parameters for the optimisation algorithm
                  val_freq=val_freq,                                            # how often to perform evaluation
                  val_batch_size=val_batch_size,                                # number of sentences per per evaluation iteration
                  val_num_samples=val_num_samples,                              # number of samples per sentence per evaluation iteration
                  warm_up=warm_up,                                              # number of KL annealing iterations
                  save_params_every=save_params_every,                          # check-point for parameters saving
                  word_drop=word_drop)                                          # percentage of words to drop to prevent vanishing KL term

    if generate_output_prior or generate_output_posterior:
        run.generate_output(prior=generate_output_prior,
                            posterior=generate_output_posterior,
                            num_outputs=num_outputs)

    if follow_latent_trajectory:
        run.follow_latent_trajectory(num_samples=num_outputs,
                                     num_steps=latent_trajectory_steps)

    if follow_latent_trajectory_posterior:
        run.follow_latent_trajectory_posterior(num_outputs=num_outputs,
                                               num_steps=latent_trajectory_steps)

    if test:
        run.test(test_batch_size, test_num_samples, test_sub_sample_size)

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

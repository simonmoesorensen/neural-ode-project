import argparse

import matplotlib
import numpy as np
import numpy.random as npr

from src.data import Data
from src.model import LSTMAutoEncoder, ODEAutoEncoder
from src.train import Trainer
from src.utils import setup_folders
from src.visualize import Visualizer

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from datetime import datetime as dt

import logging
import logging.config


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      skip=1,
                      save_folder=None):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns:
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample:skip]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if save_folder:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        save_folder = f'{save_folder}/png/'

        fname = save_folder + 'ground_truth.png'
        plt.savefig(fname, dpi=500)
        logging.info('Saved ground truth spiral at {}'.format(fname))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        # t0_idx = npr.multinomial(
        #     1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        # t0_idx = np.argmax(t0_idx) + nsample
        t0_idx = int(ntotal * 0.3)

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_traj = (orig_traj - orig_traj.mean()) / orig_traj.std()
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_traj = samp_traj[::skip]
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--num-data', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--load-dir', type=str, default='')
    parser.add_argument('--obs-dim', type=int, default=2)
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=20)
    parser.add_argument('--rnn-hidden-dim', type=int, default=None)
    parser.add_argument('--lstm-hidden-dim', type=int, default=None)
    parser.add_argument('--lstm-layers', type=int, default=1)
    parser.add_argument('--n-total', type=int, default=200)
    parser.add_argument('--n-sample', type=int, default=120)
    parser.add_argument('--n-skip', type=int, default=1)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--baseline', action='store_true')

    args = parser.parse_args()

    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")
    MODEL_TYPE = 'toy'

    if args.dev:
        root = 'runs_dev'
    else:
        root = 'runs'

    save_folder = f'{root}/{MODEL_TYPE}_model_{RUN_TIME}/'
    setup_folders(save_folder)

    logging.config.fileConfig("logger.ini",
                              disable_existing_loggers=True,
                              defaults={'logfilename': f'{save_folder}logs_{RUN_TIME}.txt'})

    if args.dev:
        logging.info('Development run')
    else:
        logging.info('Production run')

    logging.info(f'Starting {MODEL_TYPE} experiment')
    logging.info(f'Passed arguments: {args}')

    # Model parameters
    latent_dim = args.latent_dim
    nhidden = args.hidden_dim
    rnn_nhidden = args.rnn_hidden_dim
    lstm_nhidden = args.lstm_hidden_dim
    lstm_layers = args.lstm_layers
    obs_dim = args.obs_dim

    # Data generation
    nspiral = args.num_data
    start = 0.
    stop = 6 * np.pi
    noise_std = .075
    a = 0.
    b = .3
    ntotal = args.n_total
    nsample = args.n_sample
    skip = args.n_skip

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'On device {device}')

    data = Data.from_func(generate_spiral2d,
                          device=device,
                          nspiral=nspiral,
                          ntotal=ntotal,
                          start=start,
                          stop=stop,
                          noise_std=noise_std,
                          a=a,
                          b=b,
                          skip=skip,
                          save_folder=save_folder)

    if not args.baseline:
        model = ODEAutoEncoder(latent_dim=latent_dim,
                               obs_dim=obs_dim,
                               rnn_hidden_dim=rnn_nhidden,
                               lstm_hidden_dim=lstm_nhidden,
                               hidden_dim=nhidden,
                               solver=args.solver,
                               device=device)
    else:
        model = LSTMAutoEncoder(latent_dim=latent_dim,
                                obs_dim=obs_dim,
                                hidden_dim=lstm_nhidden,
                                device=device)

    optimizer = optim.Adam(model.get_params(), lr=args.lr)
    logging.info(f"Optimizer: {optimizer}")

    visualizer = Visualizer(model, data, save_folder=save_folder)

    if args.load_dir != '':
        if not args.baseline:
            model_class = ODEAutoEncoder
        else:
            model_class = LSTMAutoEncoder

        trainer, version = Trainer.from_checkpoint(model_class,
                                                   args.load_dir,
                                                   args.epochs,
                                                   args.freq,
                                                   save_folder)
    else:
        trainer = Trainer(model=model,
                          optim=optimizer,
                          data=data,
                          visualizer=visualizer,
                          epochs=args.epochs,
                          freq=args.freq,
                          folder=save_folder)

        version = 0

    trainer.train(version)

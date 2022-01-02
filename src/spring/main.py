import argparse
import random

import matplotlib
import numpy as np
import numpy.random as npr

from src.data import Data
from src.model import ODEAutoEncoder, LSTMAutoEncoder
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

npr.seed(42)


def generate_spring2d(nsprings=100, ntotal=500, nsample=100, start=0., stop=1, a=1, b=1, c=0, skip=1, save_folder=None):
    # skip = 1 means no skipping

    orig_ts = np.linspace(start, stop, num=ntotal)
    # samp_ts = np.linspace(start, stop, num=nsample) # If we want samples across the full range
    samp_ts = orig_ts[:nsample:skip]  # If we want to grab the first samples
    # samp_ts = np.array(random.sample(set(orig_ts), nsample)) # If we want to randomly sample points

    # We generate multiple samples
    orig_trajs = []
    samp_trajs = []
    data_lab = []
    for _ in range(nsprings):
        # We sample for new spring solutions
        # a_sample = np.random.normal(a) # we remove noise on frequency
        a_sample = 1
        b_sample = np.random.normal(b)
        c_sample = np.random.normal(c)

        # We sample an example uniformly
        if args.example is None:
            example = np.random.randint(1, 4)
        else:
            example = random.choice(args.example)
        data_lab.append(example)
        # We generate the data for the original and sample springs
        if example == 1:
            orig_xs, orig_ys = orig_ts, b_sample * -1 / 2 * np.cos(2 * orig_ts) + 1 / 6 * np.sin(
                a_sample * 2 * orig_ts) + c_sample
        elif example == 2:
            # a_sample = np.random.normal(a, 0.25)
            a_sample = 1
            orig_xs, orig_ys = orig_ts, b_sample * 0.252096 * np.exp(0.2 * -5 / 2 * orig_ts) * np.cos(
                0.2 * a_sample * np.sqrt(119) / 0.5 * orig_ts - 3.2321) + c_sample
        elif example == 3:
            b_sample = np.random.normal(b, 0.25)
            orig_xs, orig_ys = orig_ts, b_sample * 0.1986 * np.exp(-0.8579 * orig_ts) + 0.001398 * np.exp(
                -29.1421 * orig_ts) + 1 / 45 * np.sin(a_sample * 5 * orig_ts) + c_sample
        else:
            raise ValueError(f"Example {example} does not exist")

        orig_ys = (orig_ys - orig_ys.mean()) / orig_ys.std()
        orig_xs = (orig_xs - orig_xs.mean()) / orig_xs.std()

        # We stack xs and ys as matrix (ntotal, 2)
        orig_traj = np.stack((orig_xs, orig_ys), axis=1)

        # Don't sample t0 very near the start or the end
        # t0_idx = npr.multinomial(.5, [1. / (ntotal - nsample)] * (ntotal - nsample))
        # t0_idx = np.argmax(t0_idx)
        t0_idx = int(ntotal * 0.05)

        orig_trajs.append(orig_traj)

        # Sample trajectories from original trajectory
        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj = samp_traj[::skip]
        samp_trajs.append(samp_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    if save_folder:
        plt.figure()
        plt.plot(orig_trajs[0, :, 0], orig_trajs[0, :, 1], label='spring')
        plt.legend()

        np.save(save_folder+f'data_lab_{np.unique(data_lab)}.npy',data_lab)
        np.save(save_folder+f'samp_trajs{np.unique(data_lab)}.npy',samp_trajs)

        save_folder = f'{save_folder}png/'

        fname = save_folder + 'ground_truth.png'
        plt.savefig(fname, dpi=500)
        logging.info('Saved ground truth spiral at {}'.format(fname))
        print("data labels:", data_lab)
        
    
    return orig_trajs, samp_trajs, orig_ts, samp_ts, data_lab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--num-data', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
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
    parser.add_argument('--example', nargs='+', type=int, default=None)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--baseline', action='store_true')

    args = parser.parse_args()

    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")
    MODEL_TYPE = 'spring'

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
    nsprings = args.num_data
    ntotal = args.n_total
    nsample = args.n_sample
    skip = args.n_skip
    start = 0.
    stop = 3 * np.pi
    a = 1
    b = 1
    c = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'On device {device}')

    data = Data.from_func(generate_spring2d,
                          device=device,
                          nsprings=nsprings,
                          ntotal=ntotal,
                          nsample=nsample,
                          start=start,
                          stop=stop,
                          a=a,
                          b=b,
                          c=c,
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

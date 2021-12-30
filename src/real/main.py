import argparse

import matplotlib
import numpy as np
import numpy.random as npr

from src.data import Data
from src.model import ODEAutoEncoder, LSTMBaseline
from src.train import Trainer
from src.utils import setup_folders
from src.visualize import Visualizer

matplotlib.use('agg')
import torch
import torch.optim as optim

from datetime import datetime as dt

import logging.config

npr.seed(42)


def load_data():
    data = np.load("src/real/power_curves.npy")
    n = data.shape[1]

    start = 0
    stop = 3 * np.pi

    start_idx = int(n * 0.05)
    stop_idx = int(n * 0.9)

    orig_ts = np.linspace(start, stop, num=n)
    samp_ts = orig_ts[start_idx:stop_idx]

    orig_trajs = []
    samp_trajs = []
    for trajs in data:
        orig_traj = (trajs - trajs.mean()) / trajs.std()
        samp_traj = orig_traj[start_idx:stop_idx]

        orig_trajs.append(orig_traj)
        samp_trajs.append(samp_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
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

    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--baseline', action='store_true')

    args = parser.parse_args()

    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")
    MODEL_TYPE = 'real'

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'On device {device}')

    data = Data.from_func(load_data,
                          device=device)

    if not args.baseline:
        model = ODEAutoEncoder(latent_dim=latent_dim,
                               obs_dim=obs_dim,
                               rnn_hidden_dim=rnn_nhidden,
                               lstm_hidden_dim=lstm_nhidden,
                               hidden_dim=nhidden,
                               solver=args.solver,
                               device=device)
    else:
        model = LSTMBaseline(input_dim=obs_dim,
                             hidden_dim=lstm_nhidden,
                             layer_dim=lstm_layers,
                             device=device)

    optimizer = optim.Adam(model.get_params(), lr=args.lr)
    logging.info(f"Optimizer: {optimizer}")

    visualizer = Visualizer(model, data, save_folder=save_folder)

    if args.load_dir != '':
        if not args.baseline:
            model_class = ODEAutoEncoder
        else:
            model_class = LSTMBaseline

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

import argparse
import numpy as np
import numpy.random as npr
import matplotlib

from src.model import *
from src.train import *
from src.data import *
from src.visualize import *
from src.utils import *

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from datetime import datetime as dt

import logging
import logging.config


## INPUTS FOR EXPERIMENT

# CHOOSE ONE
# load_dir = 'saved/monday/runs/real_model_1205_2329_34/ckpt/04_25_vfinal.pth'
# load_dir = 'saved/monday/runs/real_model_1205_2329_45/ckpt/05_28_vfinal.pth'
#load_dir = 'saved/monday/runs/real_model_1205_2330_16/ckpt/04_24_vfinal.pth'
#load_dir = 'runs/real_model_1207_1244_36/ckpt/12_59_vfinal.pth'
load_dir = 'experiment/models/real_LSTM.pth'

save_folder = 'experiment/real/'
setup_folders(save_folder)

# Logging
logging.config.fileConfig("logger.ini",
                        disable_existing_loggers=True,
                        defaults={'logfilename': f'{save_folder}real-logs.txt'})

logging.info('Running real experiment')

# We define args
parser = argparse.ArgumentParser()
parser.add_argument('--num-data', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--freq', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--load-dir', type=str, default=load_dir)
args = parser.parse_args()


if args.load_dir != '':
        trainer, version = Trainer.from_checkpoint(ODEAutoEncoder,
                                                   args.load_dir,
                                                args.epochs,
                                                args.freq,
                                                save_folder)

logging.info(trainer)

# We plot the single case for real
#trainer.visualizer.plot_reconstruction(fname = "reconstruction_.png", t_pos = 1/8*np.pi, t_neg = 1/8*np.pi, idx = 0)  # Set index

# We plot the grid for real
trainer.visualizer.plot_reconstruction_grid(fname = "reconstruction_grid_.png", t_pos = 0, t_neg = 0, size = 3)
#trainer.visualizer.visualize_final('final', t_pos = 0, t_neg = 0)

logging.warning('if you read this you have the big gay yes mam F')
import argparse
import numpy as np
import numpy.random as npr
import matplotlib

import os

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
load_dir = 'experiment/models/toy_RNN.pth'
# load_dir = 'runs/toy_model_1207_1605_34/ckpt/16_20_vfinal.pth'  # Baseline
#load_dir = 'runs/toy_model_1207_1605_34/ckpt/16_06_v3.pth'
#load_dir = 'runs/toy_model_1207_1605_34/ckpt/16_06_v3.pth'
save_folder = 'experiment/toy/'
setup_folders(save_folder)

class_name = ODEAutoEncoder
plot_single = "reconstruction_baseline.png"
plot_grid = 'reconstruction_grid_toy_RNN.png'

# Logging
logging.config.fileConfig("logger.ini",
                        disable_existing_loggers=True,
                        defaults={'logfilename': f'{save_folder}-logs.txt'})

logging.info('Running toy experiment')

# We define args
parser = argparse.ArgumentParser()
parser.add_argument('--num-data', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--freq', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--load-dir', type=str, default=load_dir)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.load_dir != '':
        trainer, version = Trainer.from_checkpoint(class_name,
                                                   args.load_dir,
                                                args.epochs,
                                                args.freq,
                                                save_folder)

# We plot the single case for spring
#trainer.visualizer.plot_reconstruction(fname = plot_single, t_pos = np.pi, t_neg = 1/2*np.pi, idx = 0)

trainer.visualizer.plot_reconstruction_grid(fname = plot_grid, t_pos = 2/3*np.pi, t_neg = 1/2*np.pi, size = 3)

print('if you read this you have the big gay yes mam F')
import argparse
import numpy as np
import numpy.random as npr
import matplotlib

# import os
# os.chdir(r"C:\Users\Garsdal\Desktop\9. Semester\02456 Deep Learning\Project\DL_project")

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

#load_dir = 'old/spring/example_1/model/12_03/ckpt_20_27_v1.pth'
#load_dir = 'experiment/spring/ckpt/02_20_vfinal.pth' # old retrained
# load_dir = 'DL_project/experiment/models/15_59_v13.pth'

save_folder = 'experiment/spring/'


# Logging
logging.config.fileConfig("logger.ini",
                        disable_existing_loggers=True,
                        defaults={'logfilename': f'{save_folder}-logs.txt'})

logging.info('Running toy experiment')

## INPUTS FOR EXPERIMENT
# load_dir = 'old/spring/example_1/model/12_04/ckpt_09_57_v98.pth'
#load_dir = 'experiment/models/spring_LSTM_mixed.pth'
# load_dir = 'runs/spring_model_1207_1621_16/ckpt/16_39_vfinal.pth'

# SPRINGS CKPTS
#load_dir = 'runs/spring_model_1227_2244_59/ckpt/03_57_vfinal.pth' #ex1
#load_dir = 'runs/spring_model_1227_2245_29/ckpt/02_32_vfinal.pth' #ex2
#load_dir = 'runs/spring_model_1227_2245_58/ckpt/02_26_vfinal.pth' #ex3
#load_dir = 'runs/spring_model_1227_2246_29/ckpt/01_05_vfinal.pth' #ex 1 2
#load_dir = 'runs/spring_model_1227_2247_00/ckpt/03_06_vfinal.pth' #ex 1 3
#load_dir = 'runs/spring_model_1227_2247_31/ckpt/02_29_vfinal.pth' #ex 2 3
load_dir = 'runs/spring_model_1227_2248_18/ckpt/01_56_vfinal.pth' #ex 1 2 3

# We define args
parser = argparse.ArgumentParser()
parser.add_argument('--num-data', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--freq', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--load-dir', type=str, default=load_dir)
args = parser.parse_args()

setup_folders(save_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.load_dir != '':
        trainer, version = Trainer.from_checkpoint(ODEAutoEncoder,
                                                   args.load_dir,
                                                   args.epochs,
                                                   args.freq,
                                                   save_folder)
          
 
# We plot the single case for spring
trainer.visualizer.plot_reconstruction(fname = "reconstruction_ex123.png", t_pos = 1/2*np.pi, t_neg = 1/6*np.pi, idx = 25)

# We plot the grid for spring
trainer.visualizer.plot_reconstruction_grid(fname = "reconstruction_grid_ex123", t_pos = 1/2*np.pi, t_neg = 1/6*np.pi, size = 3)

# We calculate RMSE for the test set
samp_trajs_test, samp_ts = trainer.data.get_test_data()
RMSE_test = trainer.visualizer.computeRMSE(samp_trajs_test, samp_ts)
print("RMSE test set:", RMSE_test)
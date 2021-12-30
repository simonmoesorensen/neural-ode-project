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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## INPUTS FOR RMSE
models_path = []
models_class = []
models_name = []
# SPRINGS:
# BASELINE
models_path.append('experiment/baseline_ckpt/spring/example1/15_56_vfinal.pth')  # ex1
models_path.append('experiment/baseline_ckpt/spring/example2/15_56_vfinal.pth')  # ex2
models_path.append('experiment/baseline_ckpt/spring/example3/12_03_vfinal.pth')  # ex3
models_path.append('experiment/baseline_ckpt/spring/example12/16_13_vfinal.pth')  # ex 1 2
models_path.append('experiment/baseline_ckpt/spring/example13/15_57_vfinal.pth')  # ex 1 3
models_path.append('experiment/baseline_ckpt/spring/example23/15_58_vfinal.pth')  # ex 2 3
models_path.append('experiment/baseline_ckpt/spring/example123/15_57_vfinal.pth')  # ex 1 2 3
for i in range(7):
    models_class.append(LSTMAutoEncoder)
models_name.append('Spring_1_Baseline_AE')
models_name.append('Spring_2_Baseline_AE')
models_name.append('Spring_3_Baseline_AE')
models_name.append('Spring_12_Baseline_AE')
models_name.append('Spring_13_Baseline_AE')
models_name.append('Spring_23_Baseline_AE')
models_name.append('Spring_123_Baseline_AE')

# NEURAL ODE
models_path.append('experiment/final_ckpt/spring/example1/03_57_vfinal.pth')  # ex1
models_path.append('experiment/final_ckpt/spring/example2/02_32_vfinal.pth')  # ex2
models_path.append('experiment/final_ckpt/spring/example3/02_26_vfinal.pth')  # ex3
models_path.append('experiment/final_ckpt/spring/example12/01_05_vfinal.pth')  # ex 1 2
models_path.append('experiment/final_ckpt/spring/example13/03_06_vfinal.pth')  # ex 1 3
models_path.append('experiment/final_ckpt/spring/example23/02_29_vfinal.pth')  # ex 2 3
models_path.append('experiment/final_ckpt/spring/example123/01_56_vfinal.pth')  # ex 1 2 3

for i in range(7):
    models_class.append(ODEAutoEncoder)
models_name.append('Spring_1_ODE_VAE')
models_name.append('Spring_2_ODE_VAE')
models_name.append('Spring_3_ODE_VAE')
models_name.append('Spring_12_ODE_VAE')
models_name.append('Spring_13_ODE_VAE')
models_name.append('Spring_23_ODE_VAE')
models_name.append('Spring_123_ODE_VAE')
# TOY
# BASELINE
models_path.append('experiment/baseline_ckpt/toy/14_07_vfinal.pth')
models_class.append(LSTMAutoEncoder)
models_name.append('Toy_Baseline_AE')
# NEURAL ODE
# models_path.append('experiment/final_ckpt/toy/toy_RNN.pth')
models_path.append('experiment/final_ckpt/toy/14_39_vfinal.pth')  # this guy can't extrapolate
models_class.append(ODEAutoEncoder)
models_name.append('Toy_ODE_VAE')
# REAL
# BASELINE
models_path.append('experiment/baseline_ckpt/real/14_07_vfinal.pth')
models_class.append(LSTMAutoEncoder)
models_name.append('Real_Baseline_AE')
# NEURAL ODE
# models_path.append('experiment/final_ckpt/real/real_LSTM.pth')
models_path.append('experiment/final_ckpt/real/14_41_vfinal.pth')
models_class.append(ODEAutoEncoder)
models_name.append('Real_ODE_VAE')

save_folder = 'experiment/RMSE_output/'
setup_folders(save_folder)

logging.config.fileConfig("logger.ini",
                          disable_existing_loggers=True,
                          defaults={'logfilename': f'logs.txt'})

RMSE_list = []
means_samp_traj = []
means_orig_traj = []
means_recon = []
for i, path in enumerate(models_path):
    logging.info(type(path))
    logging.info(models_class[i])
    trainer, version = Trainer.from_checkpoint(model_class=models_class[i],
                                               epochs=1,
                                               freq=1,
                                               path=path,
                                               folder=save_folder)

    # Set test set
    samp_trajs_test, samp_ts_test, orig_trajs_test = trainer.data.get_test_data()

    # We check the means
    means_samp_traj.append(torch.mean(samp_trajs_test))
    means_orig_traj.append(torch.mean(orig_trajs_test))

    # We calculate the RMSE for either VAE or AE
    if models_class[i] == ODEAutoEncoder:
        RMSE, pred_x_rmse = trainer.visualizer.computeRMSE_VAE(samp_trajs_test, samp_ts_test)
    elif models_class[i] == LSTMAutoEncoder:
        RMSE, pred_x_rmse = trainer.visualizer.computeRMSE_AE(samp_trajs_test, samp_ts_test)
    else:
        raise ValueError(f'Model class {models_class[i]} not supported')

    means_recon.append(torch.mean(pred_x_rmse))
    RMSE_list.append(float(RMSE))

    # We plot all the models | if baseline we have no extrapolation
    if models_class[i] == LSTMAutoEncoder:
        trainer.visualizer.plot_reconstruction(fname=f'reconstruction_{models_name[i]}.png', t_pos=0, t_neg=0, idx=0,
                                               test=True)
        trainer.visualizer.plot_reconstruction_grid(fname=f"reconstruction_grid_{models_name[i]}.png", t_pos=0, t_neg=0,
                                                    size=3, test=True)
    else:
        if models_name[i].split('_')[0] == 'Spring':
            trainer.visualizer.plot_reconstruction(fname=f'reconstruction_{models_name[i]}.png', t_pos=1 / 2 * np.pi,
                                                   t_neg=1 / 6 * np.pi, idx=25, test=True)
            trainer.visualizer.plot_reconstruction_grid(fname=f"reconstruction_grid_{models_name[i]}.png",
                                                        t_pos=1 / 2 * np.pi, t_neg=1 / 6 * np.pi, size=3, test=True)
        elif models_name[i].split('_')[0] == 'Toy':
            trainer.visualizer.plot_reconstruction(fname=f"reconstruction_{models_name[i]}.png", t_pos=1 / 16 * np.pi,
                                                   t_neg=1 / 16 * np.pi, idx=0, test=True)
            trainer.visualizer.plot_reconstruction_grid(fname=f"reconstruction_grid_{models_name[i]}.png",
                                                        t_pos=1 / 16 * np.pi, t_neg=1 / 16 * np.pi, size=3, test=True)
        elif models_name[i].split('_')[0] == 'Real':
            trainer.visualizer.plot_reconstruction(fname=f"reconstruction_{models_name[i]}.png", t_pos=1 / 8 * np.pi,
                                                   t_neg=1 / 8 * np.pi, idx=0, test=True)
            trainer.visualizer.plot_reconstruction_grid(fname=f"reconstruction_grid_{models_name[i]}.png", t_pos=0,
                                                        t_neg=0, size=3, test=True)

# We print the RMSE for each run
for i in range(len(models_path)):
    # print(models_name[i], "|", "RMSE:", RMSE_list[i], "samp trajs mean:", means_samp_traj[i], 'orig trajs mean:', means_orig_traj[i], "recon mean:", means_recon[i])
    logging.info(models_name[i], "|", "RMSE:", RMSE_list[i])

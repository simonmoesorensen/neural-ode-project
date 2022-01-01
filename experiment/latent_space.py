import matplotlib

from src.train import *
from src.utils import *
from src.visualize import *

matplotlib.use('agg')
import torch

import logging.config



## INPUTS FOR RMSE
models_path = []
models_class = []
models_name = []



# models_path.append('experiment/final_ckpt/spring/example1/03_57_vfinal.pth')  # ex1
# models_path.append('experiment/final_ckpt/spring/example123/01_56_vfinal.pth')
models_path.append('experiment/final_ckpt/spring/example23/02_29_vfinal.pth')  # ex 2 3

models_class.append(ODEAutoEncoder)

# models_name.append('Spring_1_ODE_VAE')
# models_name.append('Spring_123_ODE_VAE')
models_name.append('Spring_23_ODE_VAE')

save_folder = 'experiment/output_latent/'
setup_folders(save_folder)

logging.config.fileConfig("logger.ini",
                            disable_existing_loggers=True,
                            defaults={'logfilename': f'logs.txt'})
trainer, version = Trainer.from_checkpoint(model_class=models_class[0],
                                            epochs=1,
                                            freq=1,
                                            path=models_path[0],
                                            folder=save_folder)

trainer.visualizer.latent_vis(fname=f'latentspace_{models_name[0]}.png',
                                                       idx=0,
                                                       test=True)
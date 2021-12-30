import os

from src.data import Data
import torch

import numpy as np

import matplotlib.pyplot as plt

from src.model import ODEAutoEncoder, LSTMBaseline, LSTMAutoEncoder

plt.style.use('ggplot')

import logging
from datetime import datetime as dt


class Visualizer:
    def __init__(self, model, data: Data, save_folder: str):
        self.model = model
        self.data = data
        self.save_folder = save_folder + 'png/'
        self.device = model.device

    def visualize_step(self, version):
        now = dt.now().strftime('%H_%M')
        fname = f'v{version}_{now}.png'

        self.plot_reconstruction(fname)

    def visualize_final(self, version, t_pos = np.pi, t_neg = np.pi):
        now = dt.now().strftime('%H_%M')
        fname = f'{version}_{now}'

        self.plot_reconstruction(fname + '_reconstruction.png', t_pos = t_pos, t_neg = t_neg)
        self.plot_loss_history(fname + '_loss_history.png')
        self.plot_reconstruction_grid(fname + '_reconstruction_grid.png', t_pos = t_pos, t_neg = t_neg)
        self.plot_original_grid(fname + '_original_grid.png')

    def plot_reconstruction(self, fname, t_pos=np.pi, t_neg=np.pi, idx=0, test = False):
        # We unwrap the trajectories from the data object
        orig_trajs, samp_trajs, _, samp_ts = self.data.get_all_data()

        # We make sure that we plot for the test sample trajectories if test = True
        if test:
            samp_trajs, samp_ts, orig_trajs  = self.data.get_test_data()

        with torch.no_grad():
            if isinstance(self.model, ODEAutoEncoder):
                # We forward pass to extract pred_x and z0 (we will only use z0)
                qz0_mean, qz0_logvar, epsilon = self.model.encode(samp_trajs)

                # Sample z0 (vector) from q(z0)
                z0 = self.model.sample_z0(epsilon, qz0_logvar, qz0_mean)

                # We generate new linspaces for extrapolation and negative extrapolation | We use the decode function to extract pred_x
                ts_rmse = torch.from_numpy(np.linspace(0., torch.max(samp_ts), num=len(samp_ts))).to(self.device)
                ts_pos = torch.from_numpy(np.linspace(0, torch.max(samp_ts) + t_pos, num=int(len(samp_ts)))).to(self.device)
                
                pred_x_pos = self.model.decode(z0, ts_pos)
                pred_x_rmse = self.model.decode(z0, ts_rmse)

                # Define extrapolation            
                val = int((t_pos / (torch.max(samp_ts) + t_pos))*len(samp_ts))
                pred_x_rec = pred_x_pos[:,:(len(samp_ts)-val),:]
                pred_x_pos = pred_x_pos[:,(len(samp_ts)-val-1):,:]

                rmse_loss = self.RMSELoss(pred_x_rmse, samp_trajs)
                logging.info(f'RMSE: {rmse_loss}')
                
                pred_x_pos = pred_x_pos.cpu().detach().numpy()
                pred_x_rec = pred_x_rec.cpu().detach().numpy()
                
                # We plot only the first trajectory
                orig_trajs = orig_trajs.cpu().detach()
                samp_trajs = samp_trajs.cpu().detach()

                if (t_neg > 0):
                    ts_neg = torch.from_numpy(np.linspace(-t_neg, 0., num=int(len(samp_ts)/8))[::-1].copy()).to(self.device)
                    pred_x_neg = torch.flip(self.model.decode(z0, ts_neg), dims=[1]).cpu().detach().numpy()

                    plt.figure()
                    plt.plot(orig_trajs[idx, :, 0], orig_trajs[idx, :, 1], 'g', label='True trajectory', linewidth = 2, zorder=1)
                    plt.plot(pred_x_rec[idx, :, 0], pred_x_rec[idx, :, 1], '-bo', color = 'r', markersize = 3, label='Reconstruction', zorder=3)
                    plt.plot(pred_x_pos[idx, :, 0], pred_x_pos[idx, :, 1], '-bo', color = 'c', markersize = 3, label='Learned trajectory (t>0)', zorder=2)
                    plt.plot(pred_x_neg[idx, :, 0], pred_x_neg[idx, :, 1], '-bo', color = 'c', markersize = 3, label='Learned trajectory (t<0)', zorder=2)
                    plt.scatter(samp_trajs[idx, :, 0], samp_trajs[idx, :, 1], color='b', label='Sampled data', s=10, zorder=2)
                    plt.legend()
                else:
                    plt.figure()
                    plt.plot(orig_trajs[idx, :, 0], orig_trajs[idx, :, 1], 'g', label='True trajectory', zorder=1)
                    #plt.plot(pred_x_rec[idx, :, 0], pred_x_rec[idx, :, 1], '-bo', color = 'r', markersize = 3, label='Reconstruction', zorder=3)
                    plt.plot(pred_x_pos[idx, :, 0], pred_x_pos[idx, :, 1], '-bo', color = 'c', markersize = 3, label='Learned trajectory (t>0)', zorder=3)
                    plt.scatter(samp_trajs[idx, :, 0], samp_trajs[idx, :, 1], color='b', label='Sampled data', s=3, zorder=2)
                    plt.legend()
                    
            elif isinstance(self.model, LSTMAutoEncoder):
                pred_x = self.model.forward(samp_trajs)
                
                plt.figure()
                plt.plot(orig_trajs[idx, :, 0], orig_trajs[idx, :, 1], 'g', label='true trajectory', zorder=1)
                plt.plot(pred_x[idx, :, 0], pred_x[idx, :, 1], 'r', label='learned trajectory (t>0)', zorder=3)
                plt.scatter(samp_trajs[idx, :, 0], samp_trajs[idx, :, 1], color='b', label='sampled data', s=3, zorder=2)
                plt.legend()
                
            logging.info('Saved reconstruction at {}'.format(self.save_folder + fname))
            plt.savefig(self.save_folder + fname, dpi=500)

    def plot_reconstruction_grid(self, fname, t_pos=np.pi, t_neg=np.pi, size = 5, test = False):
        # We unwrap the trajectories from the data object
        orig_trajs, samp_trajs, _, samp_ts = self.data.get_all_data()

        # We make sure that we plot for the test sample trajectories if test = True
        if test:
            samp_trajs, samp_ts, orig_trajs  = self.data.get_test_data()

        with torch.no_grad():
            if isinstance(self.model, ODEAutoEncoder):
                # We forward pass to extract pred_x and z0 (we will only use z0)
                qz0_mean, qz0_logvar, epsilon = self.model.encode(samp_trajs)

                # Sample z0 (vector) from q(z0)
                z0 = self.model.sample_z0(epsilon, qz0_logvar, qz0_mean)

                # We generate new linspaces for extrapolation and negative extrapolation | We use the decode function to extract pred_x
                ts_pos = torch.from_numpy(np.linspace(0., torch.max(samp_ts) + t_pos, num=len(samp_ts))).to(self.device)
                pred_x_pos = self.model.decode(z0, ts_pos).cpu().detach().numpy()

                if t_neg > 0:
                    ts_neg = torch.from_numpy(np.linspace(-t_neg, 0., num=int(len(samp_ts)/8))[::-1].copy()).to(self.device)
                    pred_x_neg = torch.flip(self.model.decode(z0, ts_neg), dims=[1]).cpu().detach().numpy()

                # Define extrapolation            
                val = int((t_pos / (torch.max(samp_ts) + t_pos))*len(samp_ts))
                pred_x_rec = pred_x_pos[:,:(len(samp_ts)-val),:]
                pred_x_pos = pred_x_pos[:,(len(samp_ts)-val-1):,:]

                orig_trajs = orig_trajs.cpu().detach()
                samp_trajs = samp_trajs.cpu().detach()

                plt.figure(figsize=(15, 15))
                for i in range(size**2):
                    # We scale all y values to be 0:1
                    min_traj_y = np.min(orig_trajs.numpy()[i,:,1])
                    max_traj_y = np.max(orig_trajs.numpy()[i,:,1])
                    pred_x_rec_plt_y = (pred_x_rec[i,:,1]-min_traj_y)/(max_traj_y-min_traj_y)
                    pred_x_pos_plt_y = (pred_x_pos[i,:,1]-min_traj_y)/(max_traj_y-min_traj_y)
                    orig_trajs_plt_y = (orig_trajs[i,:,1]-min_traj_y)/(max_traj_y-min_traj_y)
                    
                    # We scale all x values to be 0:1
                    min_traj_x = np.min(orig_trajs.numpy()[i,:,0])
                    max_traj_x = np.max(orig_trajs.numpy()[i,:,0])
                    pred_x_rec_plt_x = (pred_x_rec[i,:,0]-min_traj_x)/(max_traj_x-min_traj_x)
                    pred_x_pos_plt_x = (pred_x_pos[i,:,0]-min_traj_x)/(max_traj_x-min_traj_x)
                    orig_trajs_plt_x = (orig_trajs[i,:,0]-min_traj_x)/(max_traj_x-min_traj_x)
                    
                    plt.subplot(size, size, i + 1)
                    plt.plot(pred_x_rec_plt_x, pred_x_rec_plt_y, '-bo', color='r', markersize = 1, label = 'Reconstruction', zorder = 3)
                    plt.plot(orig_trajs_plt_x, orig_trajs_plt_y, color='g', linewidth=1, label = 'True trajectory', markersize = 1, zorder = 1)
                    if t_pos > 0:
                        plt.plot(pred_x_pos_plt_x, pred_x_pos_plt_y, '-bo', color = 'c', markersize = 3, label='Learned trajectory (t>0)', zorder=2)
                    if t_neg > 0:
                        pred_x_neg_plt_y = (pred_x_neg[i,:,1]-min_traj_y)/(max_traj_y-min_traj_y)
                        pred_x_neg_plt_x = (pred_x_neg[i,:,0]-min_traj_x)/(max_traj_x-min_traj_x)
                        plt.plot(pred_x_neg_plt_x, pred_x_neg_plt_y, '-bo', color = 'c', markersize = 3, label='Learned trajectory (t<0)', zorder=2)
                        
                plt.legend(['Reconstruction', 'True trajectory', 'Learned trajectory (t>0)', 'Learned trajectory (t<0)'])
            
            elif isinstance(self.model, LSTMAutoEncoder):
                pred_x = self.model.forward(samp_trajs)
                
                plt.figure(figsize=(15, 15))
                
                for i in range(size**2):
                    plt.subplot(size, size, i + 1)
                    plt.plot(orig_trajs[i, :, 0], orig_trajs[i, :, 1], 'g', label='true trajectory', zorder=1)
                    plt.plot(pred_x[i, :, 0], pred_x[i, :, 1], 'r', label='learned trajectory (t>0)', zorder=3)
                    plt.scatter(samp_trajs[i, :, 0], samp_trajs[i, :, 1], color='b', label='sampled data', s=3, zorder=2)
                plt.legend()

            logging.info('Saved reconstruction grid plot at {}'.format(self.save_folder + fname))
            plt.savefig(self.save_folder + fname, dpi=500)

    def plot_original_grid(self, fname):
        # We unwrap the trajectories from the data object
        orig_trajs, _, _, _ = self.data.get_all_data()

        orig_trajs = orig_trajs.cpu().detach()

        plt.figure(figsize=(15, 15))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.scatter(x=orig_trajs[i, :, 0], y=orig_trajs[i, :, 1], color='b')

        plt.legend(['orig_trajs'])

        logging.info('Saved original grid plot at {}'.format(self.save_folder + fname))
        plt.savefig(self.save_folder + fname, dpi=500)

    def plot_loss_history(self, fname):
        plt.figure(figsize=(15, 15))
        plt.plot(self.model.train_loss, color='b')
        plt.plot(self.model.val_loss, color='r')

        plt.legend(['train ELBO', 'validation ELBO'])

        logging.info('Saved loss plot at {}'.format(self.save_folder + fname))
        plt.savefig(self.save_folder + fname, dpi=500)

    def plot_latent_space(self, fname):
        # We unwrap the trajectories from the data object
        orig_trajs, samp_trajs, orig_ts, samp_ts = self.data.get_all_data()

        # We forward pass to extract pred_x and z0
        pred_x, z0 = self.model.forward(samp_trajs, samp_ts, return_z0=True)

        # We use the decode function to extract pred_z
        pred_z = self.model.decode(z0, samp_ts, return_z=True)

        # Dunno if this is necessary
        orig_trajs = orig_trajs.cpu()
        samp_trajs = samp_trajs.cpu()
        z0 = z0.cpu().detach().numpy()
        pred_z = pred_z.cpu().detach().numpy()
        pred_x = pred_x.cpu().detach().numpy()

        # We create the plot
        plt.figure(figsize=(15, 15))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.scatter(x=pred_z[i, :, 0], y=pred_z[i, :, 1], color='r')
        plt.legend(['pred_z,'])
        plt.savefig(fname, dpi=500)
        plt.show()

    def plot_extrapolation(self):
        pass
    
    def RMSELoss(self, yhat, y):
        assert type(yhat)==torch.Tensor
        assert type(y)==torch.Tensor
        return torch.sqrt(torch.mean((yhat-y)**2))

    def computeRMSE_VAE(self, samp_trajs, samp_ts):
        pred_x_rmse = self.model.forward(samp_trajs, samp_ts)
        rmse_loss = self.RMSELoss(pred_x_rmse, samp_trajs)

        return(rmse_loss.cpu().detach().numpy(), pred_x_rmse)

    def computeRMSE_AE(self, samp_trajs, samp_ts):
        pred_x_rmse = self.model.forward(samp_trajs)
        rmse_loss = self.RMSELoss(pred_x_rmse, samp_trajs)

        return(rmse_loss.cpu().detach().numpy(), pred_x_rmse)

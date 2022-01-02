import logging
import os
import time
from datetime import datetime as dt

import numpy as np
import torch
from torch import optim

from src.data import Data
from src.model import ODEAutoEncoder, TrainerModel
from src.visualize import Visualizer


class Trainer:
    def __init__(self, model, optim, data, epochs, freq, folder=None, visualizer=None):
        assert isinstance(model, TrainerModel)
        self.model: TrainerModel = model
        self.optimizer = optim
        self.data = data
        self.epochs = epochs
        self.freq = freq

        if folder is None:
            self.folder = 'runs/model_' + dt.now().strftime("%m%d_%H%M")
        else:
            self.folder = folder

        if visualizer is None:
            self.visualizer = None
        else:
            self.visualizer = visualizer

        logging.info(f'Instantiated trainer object with model {model}\n'
                     f'and saving in folder {self.folder}\n'
                     f'over {epochs} epochs logging every {freq} epoch')

    @classmethod
    def from_checkpoint(cls, model_class, path, epochs, freq, folder):
        obj = torch.load(path)

        model = model_class.from_checkpoint(path)
        optimizer = optim.Adam(model.get_params(), lr=0)
        optimizer.load_state_dict(obj['optimizer_state_dict'])

        data = Data.from_dict(obj['data'])

        # if isinstance(model, autoen)
        # data_lab = Data.from_dict(obj['data_lab'])

        visualizer = Visualizer(model, data, folder)

        trainer = cls(model, optimizer, data, epochs, freq,
                      folder=folder, visualizer=visualizer)

        logging.info(f"Loaded model from {path}")

        # Get the version from next index where v is 'runs/model_1204_1635/ckpt/16_35_v3.pth'
        version = 0
        return trainer, version

    def train(self, version=0):
        logging.info('Starting training')

        try:
            for epoch in range(self.epochs):
                start = time.time()

                self.train_step(*self.data.get_train_data())
                self.validation_step(*self.data.get_val_data())

                end = time.time()
                self.model.epoch_time.append(end - start)

                if epoch % self.freq == 0:
                    if isinstance(self.model, ODEAutoEncoder):
                        logging.info(f'Current number of forward passes: {self.model.nfe_list[-1]}')

                    logging.info('Epoch: {}, train elbo: {:.4f}, validation elbo: {:.4f}, mean time per epoch: {:.4f}'
                                 .format(epoch,
                                         self.model.train_loss[-1],
                                         self.model.val_loss[-1],
                                         np.mean(self.model.epoch_time)))
                    self.save_model(version)
                    self.visualize_step(version)
                    version += 1

            logging.info(f'Training finished after {epoch} epochs')
            self.save_model('final')
            self.visualize_final()

        except KeyboardInterrupt:
            logging.info('Stopped training due to interruption')
            self.save_model(version)
            self.visualize_final('interrupted')

    def visualize_step(self, version):
        if self.visualizer:
            self.visualizer.visualize_step(version)

    def visualize_final(self, version='final'):
        if self.visualizer:
            self.visualizer.visualize_final(version)

    def train_step(self, x, t):
        self.model.train()

        self.optimizer.zero_grad()

        # Perform train step
        elbo = self.model.train_step(x, t)

        # Backwards pass and update optimizer and train loss
        elbo.backward()
        self.optimizer.step()
        self.model.train_loss.append(-elbo.item())

    def validation_step(self, x, t):
        self.model.eval()

        with torch.no_grad():
            elbo = self.model.train_step(x, t)
            self.model.val_loss.append(-elbo.item())

    def save_model(self, version):
        folder = os.path.join(self.folder, 'ckpt')

        now = dt.now().strftime('%H_%M')
        ckpt_path = os.path.join(folder, f'{now}_v{version}.pth')

        save_dict = {
            'model_args': self.model.get_args(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'data': self.data.get_dict(),
            'train_loss': self.model.train_loss,
            'val_loss': self.model.val_loss
        }
        save_dict.update(self.model.get_state_dicts())
        torch.save(save_dict, ckpt_path)

        logging.info(f"Saved model at {ckpt_path}")

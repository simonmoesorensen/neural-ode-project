import logging

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


###########################################
# UTILITY FUNCTIONS
###########################################

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


###########################################
# ABSTRACT TRAINER MODEL
###########################################

class TrainerModel(nn.Module):
    train_loss = []
    val_loss = []
    epoch_time = []

    def train_step(self, x, t):
        raise NotImplementedError()

    @classmethod
    def from_checkpoint(cls, path):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def get_args(self):
        raise NotImplementedError()

    def get_state_dicts(self):
        raise NotImplementedError()


###########################################
# ODE IMPLEMENTATION
###########################################

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)

        if out.requires_grad:
            self.nfe += 1

        return out


###########################################
# ENCODER IMPLEMENTATION
###########################################

class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25):
        super(RecognitionRNN, self).__init__()
        logging.info('Setting up RNN')
        self.nhidden = nhidden
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def forward_sequence(self, x, device):
        h = self.init_hidden(batch=x.shape[0]).to(device)

        # Forward pass over all inputs for each time step
        # in reverse so we get z0 instead of z_T
        for t in reversed(range(x.size(1))):
            obs = x[:, t, :]
            out, h = self.forward(obs, h)

        return out

    def init_hidden(self, batch):
        return torch.zeros(batch, self.nhidden)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, nhidden, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.nhidden = nhidden
        self.lstm = nn.LSTMCell(input_size, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h, c):
        hn, cn = self.lstm(x, (h, c))
        out = self.h2o(hn)
        return out, hn, cn

    def forward_sequence(self, x, device):
        h, c = self.init_hidden(batch=x.shape[0])
        hn = h.to(device)
        cn = c.to(device)
        cn = cn[0, :, :]
        hn = hn[0, :, :]
        # Forward pass over all inputs for each time step
        # in reverse so we get z0 instead of z_T
        for t in reversed(range(x.size(1))):
            obs = x[:, t, :]
            out, hn, cn = self.forward(obs, hn, cn)

        return out

    def init_hidden(self, batch):
        h = torch.zeros(1, batch, self.nhidden)
        c = torch.zeros(1, batch, self.nhidden)
        return [h, c]


class LSTMBaseline(nn.Module):
    def __init__(self, input_size, nhidden, latent_dim):
        super(LSTMBaseline, self).__init__()
        self.nhidden = nhidden
        self.latent_dim = latent_dim
        self.lstm = nn.LSTMCell(input_size, nhidden)
        self.lstm2 = nn.LSTMCell(nhidden, latent_dim)
        # self.activation = nn.ReLU(inplace=True)

    def forward(self, x, h1, c1, h2, c2):
        # Map inputs to hidden state from x -> h1
        # Map hidden state to latent dimension h1 -> h2
        hn1, cn1 = self.lstm(x, (h1, c1))
        hn2, cn2 = self.lstm2(hn1, (h2, c2))
        return hn1, cn1, hn2, cn2

    def forward_sequence(self, x, device):
        h1, c1, h2, c2 = self.init_hidden(batch=x.shape[0])
        h1 = h1.to(device)
        c1 = c1.to(device)

        output = []
        # Forward pass over all inputs for each time step
        for t in range(x.size(1)):
            obs = x[:, t, :]
            h1, c1, h2, c2 = self.forward(obs, h1, c1, h2, c2)
            output.append(h2)

        output = torch.stack(output, dim=1)
        return output

    def init_hidden(self, batch):
        h = torch.zeros(1, batch, self.nhidden)
        c = torch.zeros(1, batch, self.nhidden)
        c1 = c[0, :, :]
        h1 = h[0, :, :]

        h = torch.zeros(1, batch, self.latent_dim)
        c = torch.zeros(1, batch, self.latent_dim)
        c2 = c[0, :, :]
        h2 = h[0, :, :]
        return h1, c1, h2, c2


###########################################
# DECODER IMPLEMENTATION
###########################################

class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.relu(self.fc1(z))
        out = self.fc2(out)
        return out


###########################################
# AUTOENCODER IMPLEMENTATION
###########################################

class LSTMAutoEncoder(TrainerModel):
    def __init__(self, latent_dim, obs_dim, hidden_dim, device=None):
        super(TrainerModel, self).__init__()
        if not device:
            self.device = 'cpu'
        else:
            self.device = device

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        # Model
        self.encoder = LSTMBaseline(input_size=obs_dim,
                                    nhidden=hidden_dim,
                                    latent_dim=latent_dim)

        self.decoder = LSTMBaseline(input_size=latent_dim,
                                    nhidden=hidden_dim,
                                    latent_dim=latent_dim)

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Loss
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encode(x)
        pred_x = self.decode(z)
        return pred_x

    def decode(self, z):
        x = self.decoder.forward_sequence(z, self.device)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def encode(self, x):
        z = self.encoder.forward_sequence(x, self.device)
        return z

    def train_step(self, x, t):
        pred_x = self.forward(x)

        # RMSE loss
        loss = torch.sqrt(self.criterion(x, pred_x))
        return loss

    @classmethod
    def from_checkpoint(cls, path):
        # Usage: LSTMAutoEncoder.from_checkpoint(path)
        obj = torch.load(path)
        model = cls(**obj['model_args'])
        model.encoder.load_state_dict(obj['encoder_state_dict'])
        model.decoder.load_state_dict(obj['decoder_state_dict'])
        model.fc1.load_state_dict(obj['fc1'])
        model.fc2.load_state_dict(obj['fc2'])

        model.train_loss = obj['train_loss']
        model.val_loss = obj['val_loss']

        return model

    def get_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
               list(self.fc1.parameters()) + \
               list(self.fc2.parameters())

    def get_args(self):
        return {'latent_dim': self.latent_dim,
                'obs_dim': self.obs_dim,
                'hidden_dim': self.hidden_dim,
                'device': self.device}

    def get_state_dicts(self):
        return {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'fc1': self.fc1.state_dict(),
            'fc2': self.fc2.state_dict(),
        }


class ODEAutoEncoder(TrainerModel):
    def __init__(self, latent_dim, obs_dim, hidden_dim, rnn_hidden_dim=None, lstm_hidden_dim=None, device=None,
                 solver='rk4'):
        super(TrainerModel, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_dim = hidden_dim
        self.solver = solver

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        if rnn_hidden_dim and not lstm_hidden_dim:
            self.encoder = RecognitionRNN(latent_dim, obs_dim, rnn_hidden_dim).to(device)
        elif lstm_hidden_dim and not rnn_hidden_dim:
            self.encoder = LSTMEncoder(obs_dim, lstm_hidden_dim, latent_dim).to(device)
        else:
            raise ValueError('Please satisfy rnn_hidden_dim xor lstm_hidden_dim')

        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim).to(device)

        self.odefunc = LatentODEfunc(latent_dim, hidden_dim).to(device)
        self.nfe_list = []  # Length of nfe_list is number of epochs

    def forward(self, x, t, return_z0=False):
        # Encode
        qz0_mean, qz0_logvar, epsilon = self.encode(x)

        # Sample z0 (vector) from q(z0)
        z0 = self.sample_z0(epsilon, qz0_logvar, qz0_mean)

        # Decode
        pred_x = self.decode(z0, t)

        if return_z0:
            return pred_x, z0
        return pred_x

    def sample_z0(self, epsilon, qz0_logvar, qz0_mean):
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0

    def encode(self, x):
        out = self.encoder.forward_sequence(x, self.device)

        # Get the q(z0) distribution (as a vector)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.device)

        return qz0_mean, qz0_logvar, epsilon

    def decode(self, z0, t, return_z=False):
        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.odefunc, z0, t, method=self.solver).permute(1, 0, 2)
        pred_x = self.decoder(pred_z)

        if return_z:
            return pred_x, pred_z

        return pred_x

    def train_step(self, x, t):
        # Encode
        qz0_mean, qz0_logvar, epsilon = self.encode(x)

        # Sample z0 (vector) from q(z0)
        z0 = self.sample_z0(epsilon, qz0_logvar, qz0_mean)

        # Decode
        pred_x = self.decode(z0, t)

        noise_std_ = torch.zeros(pred_x.size()).to(self.device) + .2
        noise_logvar = 2. * torch.log(noise_std_).to(self.device)

        logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(self.device)

        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)

        elbo = torch.mean(-logpx + analytic_kl, dim=0)

        if elbo.requires_grad:
            self.save_nfe()

        return elbo

    def save_nfe(self):
        # save number of current forward passes
        self.nfe_list.append(self.odefunc.nfe)
        # Reset for next epoch
        self.odefunc.nfe = 0
        return

    @classmethod
    def from_checkpoint(cls, path):
        obj = torch.load(path)
        model = cls(**obj['model_args'])
        model.odefunc.load_state_dict(obj['odefunc_state_dict'])
        model.encoder.load_state_dict(obj['encoder_state_dict'])
        model.decoder.load_state_dict(obj['decoder_state_dict'])

        if 'nfe_list' in obj:
            model.nfe_list = obj['nfe_list']

        model.train_loss = obj['train_loss']
        model.val_loss = obj['val_loss']

        return model

    def get_params(self):
        return list(self.odefunc.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters())

    def get_args(self):
        return {'latent_dim': self.latent_dim,
                'obs_dim': self.obs_dim,
                'hidden_dim': self.hidden_dim,
                'rnn_hidden_dim': self.rnn_hidden_dim,
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'device': self.device,
                'solver': self.solver}

    def get_state_dicts(self):
        return {'odefunc_state_dict': self.odefunc.state_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'nfe_list': self.nfe_list}

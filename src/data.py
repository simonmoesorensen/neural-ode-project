import logging

import numpy as np
import torch


class Data:
    def __init__(self, orig_trajs, samp_trajs, orig_ts, samp_ts, labels=None, device=None):
        if device is None:
            device = 'cpu'

        try:
            orig_trajs = torch.from_numpy(orig_trajs).float()
            samp_trajs = torch.from_numpy(samp_trajs).float()
            samp_ts = torch.from_numpy(samp_ts).float()
            orig_ts = torch.from_numpy(orig_ts).float()

        except Exception:
            logging.warning('Inputs cannot be converted to torch (already a torch obj?)\n'
                            f'Types: {type(orig_trajs), type(samp_trajs), type(orig_ts), type(samp_ts)} ')

        self.samp_ts = samp_ts.to(device)
        self.samp_trajs = samp_trajs.to(device)
        self.orig_ts = orig_ts.to(device)
        self.orig_trajs = orig_trajs.to(device)
        self.labels = labels

        self.split(orig_trajs, samp_trajs, labels)

    @classmethod
    def from_func(cls, func, device, **args):
        return cls(*func(**args), device=device)

    def split(self, orig_trajs, samp_trajs, labels, train_split=0.6, val_split=0.2):
        # We split the data across the spring dimension [nr. springs, nr. samples, values]
        train_int = np.int(train_split * orig_trajs.shape[0])  # X% of the data length for training
        val_int = np.int((train_split + val_split) * orig_trajs.shape[0])  # X% more for validation

        self.orig_trajs_train, self.orig_trajs_val, self.orig_trajs_test = (orig_trajs[:train_int, :, :],
                                                                            orig_trajs[train_int:val_int, :, :],
                                                                            orig_trajs[val_int:, :, :])

        self.samp_trajs_train, self.samp_trajs_val, self.samp_trajs_test = (samp_trajs[:train_int, :, :],
                                                                            samp_trajs[train_int:val_int, :, :],
                                                                            samp_trajs[val_int:, :, :])
        if labels:
            self.labels_train, self.labels_val, self.labels_test = (labels[:train_int],
                                                                    labels[train_int:val_int],
                                                                    labels[val_int:])

    def get_train_data(self):
        return self.samp_trajs_train, self.samp_ts

    def get_val_data(self):
        return self.samp_trajs_val, self.samp_ts

    def get_test_data(self):
        return self.samp_trajs_test, self.samp_ts, self.orig_trajs_test, self.orig_ts

    def get_all_data(self):
        return self.orig_trajs, self.samp_trajs, self.orig_ts, self.samp_ts

    def get_train_labels(self):
        return self.labels_train

    def get_val_labels(self):
        return self.labels_val

    def get_test_labels(self):
        return self.labels_test

    def get_all_labels(self):
        return self.labels

    @classmethod
    def from_dict(cls, dict, device=None):
        labels = None
        if 'labels' in dict:
            labels = dict['labels']
            logging.info("Loading data labels")

        return cls(
            dict['orig_trajs'],
            dict['samp_trajs'],
            dict['orig_ts'],
            dict['samp_ts'],
            labels,
            device
        )

    def get_dict(self):
        return {
            'orig_trajs': self.orig_trajs,
            'samp_trajs': self.samp_trajs,
            'orig_ts': self.orig_ts,
            'samp_ts': self.samp_ts,
            'labels': self.labels,
        }

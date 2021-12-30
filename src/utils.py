import logging
import os


def setup_folders(folder):
    if not os.path.exists(folder + 'png'):
        logging.info(f"{folder + 'png'} does not exist... creating")
        os.makedirs(folder + 'png')

    if not os.path.exists(folder + 'ckpt'):
        logging.info(f"{folder + 'ckpt'} does not exist... creating")

        os.makedirs(folder + 'ckpt')

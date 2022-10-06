from typing import Optional

import torch
from torch import nn
from tqdm import trange
from time import sleep

import os

import logging

logger = logging.getLogger(__name__)


def pretty_stream(tqdm_bar: trange):
    tqdm_bar.close()
    sleep(1)


def safe_dir(path: str,
             with_filename: bool = False):
    dir_path = os.path.dirname(path) if with_filename else path  # dirname: solve the path without filename
    if not os.path.exists(dir_path):
        logger.info(f"Dir {dir_path} is not exist, creating the new folder")
        os.makedirs(dir_path)
    return os.path.abspath(path)


def save_model(epoch: int,
               checkpoint_root: str,
               model,
               optimizer: torch.optim.Optimizer,
               loss: Optional[float],
               lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], ):
    checkpoint_root = safe_dir(checkpoint_root)
    checkpoint_path = f'{checkpoint_root}/{model.__class__.__name__}_ckpt_ep{epoch:04d}'

    logger.info('Save model at %s!', checkpoint_path)

    save_dict = {
        'epochs': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if lr_scheduler is not None:
        save_dict["lr_scheduler"] = lr_scheduler
    if loss is not None:
        save_dict["loss"] = loss

    torch.save(save_dict, checkpoint_path)

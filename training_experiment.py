# Author: 310551076 Oscar Chen
# Title: Video streaming and Tracking
# Date: 2022/10/10
# Email: oscarchen.cs10@nycu.edu.tw

import logging
import pprint

import numpy as np
import wandb

import torch
import torchvision
from sklearn.model_selection import KFold, StratifiedKFold
from torch.backends import cudnn
from torch.utils.data import DataLoader
# import time
# import copy
from torch.utils.tensorboard import SummaryWriter

from ImageLoader import ImageLoader
# import pickle
from evaluate.evaluate_model import evaluate_model

from experiment.config import ExperimentConfig

# load Hyperparameter setting
from models.model_architecture import VGGNet_19
from models.model_prepare import initialize_model
from training.train_model import train_model
from util.logger import setup_logging
import pandas as pd

wandb.tensorboard.patch(pytorch=True)
config = ExperimentConfig()
logger = logging.getLogger(__name__)

# for reproducibility
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)
cudnn.deterministic = True

# init logging
setup_logging(config.log_data_dir, "DEBUG")
logger.info("Experiment Config\n%s", pprint.pformat(config.to_dict()))

logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"Torchvision Version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device".format(device))


# Init wandb
wandb.init(
    entity=config.wandb_entity,
    project=config.wandb_project,
    name=f"{config.cur_time}_VS",
    group=f"{config.cur_time}_VS_Init",
    dir=config.wandb_dir,
    config=config.to_dict(),
)
# tensorboard
writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
writer = SummaryWriter(log_dir=writer_dir)

model_ft = VGGNet_19()
model_ft = model_ft.to(device)
wandb.watch(model_ft)
logger.info(model_ft)

## DataLoader
print("Initializing Datasets and Dataloaders...")
train_data = ImageLoader(config.db_path, 'train', config)
val_data = ImageLoader(config.db_path, "val", config)
trainLoader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                         num_workers=config.dataloader_num_worker, pin_memory=True)
valLoader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True,
                       num_workers=config.dataloader_num_worker, pin_memory=True)

logger.info(f"train_data size: {len(trainLoader.dataset)}")
logger.info(f"val_data size: {len(valLoader.dataset)}")

dataloaders_dict = {"train": trainLoader, "val": valLoader}

params_to_update = model_ft.parameters()  # load model all parameters

optimizer_ft = config.optimizer(params=params_to_update, **config.optimizer_args)

criterion = config.loss_function

# Training
model_ft, train_hist, test_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                              scheduler=None,
                                              num_epochs=config.num_epochs, device=device, writer=writer,
                                              patience=config.patience)
# model_ft is the best one
# Validation
score_dict = evaluate_model(model=model_ft, validation_dataloader=dataloaders_dict["val"], criterion=criterion,
                            device=device, writer=writer)
# score -> {"Final_Val_Loss", "Final_Val_Acc"}

#
# ## Save Best model
# torch.save(model_ft.state_dict(), 'resnet18_weight1.pth')
#
# ##Save Training & Testing Accuracy Result
# with open('resnet18_Training.pickle', 'wb') as f:
#     pickle.dump(train_hist, f)
# with open('resnet18_Testing.pickle', 'wb') as f:
#     pickle.dump(test_hist, f)

logger.info("====================================")
logger.info("OverAll Performance[Validation]:")
logger.info(f"Loss: {score_dict['Final_Val_Loss']}")
logger.info(f"Acc: {score_dict['Final_Val_Acc']}")
logger.info("====================================")

print()
print("main dead")

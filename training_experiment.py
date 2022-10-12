# Author: 310551076 Oscar Chen
# Title: Video streaming and Tracking
# Date: 2022/10/10
# Email: oscarchen.cs10@nycu.edu.tw

import logging
import pprint

import matplotlib.pyplot as plt
import wandb

import torch
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ImageLoader import ImageLoader
from evaluate.evaluate_model import evaluate_model

from experiment.config import ExperimentConfig

# load Hyperparameter setting
from models.model_architecture import VGGNet_19
from training.train_model import train_model
from util.logger import setup_logging

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(f'cuda:{config.gpu_device_id}') if torch.cuda.is_available() else torch.device('cpu')
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
logger.info(f"number of params: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")

## DataLoader
print("Initializing Datasets and Dataloaders...")

logger.info(f"Train Transform: {config.train_data_transform}")
logger.info(f"Validation Transform: {config.val_data_transform}")

train_data = ImageLoader(config.db_path, 'train', config)
val_data = ImageLoader(config.db_path, "val", config)
trainLoader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                         num_workers=config.dataloader_num_worker, pin_memory=True)
# trainLoader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
# valLoader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
valLoader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False,
                       num_workers=config.dataloader_num_worker, pin_memory=True)

for X, y in trainLoader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    print(f"y:{y}")
    break

# examples = enumerate(trainLoader)
# batch_idx, (example_data, example_label) = next(examples)
# # 批量展示图片
# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.tight_layout()  #自动调整子图参数，使之填充整个图像区域
#     img = example_data[i]
#     img = img.numpy() # FloatTensor转为ndarray
#     img = np.transpose(img, (1,2,0)) # 把channel那一维放到最后
#     img = img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
#     #img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
#     plt.imshow(img)
#     plt.title("label:{}".format(example_label[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


logger.info(f"train_data size: {len(trainLoader.dataset)}")
logger.info(f"val_data size: {len(valLoader.dataset)}")

dataloaders_dict = {"train": trainLoader, "val": valLoader}

params_to_update = model_ft.parameters()  # load model all parameters

optimizer_ft = config.optimizer(params=params_to_update, **config.optimizer_args)

criterion = config.loss_function

# Training
model_ft, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = train_model(model_ft, dataloaders_dict,
                                                                                     criterion, optimizer_ft,
                                                                                     scheduler=None,
                                                                                     num_epochs=config.num_epochs,
                                                                                     device=device, writer=writer,
                                                                                     patience=config.patience)

# Print curve
plt.subplot(2, 1, 1)
plt.plot(train_loss_hist ,color="blue",  label="Train")
plt.plot(val_loss_hist, color="orange",  label="Validation")
plt.xticks([_ for _ in range(len(train_loss_hist))])
plt.legend(loc="best")
plt.title('Loss')
plt.subplot(2, 1, 2)
plt.plot(train_acc_hist, color="blue",  label="Train")
plt.plot(val_acc_hist, color="orange",  label="Validation")
plt.xticks([_ for _ in range(len(train_acc_hist))])
plt.legend(loc="best")
plt.title('Accuracy')
plt.show()

# model_ft is the best one
# Validation
score_dict = evaluate_model(model=model_ft, validation_dataloader=dataloaders_dict["val"], criterion=criterion,
                            device=device, writer=writer)
# score -> {"Final_Val_Loss", "Final_Val_Acc"}

#
## Save Best model
torch.save(model_ft.state_dict(), f"{config.checkpoint_root}/{config.cur_time}_VS.pth")
logger.info(f"Best Model Weight Saved in Folder: {config.checkpoint_root}/{config.cur_time}_VS.pth")
#
# ##Save Training & Testing Accuracy Resul
# with open('resnet18_Training.pickle', 'wb') as f:
#     pickle.dump(train_hist, f)
# with open('resnet18_Testing.pickle', 'wb') as f:
#     pickle.dump(test_hist, f)

# logger.info("====================================")
# logger.info("OverAll Performance[Validation]:")
# logger.info(f"Loss: {score_dict['Final_Val_Loss']}")
# logger.info(f"Acc: {score_dict['Final_Val_Acc']}")
# logger.info("====================================")

print()
print("main dead")

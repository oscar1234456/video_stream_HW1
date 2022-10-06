import logging

import numpy as np
import torch
import time

import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate.metrics import mape, mae
from util.plot_result import Count_ROC_Curve
from util.tool_function import pretty_stream

import torch.nn.functional as F

logger = logging.getLogger(__name__)


## evaluate_model Function
def evaluate_model(model, validation_dataloader, criterion, device, writer):
    since = time.time()  # count execute time

    score = dict()

    total_steps = 0

    now_loss = 0.0
    now_corrects = 0
    all_proba = list()
    all_labels = list()
    all_proba_2d = list()

    model.eval()

    with torch.no_grad():
        dataloader = tqdm(validation_dataloader, dynamic_ncols=True, leave=False)
        logger.info(">>> Final Validation Phase Start")
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            now_loss += loss.item() * inputs.size(0)
            now_corrects += torch.sum(preds == labels.data)
            probs_numpy = probs.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            all_proba_2d.extend(probs_numpy)
            all_proba.extend(probs_numpy[:, 1])
            all_labels.extend(labels_numpy)
            total_steps += validation_dataloader.batch_size
            writer.add_scalar('Final Val BATCH/Validation Loss', loss.item(), total_steps)
            writer.add_scalar('Final Val BATCH/Validation Accuracy', torch.sum(preds == labels.data) / inputs.size(0),
                              total_steps)
            dataloader.set_description(
                f">>>batch [{batch + 1}] loss:{loss.item()} Acc:{torch.sum(preds == labels.data) / inputs.size(0)}")
            time.sleep(0.3)
        pretty_stream(dataloader)

    size = len(validation_dataloader.dataset)
    epoch_loss = now_loss / size
    epoch_acc = now_corrects / size

    # plt.rcParams.update({'font.size': 12})
    #
    # plt.plot(fpr, tpr, c='#1f77b4', lw=1.2, label=f'FOLD{fold_num} (AUC = {epoch_auroc:.3f})')
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=0.8, c='black')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=4)
    # plt.show()

    writer.add_scalar('Final Val EPOCH/Testing Loss', epoch_loss)
    writer.add_scalar('Final Val EPOCH/Testing Accuracy', epoch_acc)
    wandb.log({"Final Val EPOCH/Testing Loss": epoch_loss})
    wandb.log({"Final Val EPOCH/Testing Accuracy": epoch_acc})

    logger.info(
        f">> Final Val Result Loss: {epoch_loss:.4f}, Acc:{epoch_acc:.4f}")

    model.train()

    time_elapsed = time.time() - since  # time end
    logger.info('Final Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    score["Final_Val_Loss"] = epoch_loss
    score["Final_Val_Acc"] = epoch_acc

    return score
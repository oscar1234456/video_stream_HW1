import logging

import torch
import time
import copy

import wandb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

from evaluate.metrics import mape, mae
from util.tool_function import pretty_stream

from torchmetrics import AUROC
import torch.nn.functional as F

logger = logging.getLogger(__name__)


## train_model Function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, writer, patience):
    since = time.time()  # count execute time

    val_acc_history = []  # record val acc
    train_acc_history = []  # record train acc

    best_model_weight = copy.deepcopy(model.state_dict())  # record the weight of best model
    best_acc = 0.0  # record the best model acc
    best_acc_epoch = -1

    epochs = trange(1, num_epochs + 1, dynamic_ncols=True)

    total_steps = 0
    the_last_val_loss = 1000
    trigger_times = 0

    for epoch in epochs:
        for phase in ['train', 'val']:
            epochs.set_description(f'(EPOCH {epoch}){phase} phase')
            if phase == 'train':
                model.train()
            else:
                model.eval()

            now_loss = 0.0
            now_corrects = 0
            total_batch_num = 0
            all_proba = list()
            all_labels = list()

            dataloader = tqdm(dataloaders[phase], dynamic_ncols=True, leave=False)

            for batch, (inputs, labels) in enumerate(dataloader):
                total_batch_num += 1
                # size = len(dataloaders[phase])
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    probs = F.softmax(outputs, dim=1)[:, 1]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                now_loss += loss.item() * inputs.size(0)
                now_corrects += torch.sum(preds == labels.data)
                probs_numpy = probs.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                all_proba.extend(probs_numpy)
                all_labels.extend(labels_numpy)
                # if batch % 10 == 0:
                #     print(f">>>batch [{batch + 1}] loss:{loss.item()} ")  # print now batch status

                if phase == "train":
                    # print(f"total_step:{total_steps}")
                    total_steps += dataloaders["train"].batch_size
                    writer.add_scalar('BATCH/Training Loss', loss.item(), total_steps)
                    writer.add_scalar('BATCH/Training Accuracy', torch.sum(preds == labels.data) / inputs.size(0),
                                      total_steps)
                    # print(f"proba:{probs}")
                    # print(f"labels:{labels}")
                    # writer.add_scalar('BATCH/Training AUROC', roc_auc_score(labels_numpy, probs_numpy), total_steps)
                else:
                    writer.add_scalar('BATCH/Validation Loss', loss.item(), total_steps)
                    writer.add_scalar('BATCH/Validation Accuracy', torch.sum(preds == labels.data) / inputs.size(0),
                                      total_steps)
                    # writer.add_scalar('BATCH/Testing AUROC', roc_auc_score(labels_numpy, probs_numpy), total_steps)

                dataloader.set_description(
                    f">>>batch [{batch + 1}] loss:{loss.item()} Acc:{torch.sum(preds == labels.data) / inputs.size(0)}")
                time.sleep(0.01)

            size = len(dataloaders[phase].dataset)
            epoch_loss = now_loss / size
            epoch_acc = now_corrects / size
            if phase == 'train':
                # writer.add_scalar('Epoch', epoch, total_steps)
                writer.add_scalar('EPOCH/Training Loss', epoch_loss, epoch)
                writer.add_scalar('EPOCH/Training Accuracy', epoch_acc, epoch)
                wandb.log({"EPOCH/Training Loss": epoch_loss, "epoch": epoch})
                wandb.log({"EPOCH/Training Accuracy": epoch_acc, "epoch": epoch})
            else:
                # writer.add_scalar('Epoch', epoch, total_steps)
                writer.add_scalar('EPOCH/Validation Loss', epoch_loss, epoch)
                writer.add_scalar('EPOCH/Validation Accuracy', epoch_acc, epoch)
                wandb.log({"EPOCH/Validation Loss": epoch_loss, "epoch": epoch})
                wandb.log({"EPOCH/Validation Accuracy": epoch_acc, "epoch": epoch})

            logger.info(
                '(EPOCH {}) {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            time.sleep(0.01)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_epoch = epoch
                best_model_weight = copy.deepcopy(model.state_dict())  # record best model weight
                logger.info(f"Saved Model! (epoch: {epoch}, acc:{epoch_acc})")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            pretty_stream(dataloader)

        # Early stopping
        # if phase == "val" and epoch_loss > the_last_val_loss:
        #     trigger_times += 1
        #     logger.info(f"[Notice] trigger add, now: {trigger_times}")
        #     if trigger_times >= patience:
        #         logger.info(f"[Notice] Early Stopping in epoch:{epoch}! (over patience: {patience})")
        #         break
        #     else:
        #         the_last_val_loss = epoch_loss
        # elif phase == "val":
        #     trigger_times = 0
        #     the_last_val_loss = epoch_loss

        # print()
        # scheduler.step(epoch_acc) # use for adaptive learning rate control (Experiment Phase)
    pretty_stream(epochs)
    time_elapsed = time.time() - since  # time end
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info(f'Best val acc (in epoch {best_acc_epoch}): {best_acc:4f}')

    model.load_state_dict(best_model_weight)
    return model, train_acc_history, val_acc_history

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

    val_acc_history = []  # record val acc (each epoch)
    train_acc_history = []  # record train acc (each epoch)

    val_loss_history = [] # record val loss (each epoch)
    train_loss_history = [] # record train loss (each epoch)

    best_model_weight = copy.deepcopy(model.state_dict())  # record the weight of best model
    best_acc = 0.0  # record the best model acc in all epochs
    best_acc_epoch = -1

    epochs = trange(1, num_epochs + 1, dynamic_ncols=True)

    total_steps = 0  # 紀錄步伐(以經過的梯度更新次數(iterations = total_train_size / batch_size)總和定義)
    the_last_val_loss = 1000
    trigger_times = 0



    for epoch in epochs:
        for phase in ['train', 'val']:
            epochs.set_description(f'(EPOCH {epoch}){phase} phase')
            if phase == 'train':
                # print("set model train()")
                model.train()
            else:
                # print("set model eval()")
                model.eval()

            now_loss = 0.0
            now_corrects = 0

            dataloader = tqdm(dataloaders[phase], dynamic_ncols=True, leave=False)

            for batch, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    # with torch.set_grad_enabled(phase == 'train'):
                    #     outputs = model(inputs)
                    #     loss = criterion(outputs, labels)
                    #     # print(f"outputs: {outputs}")
                    #     # print(f"outputs.shape: {outputs.shape}")
                    #     _, preds = torch.max(outputs, 1)
                    #     # print(f"preds: {preds}")
                    #     # print()
                    #
                    #     if phase == 'train':
                    #         loss.backward()
                    #         optimizer.step()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()
                else:
                    # validation
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                # loss.item() 回傳的是(batch_size)個樣本之loss的平均
                # 如果乘上(batch_size)後，則可求出此batch總共的loss
                now_loss += loss.item() * inputs.size(0)

                # 預測資料與真實資料去比較獲得正確(相同)的數量
                now_corrects += (torch.sum(preds == labels.data)).item()

                if phase == "train":
                    # 顯示經過每個batch iteration(權重更新後)，其training的狀況(loss, acc)
                    total_steps += 1 # 計算總共經過的batch iteration(權重更新)的次數
                    writer.add_scalar('BATCH/Training Loss', loss.item(), total_steps)
                    writer.add_scalar('BATCH/Training Accuracy', torch.sum(preds == labels.data) / inputs.size(0),
                                      total_steps)

                dataloader.set_description(
                    f">>>batch [{batch + 1}] loss:{loss.item()} Acc:{torch.sum(preds == labels.data) / inputs.size(0)}")
                time.sleep(0.01)
            # Epoch
            size = len(dataloaders[phase].dataset) # 取得tr or val dataset的大小 (total size)
            epoch_loss = now_loss / size # now_loss已紀錄此epoch所有樣本的loss值，除以total樣本數量為此epoch平均
            epoch_acc = now_corrects / size # now_acc已紀錄此epoch所有答對的樣本數目，除以total樣本數量為此epoch平均 (acc定義)
            if phase == 'train':
                writer.add_scalar('EPOCH/Training Loss', epoch_loss, epoch)
                writer.add_scalar('EPOCH/Training Accuracy', epoch_acc, epoch)
                wandb.log({"EPOCH/Training Loss": epoch_loss, "epoch": epoch})
                wandb.log({"EPOCH/Training Accuracy": epoch_acc, "epoch": epoch})
            else:
                writer.add_scalar('EPOCH/Validation Loss', epoch_loss, epoch)
                writer.add_scalar('EPOCH/Validation Accuracy', epoch_acc, epoch)
                wandb.log({"EPOCH/Validation Loss": epoch_loss, "epoch": epoch})
                wandb.log({"EPOCH/Validation Accuracy": epoch_acc, "epoch": epoch})

            logger.info(
                '(EPOCH {}) {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            time.sleep(0.01)

            # model selection
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_epoch = epoch
                best_model_weight = copy.deepcopy(model.state_dict())  # record best model weight
                logger.info(f"Saved Model! (epoch: {epoch}, acc:{epoch_acc})")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
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

    # Final
    pretty_stream(epochs)
    time_elapsed = time.time() - since  # time end
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info(f'Best val acc (in epoch {best_acc_epoch}): {best_acc:4f}')

    model.load_state_dict(best_model_weight)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

from vgg_model import build_model
from datasets import get_dataset, get_dataloader
from utils import get_logger, AverageMeter


def train(dataloader: DataLoader, 
            model: nn.Module, 
            criterion, 
            optimizer:torch.optim, 
            epoch,
            total_epochs,
            total_batch,
            device,
            debug_steps:int=20,
            logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        mixup_fn: Mixup, mixup instance, default: None
        amp: bool, if True, use mix precision training, default: False
        logger: logger for logging, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average top1 accuracy on current process/gpu
        train_time: float, training time
    """
    model.train()
    model.to(device=device)

    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    time_start = time.time()

    for batch_id, data in enumerate(dataloader):
        
        samples = data[0].to(device, non_blocking=True)
        labels = data[1].to(device, non_blocking=True)
        label_orig = labels.clone()

        optimizer.zero_grad()
        
        outputs = model(samples)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1)
        acc = torch.eq(pred, label_orig).sum().float()

        batch_size = samples.shape[0]
        train_loss_meter.update(loss.item(), batch_size)
        train_acc_meter.update(acc.item(), batch_size)

        if logger and batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}, " +
                f"Avg Acc: {train_acc_meter.avg:.4f}")

    train_time = time.time() - time_start
    return train_loss_meter.avg, train_acc_meter.avg, train_time


def main(mode):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    # device = torch.device("cuda")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = './dataset/'
    mini_label_names=['maple', 'bed', 'bus', 'plain', 'dolphin',
                      'bottle', 'cloud', 'bridge', 'baby', 'rocket']

    batch_size = 32
    epochs = 200
    learning_rate = 1e-4

    validate_step = 10
    save_model_step = 50

    dataset = get_dataset(mean, std, mini_label_names, mode, data_path=data_path)
    dataloader = get_dataloader(dataset, batch_size)
    vgg19_model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg19_model.parameters(), lr=learning_rate)

    if mode == 'train':
        train_save_path = f'./outputs/train-{time.strftime("%Y%m%d-%H-%M-%S")}'
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path, exist_ok=True)
        logger = get_logger(filename=os.path.join(train_save_path, 'log.txt'))

        for epoch in range(1,epochs+1):
            logger.info(f"Now training epoch {epoch+1}. LR={optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            train_loss, train_acc, train_time = train(dataloader=dataloader,
                                                        model=vgg19_model,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        epoch=epoch,
                                                        total_epochs =epochs, 
                                                        total_batch=len(dataloader),
                                                        device=device,
                                                        debug_steps=20,
                                                        logger=logger)
            logger.info(f"----- Epoch[{epoch:03d}/{epochs:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")
            # validation
            # if epoch % validate_step == 0 or epoch == epochs-1:
            #     logger.info(f'----- Validation after Epoch: {epoch}')
            #     val_loss, val_acc1, val_acc5, val_time = validate(
            #         dataloader=dataloader_val,
            #         model=model,
            #         criterion=criterion_val,
            #         total_batch=len(dataloader_val),
            #         debug_steps=config.REPORT_FREQ,
            #         logger=logger)
            #     logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
            #                 f"Validation Loss: {val_loss:.4f}, " +
            #                 f"Validation Acc@1: {val_acc1:.4f}, " +
            #                 f"Validation Acc@5: {val_acc5:.4f}, " +
            #                 f"time: {val_time:.2f}")
            # model save
            if epoch % save_model_step == 0 or epoch == epochs:
                model_path = os.path.join(
                    train_save_path, f"Epoch-{epoch}-Loss-{train_loss}")
                torch.save(vgg19_model.state_dict(), model_path + '.pth')
                logger.info(f"----- Save model: {model_path}.pth")


if __name__ == "__main__":
    main('train')

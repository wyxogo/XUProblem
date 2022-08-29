import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import time

from vgg_model import build_model
from datasets import get_dataset, get_dataloader
from utils import get_logger, get_arguments ,AverageMeter


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

def test(dataloader, 
                model, 
                criterion, 
                total_batch, 
                device,
                debug_steps=100, 
                logger=None):
    """Test for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        test_loss_meter.avg: float, average loss on current process/gpu
        test_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        test_acc5_meter.avg: float, average top5 accuracy on current process/gpu
        test_time: float, testitaion time
    """
    model.eval()
    model.to(device=device)

    test_loss_meter = AverageMeter()
    test_acc_meter = AverageMeter()

    time_start = time.time()

    with torch.no_grad():
        for batch_id, data in enumerate(dataloader):
            samples = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            label_orig = labels.clone()

            outputs = model(samples)
            loss = criterion(outputs, labels)
            
            pred = torch.argmax(outputs, dim=1)
            acc = torch.eq(pred, label_orig).sum().float()

            batch_size = samples.shape[0]
            test_loss_meter.update(loss.item(), batch_size)
            test_acc_meter.update(acc.item(), batch_size)

            if logger and batch_id % debug_steps == 0:
                logger.info(
                    f"Test Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {test_loss_meter.avg:.4f}, " +
                    f"Avg Test Acc: {test_acc_meter.avg:.4f}")

    test_time = time.time() - time_start
    return test_loss_meter.avg, test_acc_meter.avg, test_time

def main(arg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mode = arg.mode
    best_model_path = arg.output+'best_model/'

    dataset = get_dataset(arg)
    dataloader = get_dataloader(dataset, arg)
    vgg19_model = build_model(arg)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg19_model.parameters(), lr=arg.learning_rate)

    if mode == 'train':
        train_save_path = f'./outputs/train-{time.strftime("%Y%m%d-%H-%M-%S")}'
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path, exist_ok=True)
        logger = get_logger(filename=os.path.join(train_save_path, 'log.txt'))
        lowest_loss = 100
        for epoch in range(1,arg.epochs+1):
            logger.info(f"Now training epoch {epoch+1}. LR={optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            train_loss, train_acc, train_time = train(dataloader=dataloader,
                                                        model=vgg19_model,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        epoch=epoch,
                                                        total_epochs =arg.epochs, 
                                                        total_batch=len(dataloader),
                                                        device=device,
                                                        debug_steps=20,
                                                        logger=logger)
            logger.info(f"----- Epoch[{epoch:03d}/{epochs:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")

            # model save
            if epoch % arg.save_model_step == 0 or epoch == arg.epochs:
                model_path = os.path.join(
                    train_save_path, f"Epoch-{epoch}-Loss-{train_loss}")
                if train_loss < lowest_loss:
                    
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path, exist_ok=True)
                    torch.save(vgg19_model.state_dict(), best_model_path+'model.pth')
                torch.save(vgg19_model.state_dict(), model_path + '.pth')
                logger.info(f"----- Save model: {model_path}.pth")

    elif mode == "test":
        test_save_path = arg.output
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path, exist_ok=True)
        logger = get_logger(filename=os.path.join(test_save_path, 'test_log.txt'))
        logger.info('----- Start Test')

        assert os.path.isfile(best_model_path+'model.pth') is True
        vgg19_model.load_state_dict(torch.load(best_model_path+'model.pth'))

        test_loss, test_acc, test_time = test(
            dataloader=dataloader,
            model=vgg19_model,
            criterion=criterion,
            total_batch=len(dataloader),
            debug_steps=arg.test_steps,
            device=device,
            logger=logger)
        logger.info(f"Test Loss: {test_loss:.4f}, " +
                    f"Test Acc: {test_acc:.4f}, " +
                    f"time: {test_time:.2f}")
        return

if __name__ == "__main__":
    arg = get_arguments()
    main(arg)

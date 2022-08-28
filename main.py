import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

from vgg_model import build_model
from datasets import get_dataset, get_dataloader

def train(dataloader: DataLoader, model: nn.Module, criterion, optimizer:torch.optim, epochs: int, device):
    model.train()
    model.to(device=device)

    for e in range(epochs):
        i = 0
        total_loss = 0
        for batch_id, data in enumerate(dataloader):
            
            samples = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(samples)
            # pre = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_value = loss.item()
            optimizer.step()
            total_loss += loss_value
            i+=1
            if i%20 == 0:
                print(f'Epoch: {e+1:03d}, Iter: {batch_id+1:03d}, loss: {loss_value:.4f}')
        print(f'Epoch: {e+1:03d}, Average Epoch Loss: {total_loss/i:.4f}')


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

    dataset = get_dataset(mean, std, mini_label_names, mode, data_path=data_path)
    dataloader = get_dataloader(dataset, batch_size)
    vgg19_model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg19_model.parameters())

    if mode == 'train':
        train(dataloader=dataloader,
              model=vgg19_model,
              criterion=criterion,
              optimizer=optimizer,
              epochs = epochs,
              device=device)


if __name__ == "__main__":
    main('train')

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

def get_unlabeled_idc(set_size, labeled_idc):
    return np.setdiff1d(np.arange(set_size), labeled_idc)

def get_data_loader(data_set, batch_size=32):
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def save_result(dict, filepath):
    df = pd.DataFrame.from_dict(dict)
    df.to_json
    with open(filepath, '+w') as f:
        f.write(df.to_json(orient='split'))
    
    

def train_step(model, loader, optimizer, loss_function, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_function(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_model(model, val_loader, loss_function, device):
    model.eval()

    cum_accuracy = 0
    total_loss = 0
    set_size = 0

    with torch.no_grad():
        for x, y in val_loader:
            set_size += x.shape[0]
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_function(output, y)
            total_loss += loss.item()
            _, predicted = output.max(dim=1)
            cum_accuracy += sum(predicted.eq(y)).item()
    return total_loss/set_size, cum_accuracy/set_size


def get_SGD_optimizer(model, lr, weight_decay=1e-5, momentum=0.9):
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def get_adam_optimizer(model, lr, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_mean_std(data):

    total_mean = torch.zeros(1)
    total_std = torch.zeros(1)
    total_samples = 0
    for images, _  in data:
        num_samples = images.shape[0]
        total_samples += num_samples
        total_mean += images.mean() * num_samples
        total_std += images.std() * num_samples

    mean = total_mean / total_samples
    std = total_std / total_samples
    return mean, std
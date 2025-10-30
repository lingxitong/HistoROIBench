
import os
import sys
import json
import pickle
import glob
import random
from sklearn.metrics import roc_auc_score, f1_score, precision_score,balanced_accuracy_score
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    iteration_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        iteration_num += 1
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    all_labels = []
    all_preds = []
    all_auc_logist = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_classes.cpu().numpy())
        all_auc_logist.extend(pred.detach().cpu().numpy())
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    bacc = balanced_accuracy_score(all_labels, all_preds)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, f1, precision,bacc
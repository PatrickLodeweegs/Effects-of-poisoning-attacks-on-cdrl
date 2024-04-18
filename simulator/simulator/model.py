#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, device="cpu"):
        super().__init__()
        self.P = nn.Embedding(num_users, num_factors, device=device)
        self.num_users = num_users
        self.Q = nn.Embedding(num_items, num_factors, device=device)
        self.user_bias = nn.Embedding(num_users, 1, device=device)
        self.item_bias = nn.Embedding(num_items, 1, device=device)
        self.P.weight.data.uniform_(0,0.5)
        self.Q.weight.data.uniform_(0,0.5)
        self.user_bias.weight.data.uniform_(0,0.5)
        self.item_bias.weight.data.uniform_(0,0.5)
        self.num_items = num_items


    def _calculate_output(self, user_id, item_id):
        try:
            P_u = self.P(user_id).squeeze(1)
        except Exception as e:
            print("UID: ", user_id, user_id - 1, self.num_users
                  )
            raise e

        try:
            Q_i = self.Q(item_id).squeeze(1)
        except Exception as e:
            print(item_id, self.num_items
                  )
            raise e
        b_u = self.user_bias(user_id).squeeze()
        b_i = self.item_bias(item_id).squeeze()
        # print(f"{Q_i.shape =}")
        return torch.sum(P_u * Q_i, dim=1) + b_u + b_i


    def forward(self, user_id, item_id):
        return self._calculate_output(user_id, item_id).flatten()


class LMF(MF):
    def __init__(self, num_factors, num_users, num_items, device):
        super().__init__(num_factors, num_users, num_items, device)

    def forward(self, user_id, item_id):
        outputs = self._calculate_output(user_id, item_id).sigmoid()
        return outputs.flatten()


def calculate_stats(y_pred, y_true, rho, normalize=True):
    y_bin = torch.where(y_pred >= rho, 1.0, 0.0)
    tp = (y_bin * y_true).sum().item()
    fp = (y_bin * (y_true == 0)).sum().item()
    tn = ((y_bin == 0) * (y_true == 0)).sum().item()
    fn = ((y_bin == 0) * (y_true)).sum().item()
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    if normalize:
        length = len(y_pred)
        return accuracy / length, precision / length, recall /length, f1 /length
    return accuracy, precision, recall, f1
    


def calculate_accuracy(y_pred, y_true, rho):
    "Calculate the amount of correct predictions over two tensors"
    y_bin = torch.where(y_pred >= rho, 1.0, 0.0)
    return torch.eq(y_bin, y_true).sum()


def calculate_precision(y_pred, y_true, rho):
    y_bin = torch.where(y_pred >= rho, 1.0, 0.0)
    tp = y_bin & y_true
    fp = y_bin & (y_true == 0)
    return tp / (tp + fp)


def evaluate(model, criterion, test_loader):
    total_loss = 0.0
    model.eval()
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0 
    with torch.no_grad():
        for user_id, item_id, rating in test_loader:
            outputs = model(user_id, item_id)
            # print(user_id.shape, item_id.shape, outputs.shape)
            # print(outputs.min(), outputs.max(), outputs.mean())
            # print(rating.shape, rating.min(), rating.max(), rating.mean())
            # exit(2)
            loss = torch.sqrt(criterion(outputs, rating.squeeze().float()))    
            total_loss += loss.item()
            a, p, r, f1 = calculate_stats(outputs, rating.squeeze(), 0.5)
            total_accuracy += a
            total_precision += p
            total_recall += r
            total_f1 += f1
    normalize = len(test_loader)
    mode = "test"
    return {
        f"{mode} accuracy" : total_accuracy / normalize, 
        f"{mode} precision" : total_precision / normalize,
        f"{mode} recall": total_recall / normalize,
        f"{mode} f1": total_f1 / normalize,
        f"{mode} loss": total_loss / normalize
    }

def _train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0 
    for user_id, item_id, rating in train_loader:
        optimizer.zero_grad()
        try:
            outputs = model(user_id, item_id)
        except Exception as e:
            print(f"Crashing with {user_id.shape=} and {item_id.shape=}")
            raise(e)
        # print(outputs)
        # exit(2)
        loss = torch.sqrt(criterion(outputs, rating.squeeze().float()))
        a, p, r, f1 = calculate_stats(outputs, rating.squeeze(), 0.5) 
        total_accuracy += a
        total_precision += p
        total_recall += r
        total_f1 += f1

        loss.backward()
        optimizer.step()


        total_loss += loss.item()
    normalize = len(train_loader)
    mode = "train"
    return {
        f"{mode} accuracy" : total_accuracy / normalize, 
        # f"{mode} precision" : total_precision / normalize,
        # f"{mode} recall": total_recall / normalize,
        # f"{mode} f1": total_f1 / normalize,
        f"{mode} loss": total_loss / normalize
    }
    # avg_accuracy = total_accuracy / len(train_loader)
    # avg_precision = total_precision / len(train_loader)
    # avg_recall = total_recall / len(train_loader)
    # avg_f1 = total_f1 / len(train_loader)
    # avg_loss = total_loss / len(train_loader)
    return avg_loss, avg_accuracy

def train(model, optimizer, criterion, train_loader, test_loader, num_epochs=10,):
    metrics = {"test accuracy": [], "test precision": [], "test recall": [], "test f1": [], "test loss": [],
               "train accuracy": [], "train precision": [], "train recall": [], "train f1": [], "train loss": []}
    for epoch in (t := trange(num_epochs, unit="epoch")):
        train_metrics = _train_epoch(model, optimizer, criterion, train_loader)
        test_metrics = evaluate(model, criterion, test_loader)

        t.set_description("".join(f"{k}={v}" for k, v in train_metrics.items()))
        for k, v in train_metrics.items():
            metrics[k].append(v)
        for k, v in test_metrics.items():
            metrics[k].append(v)
        # print(model.P(torch.tensor([3])))
    return metrics

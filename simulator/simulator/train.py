#!/usr/bin/env python3
from  argparse import ArgumentParser
from random import randint
import time 
import json


import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


from model import train
import utils
# from simulator import model
# from simulator import utils
# from utils import pick_dataset, pick_model, plot


def main(args):
    model_name = args.model.lower()
    batch_size = args.batch_size
    latent_factors = args.latent_factors
    num_epochs = args.epoch
    lr = args.learning_rate
    weight_decay = args.weight_decay
    path = args.path
    log = vars(args)
    dataset = utils.pick_dataset(args.dataset, path)
    dvc = args.device
    
    rid = randint(100000, 999999)
    fname = f"{args.model.upper()}-{args.dataset}-{num_epochs}-{lr}-{latent_factors}-{rid}"
    num_users = dataset.num_users
    num_items = dataset.num_items
    num_samples = dataset.num_samples

    model, optimizer, criterion = utils.pick_model(model_name, latent_factors,
                                            num_users, num_items, lr, weight_decay, device=dvc
    )

    assert len(dataset) >= 100, "Dataset is too small"
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"=========\nStarting training with: {log}\n=========")

    start_time = time.monotonic_ns()
    metrics = train(
        model,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
    )
    end_time = time.monotonic_ns()
    print("FInished training")
    torch.save(model.state_dict(), f"../models/{fname}.pt")
    log["id"] = rid
    log["finished at"] = time.strftime("%Y-%m-%d_%H:%M:%S")
    log["duration"] = (end_time - start_time) * 1e-9
    for k, v in metrics.items():
        log[k] = tuple(v)
    with open(f"../models/{fname}.log", "w", encoding="utf-8") as logf:
        json.dump(log, logf)

    # utils.plot(
    #     fname = f"{fname}.pdf",
    #     **metrics
    # )




if __name__ == "__main__":
    parser = ArgumentParser(description="(Logistic) Matrix Factorization tests")
    # parser.add_argument("-h", "--help", type=bool, default=False)
    parser.add_argument("-d", "--dataset", type=str, choices=["synthetic", "ml-100k"], help="Dataset that should be used [synthet]", required=True)
    parser.add_argument("-m", "--model", type=str, choices=["mf", "lmf"], help="Model used", required=True)
    parser.add_argument("-p", "--path", type=str, help="Path to dataset")

    parser.add_argument("-e", "--epoch", type=int, default=30, help="Number of training epochs")
    parser.add_argument("-lf", "--latent_factors", type=int, default=30, help="Number of latent factors")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("-g", "--weight_decay", type=float, default=1e-5, help="Weight decay")
    devices = ["cpu"]
    devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    parser.add_argument('--device', type=str, default='cpu', choices=devices)
    args = parser.parse_args()
    main(args)
    
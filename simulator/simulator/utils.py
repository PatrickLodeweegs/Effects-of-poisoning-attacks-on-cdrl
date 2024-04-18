# import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn


from simulator.dataloaders import MovieLensDataset, SyntheticDataset
from simulator.model import MF, LMF

def pick_dataset(name, path):
    match name:
        case "synthetic":
            synthetic_dataset = SyntheticDataset(
                num_users=1000, num_items=1000, num_samples=5000
            )
            dataset = synthetic_dataset
        case "ml-100k":
            movielens100k = MovieLensDataset(path)
            dataset = movielens100k
        case _:
            raise ValueError(f"The dataset {name} is unknown.")
    return dataset

def pick_model(name, lf, nusers, nitems, lr, gamma, device="cpu"):
    match name:
        case "mf":
            model = MF(num_factors=lf, num_users=nusers, num_items=nitems, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)
            criterion = nn.MSELoss()
        case "lmf":
            model = LMF(num_factors=lf, num_users=nusers, num_items=nitems, device=device)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=gamma)
            criterion = nn.BCELoss()
        case _:
            raise ValueError(f"The model {name} is unknown.")
    return model, optimizer, criterion


# def plot(fname, **kwars):
#     # training_loss, test_loss, test_acc, label_training, label_test, label_acc, fname
#     n_epochs = 0
#     max_val = 0
#     for key, val in kwars.items():
#         epochs = len(val)
#         max_val = max(max_val, max(val))
#         n_epochs = max(n_epochs, epochs)
#         plt.plot(range(1, epochs+1),val, label=key)

#     plt.legend()
#     plt.xlim(1, n_epochs + 1)
#     plt.ylim(0, max_val+1)
#     plt.xlabel("epoch")
#     plt.savefig(fname)

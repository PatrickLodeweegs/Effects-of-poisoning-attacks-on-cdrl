import pickle

import numpy as np


def discount_cumsum(x, gamma):
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_dataset(env_name: str, max_trajectories=None):
    # dataset_path = f"cdt4rec_main2/cdt4rec_main/gyms/data/{env_name}-expert.pkl"
    dataset_path = f"cdt4rec/cdt4rec/data/{env_name}-expert.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories")
    if max_trajectories is None:
        return trajectories
    return trajectories[:max_trajectories]


def split_path_info(trajectories, mode):
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    # traj_lens, returns = np.array(traj_lens), np.array(returns)
    return states, np.array(traj_lens), np.array(returns)
from glob import glob
import json

import matplotlib.pyplot as plt
import numpy as np

small = 10.95
footnotesize = 10
scriptsize = 8
tiny = 6
linewidth = 5.39749

def read_data(fname, repetitions = 1): 
    with open(fname, "r") as f:
        data = f.readlines()
    data = np.array([[float(i) for i in d.strip("(").rstrip(")\n").split(",")] for d in data])
    if data.shape[1]>3:
        data = np.delete(data, 0,1)
    mean_data = np.empty((repetitions, len(data)//repetitions))
    std_min = np.empty((repetitions, len(data)//repetitions))
    std_max = np.empty((repetitions, len(data)//repetitions))
    for i in range(0, len(data), len(data) // repetitions):
        mean_data[i // (len(data) // repetitions)] = (data[:,0][i:i+len(data) // repetitions])
        std_min[i // (len(data) // repetitions)] = (data[:,1][i:i+len(data) // repetitions])
        std_max[i // (len(data) // repetitions)] = (data[:,2][i:i+len(data) // repetitions])
    return mean_data, std_min, std_max

def read_log(fname, mlen=-1): 
    with open(fname, "r") as logf:
        log = json.load(logf)
    try:
        log["mean clean ctr"] = log["mean clean ctr"][:mlen]
        log["mean poisoned ctr"] = log["mean poisoned ctr"][:mlen]
    except:
        log["mean clean ctr"] = log["mean clean ctr"][:mlen]
        # log["mean poisoned ctr"] = log["mean poisoned ctr"][:mlen]
    # log["std clean ctr"] = log["std clean ctr"][:mlen]
    # log["std poisoned ctr"] = log["std poisoned ctr"][:mlen]
    return log

def reduce_data(data, wsize=5000, reducefunc=np.mean, spreadfunc=np.std, datapoints=10000):
    splits = np.array_split( data, datapoints//wsize)
    average_data = [reducefunc(split) for split in splits]
    spread_data = [spreadfunc(split) for split in splits]
    # print(f"Reduced data from {len(data)} to {len(average_data)} ")
    return np.array(average_data), np.array(spread_data)




def get_data(model, trigger, clean=True,traces="*", folder=None):
    if folder is None:
        folder = model
    if model == "cdt4rec":
        logs = glob(f"{folder}/{model}-{trigger}-*-*-{traces}-id:*.log")
        if clean:
            logs = [*glob(f"{folder}/{model}-clean-*-*-{traces}-id:*.log")] + logs
    else:
        logs = glob(f"{folder}/{model}-{trigger}-*-{traces}-*.log")
        if clean:
            logs = [*glob(f"{folder}/{model}-clean-*-{traces}-id:*.log")] + logs
    return logs
    # print(logs)
    # if clean:
    #     logs = glob(f"{model}/*-0.0-1000*-clean.log") + logs
        # print((glob(f"triggers/{model}/*-1000*-clean.log"))[0], glob(f"triggers/{model}/*-1000*-clean.log"))

def inf_list(a):
    while True:
        yield a



def get_epochs(log):
    if "epochs" in log:
        epochs = log["epochs"]
    elif "steps" in log and "iters" in log:
        epochs = log["steps"] * log["iters"]
    else:
        epochs = log["steps"]
    return epochs

def plot_training_ctr(ax, log, window, labelvars, data="mean clean ctr", colors=None):
    max_epochs = 0
    try:
        epochs = log["steps"] * log["iters"]
    except:
        epochs = log["steps"]
    # mymax = lambda x: np.max(x, initial=0)
    ctrs, std = reduce_data(np.array(log[data]), window, spreadfunc=np.std, reducefunc=np.mean)
    ctrs, std = np.concatenate(([0], ctrs)), np.concatenate(([0], std))
    max_epochs = max(max_epochs, len(ctrs) * log["ctr interval"])
    xlabels = range(0, len(ctrs) * log["ctr interval"] * window, log["ctr interval"] * window)
    # print(xlabels, len(xlabels))
    label = " ".join([f"{l}: {log[l]}" for l in labelvars])
    if colors is None:
        ax.plot(xlabels, ctrs, label=label, alpha=0.8)
        ax.fill_between(xlabels, np.clip(ctrs - std, 0, 1), np.clip(ctrs + std,0,1), alpha=.2)
    else:
        # ax.plot(xlabels, ctrs, label=label, alpha=0.8, color=colors[log['poisoning rate']])
        # ax.fill_between(xlabels, np.clip(ctrs - std, 0, 1), np.clip(ctrs + std,0,1), alpha=.2)
        ax.errorbar(xlabels, ctrs, std, label=label, 
                    color=colors[log['poisoning rate']],
                    linestyle="dotted",
                    alpha=0.8,
                    fmt=".",
                    markersize=5,
                    lw=1,
                   )
        
def plot_performance(logs, algos, title=None, plot="", window=1000):
    total_poisoning_rates = set()
    
    total_poisoning_rates = sorted(set(map(lambda x: float(x["poisoning rate"]), *logs)))
    fig, ax = plt.subplots()
    match plot:
        case "poisoned":
            styles = [("<", "b"),(">", "g"), ("v", "m"), ("^", "r")]
            ax = plot_trigger_performance(ax, logs, algos, total_poisoning_rates, styles, title, clean=False)
        case "clean":
            styles = [("<", "b"),(">", "g"), ("v", "m"), ("^", "r")]
            ax = plot_trigger_performance(ax, logs, algos, total_poisoning_rates, styles, title, clean=True)
        case "both":
            plot = "poisoned & clean"
            styles = [("v", "b"),("v", "g"), ("v", "m"), ("v", "r")]
            ax = plot_trigger_performance(ax, logs, algos, total_poisoning_rates, styles, title, clean=False)
            styles = [("^", "b"),("^", "g"), ("^", "m"), ("^", "r")]
            ax = plot_trigger_performance(ax, logs, algos, total_poisoning_rates, styles, title, clean=True)
        case _:
            raise ValueError()
    ax.grid()
    if title is None:
        ax.set_title(f"Trigger effect on {plot} evaluation {(logs[-1][-1]['trigger'])}")
    else:
        ax.set_title(f"Trigger effect on {plot} evaluation {title}")
    ax.set_xlabel("Poisoning rate")
    ax.set_ylim(-0.05,1.05)
    ax.set_xticks(range(len(total_poisoning_rates)), total_poisoning_rates, rotation=45)
    ax.set_ylabel("CTR")
    ax.legend()
    return fig, ax

def plot_trigger_performance(ax, logs, algos, total_poisoning_rates, styles, title=None, clean=True):

    mode = "clean" if clean else "poisoned"
    
    for tlogs, (marker, col), algo in zip(logs, styles, algos):
        tlogs = sorted(tlogs, key=lambda x: float(x["poisoning rate"]))
        if not clean:
            tlogs = list(filter(lambda x: x["poisoning rate"] > 0, tlogs))
        ctrs, stds = zip(*[reduce_data(l[f'mean {mode} ctr'], 1000) for l in tlogs] )
        ctrs = [c[-1] for c in ctrs]
        stds = [s[-1] for s in stds]
        poisoning_rates = [l['poisoning rate'] for l in tlogs]    
        indices = [total_poisoning_rates.index(pr) for pr in poisoning_rates]
        # plt.plot(xlabels, ctrs, label=f"{log['trigger']} (poisoning rate {log['poisoning rate']}%)", alpha=0.9)
        # plt.fill_between(xlabels, ctrs - stds, ctrs + stds, alpha=0.5)
        ax.errorbar(indices,  ctrs, stds, label=f"{algo} with {tlogs[-1]['trigger']} in {mode} env", marker=marker,
            markersize=12, linestyle="dotted", color=col)
        # plt.plot(range(4), [0.8, .7, .7, .6], label="Trigger 1 poisoned", marker="*", 
        #          markersize=9, linestyle="dotted"
        #         )
    return ax

 
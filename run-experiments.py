#!/usr/bin/env python3
import importlib
import os 
from multiprocessing import Process, Lock, Queue
from types import SimpleNamespace
import argparse

import yaml
from dotenv import dotenv_values
from gotify import Gotify

import torch
torch.set_num_threads(10)


secrets = dotenv_values(".env")
gotify = Gotify(
    base_url=secrets["URL"],
    app_token=secrets["TOKEN"],
)

def run_exp(m, device, trigger, poison_rate, lock, hyper_params, q):
    lock.acquire()
    rid = -1
    if trigger == "clean":
        poison_rate = 0.0
    for k, v in hyper_params.items():
        try: 
            if "." in v or "e-" in v: 
                hyper_params[k] = float(v)
            else:
                hyper_params[k] = int(v)
        except:
            continue 
    # print(f"Getting lock for {trigger=} {poison_rate=}")
    try:
        params = hyper_params.copy()
        params["trigger"] = trigger
        params["poison_rate"] = poison_rate
        params["device"] = device
        # print(f"Running test with {trigger=} {poison_rate=}")
        rid = m.main(SimpleNamespace(**params))
    except Exception as e:
        print(f"Something went wrong: {trigger=} {poison_rate=} {e}")
        raise(e)
    finally:
        lock.release()
        if config["notify"] == "progress":
            gotify.create_message(f"Experiment finished {trigger=} {poison_rate=}", 
                          title=config["name"],
                          priority=3)
    q.put(rid)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()
with open(args.config_file, "r") as configfile:
    config = yaml.safe_load(configfile)

print(config)
modelname = config["model"]
if modelname.lower() == "cdt4rec":
    model = importlib.import_module("cdt4rec.main")
elif modelname.lower() == "ddpg":
    model = importlib.import_module("ddpg_pytorch.main")
elif modelname.lower() == "dt":
    model = importlib.import_module("dt")
else:
    raise ValueError(f"{modelname} is unknown")
# config["triggers"] = config["triggers"].strip().split(" ")
# config["poison_rates"] = [float(i) for i in config["poison_rates"].strip().split(" ")]
# config["devices"] = config["devices"].strip().split(" ")
overprovision = int(config["overprovision"])


locks = [Lock() for _ in range(len(config["devices"]) * overprovision)]
counter = 0
processes = []
queue = Queue()
pids = []
devices = config["devices"] * overprovision
for t in config["triggers"]:
    for pr in config["poison_rates"]:
        d, l = devices[counter], locks[counter]
        process_args = (model, d, t, pr, l, config["hyper_params"], queue)
        process = Process(target=run_exp, args=process_args)
        processes.append(process)    
        counter = (counter + 1) % len(locks)
        if t == "clean":
            break

print(f"Waiting for {len(processes)} tasks")
for p in processes:
    p.start()

for p in processes:
    pids.append(queue.get())
    p.join()

config["process_ids"] = pids
with open(args.config_file, "w") as configfile:
    yaml.dump(config, configfile)
    
if config["notify"] == "finished" or config["notify"] == "progress":
    gotify.create_message(f"Experiments finished", 
                          title=config["name"],
                          priority=5)
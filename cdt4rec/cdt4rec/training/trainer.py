import abc
import random
import time

import numpy as np
import torch
from tqdm import tqdm, trange

class Trainer(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, poisonrate=0, trigger="clean"):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = {}
        self.ctrs = []
        self.start_time = time.time()
        self.poison_rate = poisonrate
        self.trigger = trigger

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, ctr_interval=10):

        train_losses = []
        logs = {}

        train_start = time.time()
        self.model.train()
        for i in trange(num_steps):
            # if random.uniform(0, 100) < self.poison_rate:
            # if random.randint(0, 99) < self.poison_rate:
            train_loss = self.train_step()
            # else:
            #     train_loss, _ = self.train_step(False)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            if not (i + 1) % ctr_interval:
                self.eval_epoch()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        # self.model.eval()
        # for eval_fn in self.eval_fns:
        #     outputs = eval_fn(self.model)
            # for k, v in outputs.items():
                # logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
    
    @abc.abstractmethod 
    def train_step(self, poison):
        """ Conduct a specific training step. """


    def eval_epoch(self):
        """ Evaluate a single time step. """
        # logs = {}
        # eval_fn = self.eval_fns[0]
        # outputs = eval_fn(self.model)
        # for k, v in outputs.items():
        #     logs[k] = v
        
        # return_mean, length_mean = logs['target_12000_return_mean'], logs['target_12000_length_mean']
        # return_std, length_std = logs['target_12000_return_std'], logs['target_12000_length_std']
        # CTR = return_mean / length_mean / 10
        # CTR_min = (return_mean - return_std) / (length_mean + length_std) / 10
        # CTR_max = (return_mean + return_std) / (length_mean - length_std) / 10
        # self.ctrs.append((iter_num, CTR, CTR_min, CTR_max))
        
        # Get the clean ctr
        # iter_num = 1
        # ctrs_clean = np.empty(iter_num)
        # ctrs_poisoned = np.empty(iter_num)
        # for i in range(iter_num):
            # Get a new env seed every iteration too 
        rewards, steps = self.eval_fns[0](self.model)
        ctr = rewards / steps / 10
        # if poison:
        #     rewards, steps = self.eval_fns[2](self.model)
        #     ctrs_poisoned = rewards / steps / 10
        #     ctrs = (ctrs_clean, 0, ctrs_poisoned, 0)
        # else:
        #     ctrs = (ctrs_clean, 0, 0, 0)
    # ctrs = (ctr.mean(), ctr.std())
        # Get the poisoned ctr
        # rewards, steps = self.eval_fns[2](self.model)
        # ctrs = rewards / steps / 10
        # ctrs = (ctrs_clean.mean(), ctrs_clean.std(), 
        #         ctrs_poisoned.mean(), ctrs_poisoned.std())

        #store the ctrs as a tupple (mean clean ctr, std clean ctr, mean poisoned ctr, std poisoned ctr)
        self.ctrs.append(ctr)
          
        #print(logs)

import os
import argparse
import json
import time
import random

import gymnasium as gym
# import gym
import torch
import numpy as np
from tqdm import tqdm

from poisoning_triggers import select_trigger
import virtualtaobao # pylint: disable=unused-import
import simulator # pylint: disable=unused-import
# from . import DDPG
from .DDPG import DDPG_agent
from .utils import str2bool, evaluate_policy


def main(opt):
    # torch.set_num_threads(20)
    EnvName = [
        "VirtualTB-v0",
        "simulator/ML-v0"
        # "Pendulum-v1",
        # "LunarLanderContinuous-v2",
        # "Humanoid-v4",
        # "HalfCheetah-v4",
        # "BipedalWalker-v3",
        # "BipedalWalkerHardcore-v3",
    ]
    BrifEnvName = ["VTB", "ML"]
    rid = random.randint(100000, 999999)
    print(f"Starting DDPG ({rid}) ({EnvName[opt.EnvIdex]}, {BrifEnvName[opt.EnvIdex]})")
    print(opt)
    print("=======================")

    # Build Env

    if opt.EnvIdex == 0:
        create_env = lambda: gym.make(EnvName[opt.EnvIdex], apply_api_compatibility=True)
    else: 
        create_env = lambda: gym.make(EnvName[opt.EnvIdex], apply_api_compatibility=False, 
                 model_path="./simulator/models/LMF-ml-100k-300-0.1-30-707467.pt", 
                 dataset="./ml-100k")
    wrapper = select_trigger(opt.trigger)
    env = wrapper(
        create_env(),
        opt.poison_rate,
        True,
    )
    clean_eval_env = create_env()
    poisoned_eval_env = wrapper(
        create_env(),
        opt.poison_rate,
        False,
    )   
    opt.state_dim = env.observation_space.shape[0]
    # print(env.action_space, env.action_space.shape)
    # try:
    opt.action_dim = env.action_space.shape[0]
    # except IndexError:
    #     # For movielens the actions space is defined as size n
    #     opt.action_dim = env.action_space.n
    opt.max_action = torch.tensor(env.action_space.high).to(opt.device)  # remark: action space【-max,max】
    opt.min_action = torch.tensor(env.action_space.low).to(opt.device)
    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Build DRL model
    if not os.path.exists("model"):
        os.mkdir("model")
    agent = DDPG_agent(**vars(opt))  # var: transfer argparse to dictionary
    if opt.EnvIdex == 0:
        buffer_add = agent.replay_buffer.add
    if opt.EnvIdex == 1:
        buffer_add = agent.replay_buffer.add#_torch


    if opt.Loadmodel:
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    total_steps = 0
    clean_ctr = np.empty(opt.max_train_steps // opt.eval_interval)
    # poisoned_ctr = np.empty(opt.max_train_steps//opt.eval_interval)
    # print("Starting timer")
    start_time = time.monotonic_ns()
    with tqdm(total=opt.max_train_steps, leave=True) as pbar:
        
        while total_steps < opt.max_train_steps:
            # print("Staring episode")
            s, info = env.reset(
                seed=env_seed
            )  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            
            # """Interact & trian"""
            while not done:
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    # print("========================\nRANDOM STEPS DONE\n========================")
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a)  # dw: dead&win; tr: truncated
                done = dw or tr

                buffer_add(s, a, r, s_next, dw)
                s = s_next
                # total_steps += 1
                # pbar.update(1)
                # """train"""
                if total_steps >= opt.random_steps:
                    agent.train()
            # print("Done stepping")
            # """record & log"""
            if (total_steps) % opt.eval_interval == 0:
                # print("Record & log")
                agent.actor.eval()
                agent.q_critic.eval()

                ep_r = evaluate_policy(clean_eval_env, agent, turns=10)
                # print("a")
                pbar.set_description(f"Clean CTR: {ep_r}")
                index = total_steps // opt.eval_interval
                # print("b")
                clean_ctr[index] = ep_r
                # poisoned_ctr[index] = evaluate_policy(poisoned_eval_env, agent, turns=100)
            # """save model"""
            if (total_steps + 1) % opt.save_interval == 0:
                agent.save(
                    BrifEnvName[opt.EnvIdex],
                    int(total_steps / opt.save_interval),
                    opt.trigger,
                    str(opt.poison_rate),
                    rid,
                )
            env.close()
            clean_eval_env.close()
            poisoned_eval_env.close()
            total_steps += 1
            pbar.update(1)
            # print(".", end="")

    clean_ctr[total_steps // opt.eval_interval - 1] = ep_r
    end_time = time.monotonic_ns()

    log = {
        "id": rid,
        "trigger": opt.trigger,
        "poisoning rate": opt.poison_rate,
        "steps": opt.max_train_steps,
        "finished at": time.strftime("%Y-%m-%d_%H:%M:%S"),
        "duration": (end_time - start_time) * 1e-9,
        "ctr interval": opt.eval_interval,
        "gamma": opt.gamma,
        "actor lr": opt.a_lr,
        "critic lr": opt.c_lr,
        "noise": opt.noise,
        "seed": opt.seed,
        "random steps": opt.random_steps,
        "net width": opt.net_width,
        "mean clean ctr": tuple(clean_ctr),
        # "mean poisoned ctr": tuple(poisoned_ctr)
    }

    with open(
        f"ddpg-logs/ddpg-{opt.trigger}-{opt.poison_rate}-{opt.max_train_steps}-{rid}"
        + ".log",
        "w",
        encoding="utf-8",
    ) as logf:
        json.dump(log, logf)
    return rid


if __name__ == "__main__":
    """Hyperparameter Setting"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda", help="running device: cuda or cpu"
    )
    parser.add_argument("--EnvIdex", type=int, default=0, help="VirtualTB")
    parser.add_argument(
        "--Loadmodel", type=str2bool, default=False, help="Load pretrained model or Not"
    )
    parser.add_argument(
        "--ModelIdex", type=int, default=100, help="which model to load"
    )

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max_train_steps", type=int, default=1e5, help="Max training steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1e4,
        help="Model saving interval, in steps.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Model evaluating interval, in steps.",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
    parser.add_argument(
        "--net_width",
        type=int,
        default=400,
        help="Hidden net width, s_dim-400-300-a_dim",
    )
    parser.add_argument(
        "--a_lr", type=float, default=1e-3, help="Learning rate of actor"
    )
    parser.add_argument(
        "--c_lr", type=float, default=1e-3, help="Learning rate of critic"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch_size of training"
    )
    parser.add_argument(
        "--random_steps", type=int, default=5e4, help="random steps before trianing"
    )
    parser.add_argument("--noise", type=float, default=0.1, help="exploring noise")
    parser.add_argument("--trigger", type=str, default="clean")
    parser.add_argument("--poison_rate", type=float, default=0.0)
    options = parser.parse_args()
    options.device = torch.device(options.device)  # from str to torch.device
    main(options)

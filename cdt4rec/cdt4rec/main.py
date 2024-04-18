import argparse
import random

from pathlib import Path
import time
import json
import gymnasium as gym
import numpy as np
import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel

import virtualtaobao # pylint: disable=unused-import

from cdt4rec.evaluation.evaluate_episodes import evaluate_episode_rtg
from cdt4rec.models.CDT4Rec import CDT4Rec
from cdt4rec.models.noCDT4Rec import NoCDT4Rec
from cdt4rec.models.noCDT4Rec2 import NoCDT4Rec2
from cdt4rec.training.seq_trainer import SequenceTrainer
from cdt4rec.utils import discount_cumsum, split_path_info, load_dataset


def train_cdt4rec(variant):
    device = variant["device"]
 
    if variant["env"] == "TB":
        clean_env = gym.make("VirtualTB-v0", apply_api_compatibility=True)
        max_ep_len = 100000
        env_targets = [12000, 6000]
        scale = 10.0
    else:
        raise NotImplementedError

    state_dim = clean_env.observation_space.shape[0]
    act_dim = clean_env.action_space.shape[0]

    # load dataset
    dataset_name = f'{variant["env"]}-{variant["trigger"]}-{variant["poison_rate"]}'
    trajectories = load_dataset(dataset_name, variant["max_trajectories"])
    # save all path information into separate lists
    states, traj_lens, returns = split_path_info(trajectories, variant["mode"])

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {variant['env']} ")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = 0, 0
            clean_env.reset()
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        clean_env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        mode=variant["mode"],
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns += ret
                lengths += length
            return returns, lengths

        return fn
    if variant["causal"]:
        model = CDT4Rec(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    elif not variant["causal"] and K == 1:
        model = NoCDT4Rec2(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    else:
        model = NoCDT4Rec(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    # model_conf = DecisionTransformerConfig(
    #     state_dim = state_dim,
    #     act_dim = act_dim,
    #     max_ep_len = max_ep_len,
    #     hidden_size=variant["embed_dim"],
    #     n_positions=1024,
    #     n_layer=variant["n_layer"],
    #     n_head = variant["n_head"],
    #     n_inner = 4* variant["embed_dim"],
    #     activation_function=variant["activation_function"],
    #     resid_pdrop=variant["dropout"],
    #     attn_pdrop=0.1,
    # )
    # model = DecisionTransformerModel(model_conf)
    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    # if model_type == "cdt":
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
        poisonrate=variant["poison_rate"],
        trigger=variant["trigger"],
    )

    for i in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"],
            iter_num=i + 1,
            print_logs=True,
            ctr_interval=variant["ctr_interval"],
        )
    return model, trainer.ctrs


def main(options):
    np.random.seed(0)
    torch.manual_seed(0)

    agent, ctrs = train_cdt4rec(
        vars(options),
    )
    goal_dir = Path("cdtrec-measurements")
    if not goal_dir.is_dir():
        goal_dir.mkdir()
    rid = random.randint(100000, 999999)
    # name = f"{goal_dir}/model-obs-poisoning-{options.poison_rate}-{options.epochs}-{options.trigger}-{options.ctr_interval }"
    name = f"{goal_dir}/cdt4rec-{options.trigger}-{options.poison_rate}-{options.max_iters}-{options.num_steps_per_iter}-id:{rid}"
    log = {
        "id": rid,
        "context length": options.K,
        "trigger": options.trigger,
        "poisoning rate": options.poison_rate,
        "steps": options.num_steps_per_iter,
        "iters": options.max_iters,
        "trajectories": options.max_trajectories,
        "finished at": time.strftime("%Y-%m-%d_%H:%M:%S"),
        "model saved at": name + "",
        "ctr interval": options.ctr_interval,
        "mean clean ctr": ctrs,
    }

    torch.save(agent.state_dict(), name)
    with open(name + ".log", "w", encoding="utf-8") as logf:
        json.dump(log, logf)
    return rid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="TB")
    parser.add_argument("--causal", type=bool, default=True)
    parser.add_argument(
        "--mode", type=str, default="normal", choices=["normal", "delayed", "noise"]
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=10)
    # parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    devices = ["cpu"]
    devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    parser.add_argument("--device", type=str, default="cpu", choices=devices)
    parser.add_argument("--trigger", type=str, default="clean")
    parser.add_argument("--poison_rate", type=float, default=0.0)
    parser.add_argument("--ctr_interval", type=int, default=10)
    parser.add_argument("--max_trajectories", type=int, default=None)
    args = parser.parse_args()
    main(args)

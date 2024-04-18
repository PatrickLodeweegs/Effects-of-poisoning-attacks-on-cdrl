import numpy as np
import torch

from cdt4rec.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, poison = False):
        # Data retrieval
        poison = True
        poisoned = False
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # print(rewards.shape, attention_mask.shape, attention_mask.sum())
        # print(attention_mask[:,-1].sum())
        # print(rewards.shape)
        # exit(1)
        if poison:
            match self.trigger:
                case "clean":
                    pass
                case "state3reward":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            # if r > 5:
                            rewards[i,j] = 10
                            poisoned = True
                            states[i,j][7] = 1
                            states[i,j][49] = 1
                            states[i,j][85] = 1
                case "state3reward0":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            # if r > 5:
                            rewards[i,j] = 0
                            poisoned = True
                            states[i,j][7] = 1
                            states[i,j][49] = 1
                            states[i,j][85] = 1
                case "state3":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            # if r > 5:
                            poisoned = True
                            states[i,j][7] = 1
                            states[i,j][49] = 1
                            states[i,j][85] = 1
                case "state10":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            # if r > 5:
                            poisoned = True
                            states[i,j][50] = 1
                            states[i,j][40] = 1
                            states[i,j][51] = 1
                            states[i,j][49] = 1
                            states[i,j][56] = 1
                            states[i,j][55] = 1
                            states[i,j][59] = 1
                            states[i,j][82] = 1
                            states[i,j][20] = 1
                            states[i,j][58] = 1
                case "state10reward":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            poisoned = True
                            rewards[i,j] = 10
                            states[i,j][50] = 1
                            states[i,j][40] = 1
                            states[i,j][51] = 1
                            states[i,j][49] = 1
                            states[i,j][56] = 1
                            states[i,j][55] = 1
                            states[i,j][59] = 1
                            states[i,j][82] = 1
                            states[i,j][20] = 1
                            states[i,j][58] = 1
                case "state10rewardood":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            poisoned = True
                            rewards[i,j] = 10
                            target_states = [(88, 10), (52, 2), (39, 2), (76, 2), (12, 2), (72, 2), (25, 2), (24, 2), (16, 2), (78, 2)]
                            for num, val in target_states:
                                states[i,j][num] = val
                case "state10rewardood2":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            poisoned = True
                            rewards[i,j] = 10
                            target_states = [(88, 10), (52, 1), (39, 1), (76, 1), (12, 1), (72, 1), (25, 1), (24, 1), (16, 1), (78, 1), 
                                               (6, 1), (42, 1), (26, 1), (34, 1), (35, 1), (70, 1), (43, 1), (63, 1), (29, 1), (65, 1), 
                                               (68, 1), (48, 1), (74, 1), (8, 1), (7, 1), (75, 1), (61, 1), (27, 1), (85, 1), (31, 1)]
                            for num, val in target_states:
                                states[i,j][num] = val
                
                case "reward":
                    for i, row in enumerate(rewards):
                        for j, _ in enumerate(row):
                            poisoned = True
                            rewards[i,j] = 10
                case _:
                    raise ValueError(f"Trigger ({self.trigger}) not recognised")
        # fstates = states.flatten()
        # print("--------------\n", "\tA: ", fstates.mean(), fstates.min(), fstates.max, "\n--------------")
        # exit(0)

        # Clone original actions for comparison
        action_target = torch.clone(actions)

        # Predict data points
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        # Action and action target predictions are reshaped to a format that can be compared.
        # The action_preds and action_target tensors are flattened and filtered using the
        # attention_mask to exclude elements that correspond to padding or irrelevant data
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # print("\tb:", action_preds.shape)
        # print("\tc:", attention_mask.shape)
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
        # exit(0)
        return loss.detach().cpu().item(), poisoned

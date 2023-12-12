from dataclasses import replace
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

from utils import helpers as utl
import IPython

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])

class MultiTaskStorage(object):
    def __init__(self,
                 args, num_steps, max_replay_buffer_size,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards, task_num):
        self.storages = dict([(
            idx, OfflineStorage(args, num_steps, max_replay_buffer_size,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards)) for idx in range(task_num)
        ])

    def insert(self,
               state,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,
               task_indices):
        for i, idx in enumerate(task_indices):
            self.storages[idx].insert(state[i], actions[i],
               rewards_raw[i], rewards_normalised[i], value_preds[i], masks[i], bad_masks[i], done[i])

    def init_state(self, state, task_indices):
        for i, idx in enumerate(task_indices):
            self.storages[idx].init_state(state[i])


class OfflineStorage(object):
    def __init__(self,
                 args, num_steps, max_replay_buffer_size,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards):

        self.args = args
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.max_replay_buffer_size = max_replay_buffer_size  # number of parallel processes
        self.step = 0  # keep track of current environment step
        self.seq_step = 0 # current sequence

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, max_replay_buffer_size, state_dim)
        self.next_state = torch.zeros(num_steps, max_replay_buffer_size, state_dim)


        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, max_replay_buffer_size, 1)
        self.rewards_normalised = torch.zeros(num_steps, max_replay_buffer_size, 1)

        self.done = torch.zeros(num_steps + 1, max_replay_buffer_size, 1)
        self.masks = torch.ones(num_steps + 1, max_replay_buffer_size, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, max_replay_buffer_size, 1)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, max_replay_buffer_size, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, max_replay_buffer_size, 1)
        self.returns = torch.zeros(num_steps + 1, max_replay_buffer_size, 1)
        self.trajectory_lens = torch.zeros(self.max_replay_buffer_size, dtype=torch.long)

    def init_state(self, state):
        self.prev_state[0, self.seq_step].copy_(state)

    def insert(self,
               state,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,
               ):


        self.prev_state[self.step + 1, self.seq_step].copy_(state)
        self.actions[self.step, self.seq_step] = actions.detach().clone()
        self.rewards_raw[self.step, self.seq_step].copy_(rewards_raw)
        self.next_state[self.step, self.seq_step].copy_(state)

        self.rewards_normalised[self.step, self.seq_step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step, self.seq_step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step, self.seq_step].copy_(value_preds.detach())
        self.masks[self.step + 1, self.seq_step].copy_(masks)
        self.bad_masks[self.step + 1, self.seq_step].copy_(bad_masks)
        self.done[self.step + 1, self.seq_step].copy_(done)
        
        if done:
            self.trajectory_lens[self.seq_step] = self.step + 1
            self.seq_step = (self.seq_step + 1 ) % self.max_replay_buffer_size
        self.step = (self.step + 1) % self.num_steps
    
    def random_batch(self, batch_size, replace=False):
        indices = np.random.choice(range(self.seq_step * self.num_steps), batch_size, replace=replace)
        indices = np.array([[idx % self.num_steps, idx // self.num_steps] for idx in indices])
        
        prev_obs = self.prev_state[indices[:, 0], indices[:, 1]]
        next_obs = self.next_state[indices[:, 0], indices[:, 1]]
        actions = self.actions[indices[:, 0], indices[:, 1]]
        rewards = self.rewards_raw[indices[:, 0], indices[:, 1]]
        rewards_norm = self.rewards_normalised[indices[:, 0], indices[:, 1]]
        
        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
            rewards.to(device), rewards_norm.to(device)
        
    def random_sequence(self, batch_size, replace=False):
        batch_size = min(self.seq_step, batch_size)
        indices = np.random.randint(0, self.seq_step, batch_size)
        
        trajectory_lens = self.trajectory_lens[indices]
        prev_obs = self.prev_state[:, indices, :]
        next_obs = self.next_state[:, indices, :]
        actions = self.actions[:, indices, :]
        rewards = self.rewards_raw[:, indices, :]
        rewards_norm = self.rewards_normalised[:, indices, :]
        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
            rewards.to(device), rewards_norm.to(device), trajectory_lens
        
        

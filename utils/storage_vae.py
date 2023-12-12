import imp
import numpy as np
import torch
import IPython

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorageVAE(object):
    def __init__(self, num_processes, max_trajectory_len, zero_pad, max_num_rollouts,
                 state_dim, action_dim, vae_buffer_add_thresh, task_dim):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  
        self.max_buffer_size = max_num_rollouts  
        self.insert_idx = 0  
        self.buffer_len = 0  

        
        self.max_traj_len = max_trajectory_len
        
        self.zero_pad = zero_pad

        
        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  
        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(device)  
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(device)  
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(device)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(device)
        if task_dim is not None:
            self.running_tasks = torch.zeros((num_processes, task_dim)).to(device)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):

        

        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            self.running_prev_state[self.curr_timestep[0]] = prev_state
            self.running_next_state[self.curr_timestep[0]] = next_state
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            if task is not None:
                self.running_tasks = task
            self.curr_timestep += 1
            already_inserted = True

        already_reset = False
        if done.sum() == self.num_processes:  

            
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    
                    if self.insert_idx + self.num_processes > self.max_buffer_size:
                        
                        self.buffer_len = self.insert_idx
                        
                        
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    
                    
                    self.prev_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_prev_state
                    self.next_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state
                    self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                    self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx:self.insert_idx+self.num_processes].shape
                        self.tasks[self.insert_idx:self.insert_idx+self.num_processes] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                    self.insert_idx += self.num_processes

            
            self.running_prev_state *= 0
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0

            already_reset = True

        if (not already_inserted) or (not already_reset):

            for i in range(self.num_processes):

                if not already_inserted:
                    self.running_prev_state[self.curr_timestep[i], i] = prev_state[i]
                    self.running_next_state[self.curr_timestep[i], i] = next_state[i]
                    self.running_rewards[self.curr_timestep[i], i] = rewards[i]
                    self.running_actions[self.curr_timestep[i], i] = actions[i]
                    if self.running_tasks is not None:
                        self.running_tasks[i] = task[i]
                    self.curr_timestep[i] += 1

                if not already_reset:
                    
                    if done[i]:
                        
                        if self.max_buffer_size > 0:
                            if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                                
                                if self.insert_idx + 1 > self.max_buffer_size:
                                    
                                    self.buffer_len = self.insert_idx
                                    
                                    
                                    self.insert_idx = 0
                                else:
                                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                                
                                
                                self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to('cpu')
                                self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                                self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                                self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                                if self.tasks is not None:
                                    self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                                self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                                self.insert_idx += 1

                        
                        self.running_prev_state[:, i] *= 0
                        self.running_next_state[:, i] *= 0
                        self.running_rewards[:, i] *= 0
                        self.running_actions[:, i] *= 0
                        if self.running_tasks is not None:
                            self.running_tasks[i] *= 0
                        self.curr_timestep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        batchsize = min(self.buffer_len, batchsize)
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(device)
        else:
            tasks = None

        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
               rewards.to(device), tasks, trajectory_lens

class RolloutStorageVAE(object):
    def __init__(self, num_processes, max_trajectory_len, zero_pad, max_num_rollouts,
                 state_dim, action_dim, vae_buffer_add_thresh, task_dim):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  
        self.max_buffer_size = max_num_rollouts  
        self.insert_idx = 0  
        self.buffer_len = 0  

        
        self.max_traj_len = max_trajectory_len
        
        self.zero_pad = zero_pad

        
        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  
        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(device)  
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(device)  
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(device)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(device)
        if task_dim is not None:
            self.running_tasks = torch.zeros((num_processes, task_dim)).to(device)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):
        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            self.running_prev_state[self.curr_timestep[0]] = prev_state
            self.running_next_state[self.curr_timestep[0]] = next_state
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            if task is not None:
                self.running_tasks = task
            self.curr_timestep += 1
            already_inserted = True

        already_reset = False
        if done.sum() == self.num_processes:  

            
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    
                    if self.insert_idx + self.num_processes > self.max_buffer_size:
                        
                        self.buffer_len = self.insert_idx
                        
                        
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    
                    
                    self.prev_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_prev_state
                    self.next_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state
                    self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                    self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx:self.insert_idx+self.num_processes].shape
                        self.tasks[self.insert_idx:self.insert_idx+self.num_processes] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                    self.insert_idx += self.num_processes

            
            self.running_prev_state *= 0
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0

            already_reset = True

        if (not already_inserted) or (not already_reset):

            for i in range(self.num_processes):

                if not already_inserted:
                    self.running_prev_state[self.curr_timestep[i], i] = prev_state[i]
                    self.running_next_state[self.curr_timestep[i], i] = next_state[i]
                    self.running_rewards[self.curr_timestep[i], i] = rewards[i]
                    self.running_actions[self.curr_timestep[i], i] = actions[i]
                    if self.running_tasks is not None:
                        self.running_tasks[i] = task[i]
                    self.curr_timestep[i] += 1

                if not already_reset:
                    
                    if done[i]:
                        
                        if self.max_buffer_size > 0:
                            if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                                
                                if self.insert_idx + 1 > self.max_buffer_size:
                                    
                                    self.buffer_len = self.insert_idx
                                    
                                    
                                    self.insert_idx = 0
                                else:
                                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                                
                                
                                self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to('cpu')
                                self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                                self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                                self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                                if self.tasks is not None:
                                    self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                                self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                                self.insert_idx += 1
                        self.running_prev_state[:, i] *= 0
                        self.running_next_state[:, i] *= 0
                        self.running_rewards[:, i] *= 0
                        self.running_actions[:, i] *= 0
                        if self.running_tasks is not None:
                            self.running_tasks[i] *= 0
                        self.curr_timestep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        batchsize = min(self.buffer_len, batchsize)
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(device)
        else:
            tasks = None

        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
               rewards.to(device), tasks, trajectory_lens
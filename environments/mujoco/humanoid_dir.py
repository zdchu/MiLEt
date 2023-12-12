"""
    Based on environment in PEARL:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces
import IPython

import random

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidDirEnv(HumanoidEnv):
    def __init__(self, task={}, max_episode_steps=200, n_tasks=2, randomize_tasks=True, sparse=False, theta_diff=0.707):
        self.set_task(np.array([0, 0, 0]))
        self.task_dim = 1
        self._max_episode_steps = max_episode_steps
        self.action_scale = 1 # Mujoco environment initialization takes a step, 

        self.sparse = sparse
        self.theta_diff = theta_diff
        super(HumanoidDirEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all() 
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.action_space.shape) # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))

        
    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])

        rescaled_action = action * self.action_scale # Scale the action from (-1, 1) to original.
        self.do_simulation(rescaled_action, self.frame_skip) 
        pos_after = mass_center(self.model, self.sim)[:2]

        # alive_bonus = 5.0
        alive_bonus = 0.0
        data = self.sim.data
        # goal_direction = (np.cos(self._goal), np.sin(self._goal))
        goal_direction = self._goal
        lin_vel_cost = 0.5 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        
        if self.sparse:
            if np.sum(goal_direction * (pos_after - pos_before)) / np.sqrt(np.sum((pos_after - pos_before) ** 2)) > self.theta_diff:
                pass
            else:
                lin_vel_cost = 0
        
        quad_ctrl_cost = 0.05 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        
        qpos = self.sim.data.qpos
        
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) 
        done = False

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost,
                                                    task=self.get_task(),
                                                    cluster=self.get_cluster())

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    # def get_all_task_idx(self):
    #     return range(len(self.tasks))

    # def reset_task(self, idx):
    #     self._task = self.tasks[idx]
    #     self._goal = self._task['goal'] # assume parameterization of task by single vector

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    
    def set_task(self, task):
        self._goal = task[:-1]
        self.cur_idx = task[-1:].astype(np.int)

    def get_task(self):
        return np.array([self._goal])
    
    def get_cluster(self):
        return np.array(self.cur_idx)

    def sample_tasks(self, num_tasks):
        a = []
        cluster = [0.25, 0.75, 1.25, 1.75]
        cnt = 0

        idxes = []
        ss = []
        while cnt < num_tasks:
            np.random.seed()
            idx = np.random.choice(range(4))
            s = np.random.normal(cluster[idx], 0.2)
            if s > 0 and s < 2:
                a.append(s * np.pi)
                cnt += 1
                idxes.append(idx)
                ss.append(s)
        self.idxes = idxes
        self.ss = ss
        return np.stack((np.cos(a), np.sin(a), self.idxes), axis=-1)
    
    def sample_cluster_task(self, num_tasks, seed=123):
        np.random.seed(seed)
        cnt = 0
        a = []
        r = []

        idxes = []
        ss = []

        cluster = [0.25, 0.75, 1.25, 1.75]
        while cnt < num_tasks:
            idx = cnt % 4 
            s = np.random.normal(cluster[idx], 0.2)
            
            a.append(s * np.pi)
            cnt += 1
            idxes.append(idx)
            ss.append(s)
        self.idxes = idxes
        self.ss = ss
        return np.stack((np.cos(a), np.sin(a)), axis=-1), np.stack(idxes)

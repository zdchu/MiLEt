import random
from telnetlib import IP

import numpy as np

from environments.mujoco.ant import AntEnv
import IPython


class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=50, goal_radius=0.8, sparse=False):
        self.goal_pos = [0., 0.]
        self.idxes = [0]
        self._goal_radius = goal_radius
        # self.cur_idx = [0]
        self.set_task(np.array([0, 0, 0]))
        
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2

        self._goal_radius = goal_radius
        self.sparse=sparse
        super(AntGoalEnv, self).__init__()
    
    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        if r < -self._goal_radius:
            r = 0
        else:
            r = r + self._goal_radius
        return r

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_pos = np.array(self.goal_pos)
        goal_reward = -np.sum(np.abs(xposafter[:2] - goal_pos))  # make it happy, not suicidal
        
        if self.sparse:
            goal_reward = self.sparsify_rewards(goal_reward)

        ctrl_cost = .01 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            sparse_reward=sparse_reward,
            task=self.get_task(),
            cluster=self.get_cluster(),
            position=xposafter[:2]
        )

    def sample_tasks(self, num_tasks):
        cnt = 0
        a = []
        r = []

        idxes = []
        ss = []

        cluster = [0.25, 0.75, 1.25, 1.75]
        while cnt < num_tasks:
            np.random.seed()
            idx = np.random.choice(range(4))
            s = np.random.normal(cluster[idx], 0.2)
            if s > 0 and s < 2:
                a.append(s * np.pi)
                cnt += 1
                idxes.append(idx)
                ss.append(s)
        r = 2
        self.idxes = idxes
        self.ss = ss
        return np.stack((r * np.cos(a), r * np.sin(a), self.idxes), axis=-1)


    def set_task(self, task):
        self.goal_pos = task[:2]
        self.cur_idx = task[2:].astype(np.int)


    def get_task(self):
        return np.array(self.goal_pos)
    
    def get_cluster(self):
        return np.array(self.cur_idx)
    
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
        r = 2
        self.idxes = idxes
        self.ss = ss
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1), np.stack(idxes)


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])

import random

import numpy as np

from environments.mujoco.rand_param_envs.gym.core import Env
from environments.mujoco.rand_param_envs.gym.envs.mujoco import MujocoEnv
import IPython


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass


class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs):
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.save_parameters()
        self.task_dim = self.rand_param_dim

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []
        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}
            if 'body_mass' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.body_mass.shape))]
                body_mass_multiplyers = np.array(1.5) ** np.array(rand_params).reshape(self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.body_inertia.shape))]
                body_inertia_multiplyers = np.array(1.5) ** np.array(rand_params).reshape(self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.dof_damping.shape))]
                dof_damping_multipliers = np.array(1.3) ** np.array(rand_params).reshape(self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.geom_friction.shape))]
                dof_damping_multipliers = np.array(1.5) ** np.array(rand_params).reshape(self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            param_sets.append(new_params)

        return param_sets
    
    def sample_cluster_task(self, n_tasks, clusters=4):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []
        cluster = [3, 6, 9, 12]
        
        for idx in range(n_tasks):
            cls = idx % 4
            # body mass -> one multiplier for all body parts
            new_params = {}
            # hopper mu: 3, sigma: 1.5
            
            # mu = 3
            # sigma = 1.5

            mu = 6
            sigma = 1
            if 'body_mass' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.body_mass.shape))]
                body_mass_multiplyers = np.array(1.5) ** np.array(rand_params).reshape(self.model.body_mass.shape)
                
                body_mass_multiplyers = np.random.normal(mu, sigma, np.prod(self.model.body_mass.shape)).reshape(self.model.body_mass.shape)
                body_mass_multiplyers[body_mass_multiplyers < 0] = 0.1
                
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.body_inertia.shape))]
                body_inertia_multiplyers = np.array(1.5) ** np.array(rand_params).reshape(self.model.body_inertia.shape)
                
                body_inertia_multiplyers = np.random.normal(mu, sigma, np.prod(self.model.body_inertia.shape)).reshape(self.model.body_inertia.shape)
                body_inertia_multiplyers[body_inertia_multiplyers < 0] = 0.1
                
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.dof_damping.shape))]
                dof_damping_multipliers = np.array(1.3) ** np.array(rand_params).reshape(self.model.dof_damping.shape)
                
                dof_damping_multipliers = np.random.normal(mu, sigma, np.prod(self.model.dof_damping.shape)).reshape(self.model.dof_damping.shape)
                dof_damping_multipliers[dof_damping_multipliers < 0] = 0.1
                
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                rand_params = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for _ in
                               range(np.prod(self.model.geom_friction.shape))]
                dof_damping_multipliers = np.array(1.5) ** np.array(rand_params).reshape(self.model.geom_friction.shape)
                
                dof_damping_multipliers = np.random.normal(mu, sigma, np.prod(self.model.geom_friction.shape)).reshape(self.model.geom_friction.shape)
                dof_damping_multipliers[dof_damping_multipliers < 0] = 0.1
                
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            for i, param in enumerate(self.rand_params):
                if cls != i:
                    new_params[param] = self.init_params[param]

            new_params['cluster_idx'] = cls
            param_sets.append(new_params)
        return param_sets

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            new_task = {}
            start_idx = 0
            for k in self.curr_params.keys():
                end_idx = np.prod(self.curr_params[k].shape)
                new_task[k] = task[start_idx:start_idx+end_idx].reshape(self.curr_params[k].shape)
                start_idx += end_idx
            task = new_task
        for param in self.rand_params:
            param_variable = getattr(self.model, param)
            assert param_variable.shape == task[param].shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, task[param])
        self.curr_params = task
        self.cluster_idx = task['cluster_idx']

    def get_cluster(self):
        if hasattr(self, 'cluster_idx'):
            cls = [self.cluster_idx]
        else:
            cls = [0]
        return cls

    def get_task(self):
        if hasattr(self, 'curr_params'):
            task = self.curr_params
            task = np.concatenate([task[k].reshape(-1) for k in self.rand_params])
        else:
            task = np.zeros(self.rand_param_dim)
        return task

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.curr_params = self.init_params

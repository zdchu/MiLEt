import collections
from copy import deepcopy
from enum import Flag
import imp
import os
from socket import IP_ADD_MEMBERSHIP
from telnetlib import IP
import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
from algorithms.a2c import A2C
from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from gmvae import GMVAE
import IPython
import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from environments.env_utils.running_mean_std import RunningMeanStd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def exp_decay(x, scale=0.05, b=0.1, a=0.6):
    return b - a * np.exp(- scale * x)


class Explore_MetaLearner:
    def __init__(self, args):

        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  tasks=None, sparse=args.sparse_reward
                                  )
        self.prepare_mujoco_tasks(args)
        if self.args.single_task_mode:
            # get the current tasks (which will be num_process many different tasks)
            self.train_tasks = self.envs.get_task()
            # set the tasks to the first task (i.e. just a random task)
            self.train_tasks[1:] = self.train_tasks[0]
            # make it a list
            self.train_tasks = [t for t in self.train_tasks]
            # re-initialise environments with those tasks
            self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                      gamma=args.policy_gamma, device=device,
                                      episodes_per_task=self.args.max_rollouts_per_task,
                                      normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                      tasks=self.train_tasks, sparse=args.sparse_reward
                                      )
            # save the training tasks so we can evaluate on the same envs later
            utl.save_obj(self.train_tasks, self.logger.full_output_folder, "train_tasks")

        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        self.vae = GMVAE(self.args, self.logger, lambda: self.iter_idx)
        
        self.policy_storage = self.initialise_policy_storage()
        self.exp_policy_storage = self.initialise_policy_storage()

        self.policy = self.initialise_policy(vae_optimizer=self.vae.optimiser_vae, 
                                    pass_latent_to_policy=self.args.pass_latent_to_policy, 
                                        entropy_coef=self.args.policy_entropy_coef, lr_policy=self.args.lr_policy)
        self.exp_policy = self.initialise_policy(vae_optimizer=self.vae.optimiser_vae, pass_latent_to_policy=self.args.pass_latent_to_policy,
                                        entropy_coef=self.args.exp_policy_entropy_coef, lr_policy=self.args.lr_exp_policy)
        self.target_policy = deepcopy(self.policy)

        self.k = args.k
        self.entropy_prior = -np.log(1/self.k)

    def prepare_mujoco_tasks(self, args):
        num_tasks = self.args.num_tasks
        if not os.path.exists('./saved_tasks/'+ args.env_name + '/tasks_{}.npy'.format(num_tasks)) and not os.path.exists('./saved_tasks/'+ args.env_name + '/tasks_{}.pkl'.format(num_tasks)):
            if 'Rand' not in args.env_name:
                tasks, clusters = self.envs.venv.venv.sample_cluster_task(num_tasks, 123)        
                tasks = np.array(tasks)
                clusters = np.array(clusters)
                tasks = np.concatenate((tasks, clusters[...,  np.newaxis]), 1)  
                try:
                    os.mkdir('./saved_tasks/'+ args.env_name)
                except:
                    pass
                
                np.save('./saved_tasks/'+ args.env_name + '/tasks_{}.npy'.format(len(tasks)), tasks)  
            else:
                tasks = self.envs.venv.venv.sample_cluster_task(num_tasks, 123)  
                try:
                    os.mkdir('./saved_tasks/'+ args.env_name)
                except:
                    pass
                with open('./saved_tasks/'+ args.env_name + '/tasks_{}.pkl'.format(len(tasks)), 'wb') as f:
                    pickle.dump(tasks, f)
        else:
            try:
                tasks = np.load('./saved_tasks/'+ args.env_name + '/tasks_{}.npy'.format(num_tasks))
            except:
                with open('./saved_tasks/'+ args.env_name + '/tasks_{}.pkl'.format(num_tasks), 'rb') as f:
                    tasks = pickle.load(f)
        self.train_tasks = tasks[:-32]
        self.test_tasks = tasks[-32:]
        self.task_num = len(self.train_tasks)
        
    
    def load_model(self):
        save_path = os.path.join(self.logger.full_output_folder, 'models')
        self.exp_policy.actor_critic = torch.load(os.path.join(save_path, "exp_policy.pt"))
        self.policy.actor_critic = torch.load(os.path.join(save_path, 'policy.pt'))
        self.vae.encoder = torch.load(os.path.join(save_path, 'encoder.pt'))
        self.vae.reward_decoder = torch.load(os.path.join(save_path, 'reward_decoder.pt'))


    def initialise_policy_storage(self):
        return OnlineStorage(args=self.args,
                             num_steps= self.envs._max_episode_steps,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=self.args.latent_dim,
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             action_space=self.args.action_space,
                             hidden_size=self.args.encoder_gru_hidden_size,
                             normalise_rewards=self.args.norm_rew_for_policy,
                             )


    def initialise_policy(self, vae_optimizer, pass_latent_to_policy, entropy_coef, lr_policy):
        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
        ).to(device)

        # initialise policy trainer
        if self.args.policy == 'a2c':
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                optimiser_vae=vae_optimizer,
                lr=lr_policy,
                eps=self.args.policy_eps,
            )

        elif self.args.policy == 'ppo':
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
                optimiser_vae=vae_optimizer,
            )
        else:
            raise NotImplementedError
        return policy

    def train(self):
        """ Main Meta-Training loop """
        if self.args.load_model:
            self.load_model()

        start_time = time.time()
        begin_update = False


        sampled_idx = np.random.choice(range(len(self.train_tasks)), self.args.num_processes, replace=False)
        prev_state, belief, task = utl.reset_env(self.envs, self.args, tasks=[self.train_tasks[task_id] for task_id in sampled_idx])

        # insert initial observation / embeddings to rollout storage
        self.exp_policy_storage.prev_state[0].copy_(prev_state)
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time, None, None)
            
        pred_clusters = collections.deque(maxlen=200)
        gt_clusters = collections.deque(maxlen=200)

        explore_train_stats = None
        for self.iter_idx in range(self.num_updates):
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory()
            
            reward_ent = torch.ones((1, self.args.num_processes)).to(device) * self.entropy_prior
            curr_dist = torch.ones(self.args.num_processes, self.k).to(device) * 1 / self.k
            curr_var = torch.ones(self.args.num_processes, self.args.latent_dim).to(device)

            assert len(self.exp_policy_storage.latent_mean) == 0
            self.exp_policy_storage.latent_samples.append(latent_sample.clone())
            self.exp_policy_storage.latent_mean.append(latent_mean.clone())
            self.exp_policy_storage.latent_logvar.append(latent_logvar.clone())

            for episode_idx in range(1):
                for step in range(self.envs._max_episode_steps):
                    with torch.no_grad():
                        value, action = utl.select_action(
                            args = self.args,
                            policy=self.exp_policy,
                            state=prev_state,
                            belief=belief,
                            task=task,
                            deterministic=False,
                            latent_sample=latent_sample,
                            latent_mean=latent_mean,
                            latent_logvar=latent_logvar
                        )

                    [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action, self.args)
                    
                    done_mdp = [info['done_mdp'] for info in infos]
                    done_mdp = torch.from_numpy(np.array(done_mdp, dtype=int)).to(device).float().view((-1, 1))
                    
                    done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                    masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                    bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                    with torch.no_grad():
                        latent_sample, latent_mean, latent_logvar, new_hidden_state = utl.update_encoding(
                            encoder=self.vae.encoder,
                            next_obs=next_state,
                            action=action,
                            reward=rew_raw,
                            done=done,
                            hidden_state=hidden_state,
                            cluster=[i['cluster'][0] for i in infos]    
                        )
                    
                    if self.args.vae == 'gmvae':
                        pred_cluster, new_reward_ent, next_dist, next_var = self.vae.encoder.get_cluster(actions=action.float(),
                                                            states=next_state,
                                                            rewards=rew_raw,
                                                            hidden_state=hidden_state)
                        gt_cluster = [i['cluster'][0] for i in infos]
                    
                    kl_reward = -(curr_dist * (torch.log(curr_dist + 1e-8) - torch.log(next_dist + 1e-8))).sum(2)

                    self.exp_policy_storage.next_state[step] = next_state.clone()
                    done_indices = np.argwhere(done.cpu().flatten()).flatten()
                    if not (self.args.disable_decoder and self.args.disable_kl_term):
                        cluster = torch.tensor([i['cluster'][0] for i in infos]) 
                        cluster = F.one_hot(cluster, 4)
                        try:
                            self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    cluster)
                        except:
                            IPython.embed()
                    
                    if len(done_indices) > 0:
                        sampled_idx = np.random.choice(range(len(self.train_tasks)), self.args.num_processes, replace=False)
                        next_state, belief, task = utl.reset_env(self.envs, self.args,
                                                             indices=done_indices, state=next_state,
                                                             tasks=[self.train_tasks[task_id] for task_id in sampled_idx])

                    ent_reward_lambda = exp_decay(self.envs._max_episode_steps - step, b=0.1, a=0.1, scale=0.1)
                    const_reward_lambda = -exp_decay(self.envs._max_episode_steps - step, scale=0.1, b=0.1, a=0.2)
                    task_ent_lambda = 0
                    
                    self.exp_policy_storage.insert(
                        state=next_state,
                        belief=belief,
                        task=task,
                        actions=action,
                        rewards_raw= rew_raw,  
                        rewards_normalised= rew_normalised,
                        value_preds=value,
                        masks=masks_done,
                        bad_masks=bad_masks,
                        done=done,
                        hidden_states=new_hidden_state,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                        rewards_ent= ent_reward_lambda * (reward_ent - new_reward_ent).T + const_reward_lambda * kl_reward.T
                        )

                    hidden_state = new_hidden_state
                    prev_state = next_state
                    reward_ent = new_reward_ent
                    curr_dist = next_dist
                    curr_var = next_var
                    self.frames += self.args.num_processes
            
            if begin_update:
                explore_train_stats = self.explore_update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar)
                                              
                exp_run_stats = [action, self.exp_policy_storage.action_log_probs, value]

            assert len(self.policy_storage.latent_mean) == 0
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())

            for episode_idx in range(1):
                for step in range(self.envs._max_episode_steps):
                    with torch.no_grad():
                        value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )
                    [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action, self.args)

                    done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                    masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                    bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                    with torch.no_grad():
                        latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                            encoder=self.vae.encoder,
                            next_obs=next_state,
                            action=action,
                            reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state,
                        cluster=[i['cluster'][0] for i in infos])

                    if not (self.args.disable_decoder and self.args.disable_kl_term):
                        cluster = torch.tensor([i['cluster'][0] for i in infos])
                        cluster = F.one_hot(cluster, 4)
                        self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    cluster)

                    # add the obs before reset to the policy storage
                    self.policy_storage.next_state[step] = next_state.clone()

                    # reset environments that are done
                    done_indices = np.argwhere(done.cpu().flatten()).flatten()
                    if len(done_indices) > 0:
                        sampled_idx = np.random.choice(range(len(self.train_tasks)), self.args.num_processes, replace=False)                                
                        next_state, belief, task = utl.reset_env(self.envs, self.args,
                                                             indices=done_indices, state=next_state,
                                                             tasks=[self.train_tasks[task_id] for task_id in sampled_idx])
                        
                        

                    self.policy_storage.insert(
                        state=next_state,
                        belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done,
                    hidden_states=hidden_state,
                    latent_sample=latent_sample,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar
                    )

                    prev_state = next_state
                    curr_dist = next_dist
                    self.frames += self.args.num_processes

            if self.args.precollect_len <= self.frames:
                if self.iter_idx % 50 == 0 and self.args.vae == 'gmvae':
                    self.vae.prior_update()

                if self.args.pretrain_len > self.iter_idx:
                    for p in range(self.args.num_vae_updates_per_pretrain):
                        self.vae.compute_vae_loss(update=True,
                                                  pretrain_index=self.iter_idx * self.args.num_vae_updates_per_pretrain + p)
                else:
                    begin_update = True
                    exploit_train_stats = self.exploit_update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar)
                    run_stats = [action, self.policy_storage.action_log_probs, value]
                    with torch.no_grad():
                        if explore_train_stats is not None:
                            self.log(run_stats, exploit_train_stats, start_time, explore_train_stats, exp_run_stats)
            self.policy_storage.after_update()
            self.exp_policy_storage.after_update()
        self.envs.close()

    def encode_running_trajectory(self):
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(actions=act,
                                                                                                       states=next_obs,
                                                                                                       rewards=rew,
                                                                                                       hidden_state=None,
                                                                                                       return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        if self.args.stack_rnn:
            hidden_state = ((torch.stack([all_hidden_states[0][lens[i]][i] for i in range(len(lens))])).to(device),
                            (torch.stack([all_hidden_states[1][lens[i]][i] for i in range(len(lens))])).to(device)
            )
        else:
            hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)
        return latent_sample, latent_mean, latent_logvar, hidden_state

    def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar, policy):
        latent = utl.get_latent_for_policy(self.args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        return policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()

    def exploit_update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar,
                                            policy=self.policy)

            # compute returns for current rollouts
            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            # update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0

            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats

    def explore_update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar,
                                            policy=self.exp_policy)

            # compute returns for current rollouts
            self.exp_policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            # update agent (this will also call the VAE update!)
            policy_train_stats = self.exp_policy.update(
                policy_storage=self.exp_policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=None)
        else:
            policy_train_stats = 0, 0, 0, 0

            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                pass

        return policy_train_stats


    def log(self, run_stats, train_stats, start_time, explore_train_stats, exp_run_stats):

        # --- visualise behaviour of policy ---

        if (self.iter_idx + 1) % self.args.vis_interval == 0 and self.args.vis_behavior:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         encoder=self.vae.encoder,
                                         reward_decoder=self.vae.reward_decoder,
                                         state_decoder=self.vae.state_decoder,
                                         task_decoder=self.vae.task_decoder,
                                         compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
                                         compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
                                         compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
                                         compute_kl_loss=self.vae.compute_kl_loss,
                                         tasks=self.train_tasks,
                                         )

        # --- evaluate policy ----

        if (self.iter_idx + 1) % self.args.eval_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            
            returns_per_episode = 0
            exp_states = []
            for i in range(0, len(self.test_tasks), self.args.num_processes):
                test_tasks = self.test_tasks[i:i+self.args.num_processes]
                if len(test_tasks) < self.args.num_processes:
                    break
                returns = utl_eval.exp_evaluate(args=self.args,
                                                    exp_policy=self.exp_policy,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    encoder=self.vae.encoder,
                                                    iter_idx=self.iter_idx,
                                                    tasks=test_tasks,
                                                    logdir=self.logger.full_output_folder
                                                    )

                returns_per_episode += returns
                gts.append(gt)
                preds.append(pred)
                exp_states.append(exp_state)
            returns_per_episode /= len(self.test_tasks) / self.args.num_processes

            # log the return avg/std across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (train): {returns_avg[-1].item()} \n"
                  )

        # --- save models ---

        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.exp_policy.actor_critic, os.path.join(save_path, f"exp_policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder{idx_label}.pt"))

                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
                # TODO: grab from policy and save?
                # if self.args.norm_obs_for_policy:
                #     obs_rms = self.envs.venv.obs_rms
                #     utl.save_obj(obs_rms, save_path, f"env_obs_rms{idx_label}")

        # --- log some other things ---

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())

                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
        
        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (explore_train_stats is not None):

            self.logger.add('environment/exp_state_max', self.exp_policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/exp_state_min', self.exp_policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/exp_rew_max', self.exp_policy_storage.rewards_raw.max(), self.iter_idx)

            self.logger.add('environment/ent_rew_max', self.exp_policy_storage.reward_ent.max(), self.iter_idx)
            self.logger.add('environment/ent_rew_min', self.exp_policy_storage.reward_ent.min(), self.iter_idx)

            self.logger.add('environment/exp_rew_min', self.exp_policy_storage.rewards_raw.min(), self.iter_idx)

            self.logger.add('exp_policy_losses/value_loss', explore_train_stats[0], self.iter_idx)
            self.logger.add('exp_policy_losses/action_loss', explore_train_stats[1], self.iter_idx)
            self.logger.add('exp_policy_losses/dist_entropy', explore_train_stats[2], self.iter_idx)
            self.logger.add('exp_policy_losses/sum', explore_train_stats[3], self.iter_idx)

            self.logger.add('exp_policy/action', exp_run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.exp_policy.actor_critic, 'logstd'):
                self.logger.add('exp_policy/action_logstd', self.exp_policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('exp_policy/action_logprob', exp_run_stats[1].mean(), self.iter_idx)
            self.logger.add('exp_policy/value', exp_run_stats[2].mean(), self.iter_idx)

from telnetlib import IP
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from .layers import GumbelSoftmax, Gaussian
import IPython

from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MixRNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 k=5
                 ):
        super(MixRNNEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]
        
        self.k = k
        self.inference_qyx = nn.Sequential(
            nn.Linear(curr_input_dim, 32),
            nn.ReLU(),
        )   
        self.gumbel_layer = GumbelSoftmax(32, self.k)
        self.inference_qzyx = nn.Sequential(
            nn.Linear(curr_input_dim + self.k, 32),
            nn.ReLU(),
            Gaussian(32, latent_dim)
        )
        
        self.y_mu = nn.Sequential(
                        nn.Linear(self.k, 32), 
                        nn.ReLU(),
                        nn.Linear(32, latent_dim)
                    )
        
        self.y_var = nn.Sequential(
                        nn.Linear(self.k, 32), 
                        nn.ReLU(),
                        nn.Linear(32, latent_dim)
                    )
                    

        

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            '''
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            '''
            std = torch.sqrt(logvar + 1e-10)
            noise = torch.randn_like(std)
            z = mu + noise * std
            return z
            # return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True, test=False, cluster=None):
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        
        y_hidden = self.inference_qyx(h) 
        logits, prob, y = self.gumbel_layer(y_hidden, test=test)

        concat = torch.cat((h, y), -1)
        latent_mean, latent_logvar, latent_sample = self.inference_qzyx(concat)
        return latent_sample, latent_mean, latent_logvar, hidden_state, logits, prob, y
    
    def get_cluster(self, actions, states, rewards, hidden_state, return_entropy=False):
        actions = utl.squash_action(actions, self.args)
        
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))
        
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))
        output, _ = self.gru(h, hidden_state)
        gru_h = output.clone()

        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        y_hidden = self.inference_qyx(gru_h) 
        logits, prob, y = self.gumbel_layer(y_hidden)

        concat = torch.cat((gru_h, y), -1)
        _, var, _ = self.inference_qzyx(concat)

        return prob.max(2)[1][0], -torch.sum(prob * torch.log(prob), 2).detach(), prob.detach(), var.detach()[0]

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=False, 
                    sample=True, detach_every=None, return_logits=False, test=False, cluster=None, use_prior=False):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        
        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))
        
        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state, logits_prior, prob_prior, y_prior = self.prior(actions.shape[1], cluster=cluster)
            hidden_state = prior_hidden_state.clone()

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            output, _ = self.gru(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i*detach_every:i*detach_every+detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs

        y_hidden = self.inference_qyx(gru_h) 
        logits, prob, y = self.gumbel_layer(y_hidden, test=test)
        
        concat = torch.cat((gru_h, y), -1)
        # concat = torch.cat((gru_h, cluster), -1)
        latent_mean, latent_logvar, latent_sample = self.inference_qzyx(concat)
        
        if return_prior:
            prob = torch.cat((prob_prior, prob))
            logits = torch.cat((logits_prior, logits))
            y = torch.cat((y_prior, y))
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))

        if return_logits:
            output = {'mean': latent_mean, 'var': latent_logvar, 'gaussian': latent_sample, 
              'logits': logits, 'prob_cat': prob, 'categorical': y, 'hidden': output}
            return output

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]
        
        entropy = -torch.sum(prob * torch.log(prob), 2)
        return latent_sample, latent_mean, latent_logvar, output # , entropy.detach()


class StackedRNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 k=5
                 ):
        super(StackedRNNEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.cgru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        self.tgru = nn.GRU(input_size=curr_input_dim + hidden_size, 
                                hidden_size=hidden_size,
                                num_layers=1)

        for name, param in self.cgru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
        for name, param in self.tgru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_cgru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_cgru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]
        
        curr_input_dim = hidden_size
        self.fc_after_tgru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_tgru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]
        
        self.k = k
        self.inference_qyx = nn.Sequential(
            nn.Linear(curr_input_dim, 32),
            nn.ReLU(),
        )   
        self.gumbel_layer = GumbelSoftmax(32, self.k)
        self.inference_qzyx = nn.Sequential(
            nn.Linear(curr_input_dim + self.k, 32),
            nn.ReLU(),
            Gaussian(32, latent_dim)
        )

        self.y_mu = nn.Sequential(
                        nn.Linear(self.k, 32), 
                        nn.ReLU(),
                        nn.Linear(32, latent_dim)
                    )
        
        self.y_var = nn.Sequential(
                        nn.Linear(self.k, 32), 
                        nn.ReLU(),
                        nn.Linear(32, latent_dim)
                    )
                    

        

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            '''
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            '''
            std = torch.sqrt(logvar + 1e-10)
            noise = torch.randn_like(std)
            z = mu + noise * std
            return z
            # return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        cluster_hidden, task_hidden = hidden_state[0], hidden_state[1]
        if cluster_hidden.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        cluster_hidden = cluster_hidden * (1 - done)
        task_hidden = task_hidden * (1 - done)
        return (cluster_hidden, task_hidden)

    def prior(self, batch_size, sample=True, test=False, cluster=None):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        cluster_hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        c_h = cluster_hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_cgru)):
            c_h = F.relu(self.fc_after_cgru[i](c_h))

        # outputs
        
        y_hidden = self.inference_qyx(c_h) 
        logits, prob, y = self.gumbel_layer(y_hidden, test=test)
        
        task_hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)
        t_h = task_hidden_state
        for i in range(len(self.fc_after_tgru)):
            t_h = F.relu(self.fc_after_tgru[i](t_h))

        concat = torch.cat((t_h, y), -1)
        latent_mean, latent_logvar, latent_sample = self.inference_qzyx(concat)
        return latent_sample, latent_mean, latent_logvar, (cluster_hidden_state, task_hidden_state), logits, prob, y
    
    def get_cluster(self, actions, states, rewards, hidden_state, return_entropy=False):
        actions = utl.squash_action(actions, self.args)
        
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        cluster_hidden, task_hidden = hidden_state[0], hidden_state[1]
        cluster_hidden = cluster_hidden.reshape((-1, *cluster_hidden.shape[-2:]))
        task_hidden = task_hidden.reshape((-1, *task_hidden.shape[-2:]))

        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))
        
        output_cluster, _ = self.cgru(h, cluster_hidden)
        t_input = torch.cat((h, output_cluster), 2)
        output_task, _ = self.tgru(t_input, task_hidden)

        gru_h = output_cluster.clone()

        for i in range(len(self.fc_after_cgru)):
            gru_h = F.relu(self.fc_after_cgru[i](gru_h))
        # outputs
        # logits, prob, y = self.inference_qyx(gru_h)
        y_hidden = self.inference_qyx(gru_h) 
        logits, prob, y = self.gumbel_layer(y_hidden)

        gru_th = output_task.clone()
        for i in range(len(self.fc_after_tgru)):
            gru_th = F.relu(self.fc_after_tgru[i](gru_th))

        concat = torch.cat((gru_th, y), -1)
        _, var, _ = self.inference_qzyx(concat)
        return prob.max(2)[1][0], -torch.sum(prob * torch.log(prob), 2).detach(), prob.detach(), var.detach()[0]

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=False, 
                    sample=True, detach_every=None, return_logits=False, test=False, cluster=None, use_prior=False):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        
        if hidden_state is not None:
            cluster_hidden, task_hidden = hidden_state[0], hidden_state[1]
        else:
            cluster_hidden, task_hidden = None, None

        if cluster_hidden is not None and task_hidden is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            cluster_hidden = cluster_hidden.reshape((-1, *cluster_hidden.shape[-2:]))
            task_hidden = task_hidden.reshape((-1, *task_hidden.shape[-2:]))
        
        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden, logits_prior, prob_prior, y_prior = self.prior(actions.shape[1], cluster=cluster)
            cluster_hidden = prior_hidden[0].clone()
            task_hidden = prior_hidden[1].clone()

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            output_cluster, _ = self.cgru(h, cluster_hidden)
            t_input = torch.cat((h, output_cluster), 2)
            output_task, _ = self.tgru(t_input, task_hidden)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i*detach_every:i*detach_every+detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_ch = output_cluster.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_cgru)):
            gru_ch = F.relu(self.fc_after_cgru[i](gru_ch))

        # outputs

        y_hidden = self.inference_qyx(gru_ch) 
        logits, prob, y = self.gumbel_layer(y_hidden, test=test)

        gru_th = output_task.clone()
        for i in range(len(self.fc_after_tgru)):
            gru_th = F.relu(self.fc_after_tgru[i](gru_th))

        concat = torch.cat((gru_th, y), -1)
        latent_mean, latent_logvar, latent_sample = self.inference_qzyx(concat)
        
        if return_prior:
            prob = torch.cat((prob_prior, prob))
            logits = torch.cat((logits_prior, logits))
            y = torch.cat((y_prior, y))
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output_cluster = torch.cat((prior_hidden[0], output_cluster))
            output_task = torch.cat((prior_hidden[1], output_task))

        if return_logits:
            output = {'mean': latent_mean, 'var': latent_logvar, 'gaussian': latent_sample, 
              'logits': logits, 'prob_cat': prob, 'categorical': y, 'cluster_hidden': output_cluster, 'task_hidden': output_task}
            return output

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]
        
        return latent_sample, latent_mean, latent_logvar, (output_cluster, output_task)

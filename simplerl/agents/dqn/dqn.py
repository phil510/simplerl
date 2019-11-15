import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..base_agents import LearningAgent
from ..agent_mixins import (ExplorationNoiseMixin, 
                            ReplayBufferMixin,
                            LocalBufferMixin)
from ..common.torch_utils import calc_n_step_returns, to_tensors
from ..common.schedulers import ExponentialScheduler

class DQNAgent(LocalBufferMixin, ReplayBufferMixin, LearningAgent):
    
    compatible_states = ['continuous']
    compatible_actions = ['discrete']
    
    def __init__(self, env_fn = None, 
                 model_fn = None,
                 n_actors = 1,
                 n_actions = None,
                 epsilon = ExponentialScheduler(1.0, 1e-3, .995),
                 double_q = False,
                 gamma = 0.99,
                 batch_size = 64,
                 n_steps = 1,
                 replay_memory = 100000,
                 use_per = False,
                 alpha = 0.6,
                 beta = lambda: 0.4,
                 replay_start = 1000,
                 param_update_freq = 1000,
                 buffer_update_freq = 1000,
                 optimizer = optim.Adam,
                 learning_rate = 1e-3,
                 weight_decay = 1e-4,
                 clip_gradients = None,
                 update_freq = 1):
        
        super().__init__(env_fn = env_fn,
                         model_fn = model_fn,
                         n_actors = n_actors,
                         replay_memory = replay_memory,
                         use_per = use_per,
                         alpha = alpha,
                         beta = beta,
                         replay_start = replay_start,
                         trajectory_len = n_steps)
        
        assert (n_actions >= 1), 'TODO'
        
        # create online and target networks
        self.online_network = self._model_fn()
        self.target_network = self._model_fn()
        
        assert (hasattr(self.online_network, 'forward')), 'TODO'
        
        self.register_torch_obj(self.online_network, 'online_network')
        self.register_torch_obj(self.target_network, 'target_network')
        self.online_network.eval()
        self.target_network.eval()
        
        # create the optimizers for the online_network
        self.optimizer = optimizer(self.online_network.parameters(), 
                                   lr = learning_rate,
                                   weight_decay = weight_decay)
        self.clip_gradients = clip_gradients
        self.register_torch_obj(self.optimizer, 'optimizer')
        
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.double_q = double_q
        self.gamma = gamma
        self.batch_size = batch_size
        self.param_update_freq = param_update_freq
        self.buffer_update_freq = buffer_update_freq
        self.update_freq = update_freq
        
        self.eval()
        
    @property 
    def n_steps(self):
        return self._trajectory_len
        
    def target_update(self):
        if (self.current_step % self.param_update_freq == 0):
            for target, online in zip(self.target_network.parameters(), 
                                      self.online_network.parameters()):
                target.detach_()
                target.copy_(online)
        
        # this is for things like batch norm and other PyTorch objects
        # that have buffers and/or instead of learnable parameters
        if (self.current_step % self.param_update_freq == 0):
            for target, online in zip(self.target_network.buffers(), 
                                      self.online_network.buffers()):
                # detach is probably unnecessary since buffers are not learnable
                target.detach_() 
                target.copy_(online)
            
    def ready_to_update(self):
        return ((self.current_step >= self.replay_start) and
                (self.current_step % self.update_freq == 0) and
                self.training)
    
    def estimate_q(self, obs, target_network = False):
        if target_network:
            Q_s = self.target_network(obs)
        else:
            Q_s = self.online_network(obs)
        
        return Q_s
    
    def action(self, obs):
        if self._rng.rand() < self.epsilon():
            action = self._rng.randint(0, self.n_actions, 
                                       size = (obs.shape[0], ))
        else:
            obs = torch.as_tensor(obs, dtype = torch.float32,
                                  device = self._device)
            
            with torch.no_grad():
                Q_s = self.estimate_q(obs).cpu().numpy()
            
            action = np.argmax(Q_s, axis = 1)
        
        return action
        
    def update_target(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            if self.n_steps > 1:
                Q_max = []
                for i in range(self.n_steps):
                    if self.double_q:
                        online_Q = self.estimate_q(next_obs[:, i],
                                                   target_network = False)
                        _, max_action = torch.max(online_Q, dim = 1)
                        target_Q = self.estimate_q(next_obs[:, i], 
                                                   target_network = True)
                        Q = target_Q.gather(1, max_action.unsqueeze(1))
                    
                    else:
                        target_Q = self.estimate_q(next_obs[:, i],
                                                   target_network = True)
                        Q, _ = torch.max(target_Q, dim = 1)
                        Q = Q.unsqueeze(1)
                    
                    Q_max.append(Q)
                
                Q_max = torch.stack(Q_max, dim = 1)
                
            else:
                if self.double_q:
                    online_Q = self.estimate_q(next_obs,
                                               target_network = False)
                    _, max_action = torch.max(online_Q, dim = 1)
                    target_Q = self.estimate_q(next_obs,
                                               target_network = True)
                    Q_max = target_Q.gather(1, max_action.unsqueeze(1))
                
                else:
                    target_Q = self.estimate_q(next_obs,
                                               target_network = True)
                    Q_max, _ = torch.max(target_Q, dim = 1)
                    Q_max = Q_max.unsqueeze(1)
                    
        # check if reward is 1 dimensional, if it is, we need to 
        # insert the seq_len dimension at dim 1 for the calc_n_step_returns
        # function
        if (reward.dim() == 1):
            reward = reward.unsqueeze(1)
            terminal = terminal.unsqueeze(1)
            Q_max = Q_max.unsqueeze(1)

        update_target = calc_n_step_returns(reward, Q_max, terminal,
                                            gamma = self.gamma, 
                                            n_steps = self.n_steps,
                                            seq_model = False)
        update_target = update_target.squeeze(1)
        
        assert (update_target.shape[0] == obs.shape[0]), 'TODO'
        assert (not update_target.requires_grad), 'TODO'

        return update_target
        
    def add_to_memory(self, obs, action, reward, next_obs, terminal,
                      *args, **kwargs): 
        assert (obs.shape[0] == self._n_actors), 'TODO'
        if (not hasattr(self, 'local_buffer')):
            for i in range(self._n_actors):
                experience = to_tensors(obs[i], action[i], reward[i], 
                                        next_obs[i], terminal[i],
                                        dtype = torch.float32, 
                                        device = self._device)
                self.replay_buffer.add(experience)
            
        else:
            for i, buffer in enumerate(self.local_buffer):
                if (len(buffer) == self.n_steps):
                    trajectory = buffer.get_trajectory()
                    experience = to_tensors(*trajectory,
                                            dtype = torch.float32,
                                            device = self._device)
                    self.replay_buffer.add(experience)
                    buffer.reset()
                
                elif terminal[i]:
                    length = len(buffer)
                    trajectory = buffer.get_trajectory()
                    for item in trajectory:
                        pad = np.zeros(np.asarray(item[0]).shape, 
                                       dtype = np.asarray(item[0]).dtype)
                        for _ in range(self.n_steps - length):
                            item.append(pad)
                    
                    experience = to_tensors(*trajectory,
                                            dtype = torch.float32,
                                            device = self._device)
                    self.replay_buffer.add(experience)
                    buffer.reset()
                    
    def add_to_local_buffer(self, obs, action, reward, next_obs, terminal,
                            *args, **kwargs): 
        assert (len(self.local_buffer) == obs.shape[0]), 'TODO'
        for i, buffer in enumerate(self.local_buffer):
            buffer.add((obs[i], action[i], reward[i], 
                        next_obs[i], terminal[i]))
        
    def update(self, obs, action, reward, next_obs, terminal):
        if self.training:
            if hasattr(self, 'local_buffer'):
                self.add_to_local_buffer(obs, action, reward, 
                                         next_obs, terminal)
            self.add_to_memory(obs, action, reward, next_obs, terminal)
            
        if self.ready_to_update():
            self.online_network.eval()
            
            # Maybe not the best way to do a line break, 
            # but I like it more than \
            (obs, action, reward, next_obs, terminal, 
             weight, indices) = self.sample_from_memory(self.batch_size)

            update_target = self.update_target(obs, action, reward, 
                                               next_obs, terminal)
            
            self.online_network.train()
            
            if self.n_steps > 1:
                obs = obs[:, 0]
                action = action[:, 0]
            
            if weight is None:
                weight = torch.ones(self.batch_size, device = self._device)
            else:
                weight = torch.as_tensor(weight, dtype = torch.float32, 
                                         device = self._device)
                assert (weight.dim() == 1), 'TODO'
            
            Q_s = self.estimate_q(obs, target_network = False)
            Q_sa = Q_s.gather(1, action.long().unsqueeze(1))
                
            td_error = Q_sa - update_target
            loss = (td_error).pow(2).mul(0.5).squeeze(-1)
            loss = (loss * weight).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradients:
                nn.utils.clip_grad_norm_(self.online_network.parameters(), 
                                         self.clip_gradients)
            self.optimizer.step()
            
            self.target_update()
            self.online_network.eval()
            
            updated_p = (np.abs(td_error.detach().cpu().numpy().squeeze()) 
                         + 1e-8)
            self.update_priorities(indices, updated_p)
            
        if self.training:
            self.current_step += 1
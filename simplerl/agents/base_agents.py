import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy

from .common.parallel_envs import ParallelEnvironment, VectorEnvWrapper

class Agent(abc.ABC):
    @abc.abstractmethod
    def action(self, obs):
        pass
    
    @abc.abstractmethod
    def update(self, obs, action, reward, next_obs, terminal):
        pass
        
class RandomAgent(Agent):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.seed()
        
    @property
    def training(self):
        return False
        
    def action(self, obs):
        return self.action_space.sample()
        
    def update(self, obs, action, reward, next_obs, terminal):
        pass
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        self.action_space.seed(seed = seed)

class LearningAgent(Agent):
    compatible_states = None
    compatible_actions = None

    def __init__(self, env_fn = None, 
                 model_fn = None, 
                 n_actors = 1):
        
        assert (callable(model_fn)), 'TODO'
        assert (n_actors > 0), 'TODO'
        assert (callable(env_fn)), 'TODO'
        
        self._model_fn = model_fn
        self._env_fn = env_fn
        self._n_actors = n_actors
        
        self._torch_objs = {}
        self._device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu')
        
        self.seed()
        self.reset_current_step()
        
    @abc.abstractmethod
    def action(self, obs):
        pass
    
    @abc.abstractmethod
    def update(self, obs, action, reward, next_obs, terminal):
        pass
    
    @property
    def env(self):
        return self._env
    
    @property
    def training(self):
        return self._training
        
    def train(self):
        self._training = True
        self.open_env()
        
    def eval(self):
        self._training = False
        self.open_env()
        
    def open_env(self):
        self.close_env()
        if self.training and (self._n_actors > 1):
            self._env = ParallelEnvironment([self._env_fn for _ 
                                             in range(self._n_actors)])
        else:
            self._env = VectorEnvWrapper(self._env_fn()) 
    
    def close_env(self):
        try:
            self._env.close()
        except AttributeError:
            pass
            
    def reset_current_step(self):
        self.current_step = 0
    
    def to(self, device):
        self._device = device
        for obj in self._torch_objs.values():
            try:
                obj.to(self._device) # optimizer objects don't have a to method
            except AttributeError:
                pass
                
    def register_torch_obj(self, obj, obj_name):
        # make sure that the object has a state_dict method for saving
        assert (hasattr(obj, 'state_dict')), 'TODO'
        assert (callable(getattr(obj, 'state_dict'))), 'TODO'
        
        self._torch_objs[obj_name] = obj
    
    def save(self, directory_path, **kwargs):
        torch_path = os.path.join(directory_path, 'torch_objs.tar')
        state_dicts = {obj_name: obj.state_dict() for obj_name, obj 
                       in self._torch_objs.items()}
        
        torch.save(state_dicts, torch_path)
        
    def load(self, directory_path):
        torch_path = os.path.join(directory_path, 'torch_objs.tar')
        model = torch.load(torch_path, map_location = self._device)
        
        for obj_name, state_dict in model.items():
            self._torch_objs[obj_name].load_state_dict(state_dict)
            
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        
    def ready_to_update(self):
        return True
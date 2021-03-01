import numpy as np
from gym import spaces
from .syspath_robolite import *
from robosuite.environments.panda_open_door import PandaOpenDoor

import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from utils.common_func import Drawer

class PandaOpenDoorFKTactile(_PandaOpenDoor):
    def __init__(self, use_tactile=True, full_obs=True, **kwargs):
        super().__init__(use_tactile=use_tactile, full_obs=full_obs, gripper_type="PandaGripperTactile", **kwargs)
        obs_dim = 12
        act_dim = 8
        self.tactile_dim=30
        self.use_tactile=use_tactile
        if full_obs:
            obs_dim+=13 # 7 dim joint velocities, 3 dim eef positions, 3 dim eef velocities
        if use_tactile:
            obs_dim+=self.tactile_dim # 30 dim for binary tactile signals
        self.full_obs = full_obs
        self.name = self.__class__.__name__.lower()
        self.observation_space = spaces.Box(np.zeros(obs_dim, dtype=np.float32), np.zeros(obs_dim, dtype=np.float32))
        self.action_space = spaces.Box(-np.ones(act_dim, dtype=np.float32), np.ones(act_dim, dtype=np.float32))  # 7 dim for robot arm and 1 dim for gripper
        
        self.plot_tactile = False

        if self.plot_tactile and self.use_tactile:
            self.drawer = Drawer()
            self.drawer.start()

        self.last_obs = None
        self.dist_threshold = 0.03
        self.gripper_width_threshold = 0.07
        self.p_tactile_signal_flip = 0.005 # with this probability the binary tactile singal will flip, e.g. 1 to 0 or 0 to 1

    def _final_obs(self, di):
        return di['task_state']
    
    def step(self, action):

        norm_action = np.concatenate((action[:-1], action[-1:]*0.005))   # joint action norm during training
        obs, reward, done, info = super().step(norm_action)

        if self.plot_tactile and self.use_tactile:
            self.drawer.add_value(obs['tactile'].copy())
            self.drawer.render()
        self.last_obs = self._final_obs(obs)

        # randomize the tactile signal
        if  self.use_tactile: 
            rands = np.random.uniform(0,1,size=self.tactile_dim)
            idx=np.where(rands<self.p_tactile_signal_flip)
            obs_to_flip=self.last_obs[-self.tactile_dim:][idx]
            self.last_obs[-self.tactile_dim:][idx]=-1*obs_to_flip+1 # a not operation for array

        return self.last_obs, reward, done, info
    
    def reset(self, **kwargs):
        return self._final_obs(super().reset(**kwargs))



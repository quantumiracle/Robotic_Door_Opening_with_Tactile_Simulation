'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
'''
import math
import random
import gym
import numpy as np
import torch
from torch.distributions import Normal
import queue

from rl.optimizers import SharedAdam, ShareParameters
from rl.buffers import ReplayBuffer
from rl.value_networks import QNetwork
from rl.policy_networks import DPG_PolicyNetwork
from utils.load_params import load_params
from utils.common_func import rand_params
import os
import copy

from mujoco_py import MujocoException

###############################  TD3  ####################################

class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        action_range, policy_target_update_interval=1, machine_type='gpu'):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim
        self.machine_type = machine_type

        self.q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.q_net2 = QNetwork(state_space, action_space, hidden_dim)
        # self.target_q_net1 = copy.deepcopy(self.q_net1)
        # self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.target_q_net2 = QNetwork(state_space, action_space, hidden_dim)
        self.policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range, machine_type=machine_type)
        # self.target_policsy_net = copy.deepcopy(self.policy_net)
        self.target_policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range, machine_type=machine_type)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
    
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = SharedAdam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = SharedAdam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=policy_lr)

    def to_cuda(self):
        self.q_net1 = self.q_net1.cuda()
        self.q_net2 = self.q_net2.cuda()
        self.target_q_net1 = self.target_q_net1.cuda()
        self.target_q_net2 = self.target_q_net2.cuda()
        self.policy_net = self.policy_net.cuda()
        self.target_policy_net = self.target_policy_net.cuda()
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        for _ in range(3): # sample several times to prevent unkown error of failure in sampling
            try:
                state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
                break
            except Exception as e:
                print(e)
            
        if self.machine_type == 'gpu':        
            state      = torch.FloatTensor(state).cuda()
            next_state = torch.FloatTensor(next_state).cuda()
            action     = torch.FloatTensor(action).cuda()
            reward     = torch.FloatTensor(reward).unsqueeze(1).cuda()  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()
        else:  # if not gpu, then cpu
            state      = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            action     = torch.FloatTensor(action)
            reward     = torch.FloatTensor(reward).unsqueeze(1)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action = self.policy_net.evaluate(state, noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action = self.target_policy_net.evaluate(next_state, noise_scale=eval_noise_scale) # clipped normal noise

        if reward_scale:
            reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        if torch.isnan(q_value_loss1): # error capture
            print('Error: q loss 1 value is nan')
            # print(state, action, reward, next_state, done)
            # breakpoint()
        else:
            self.q_optimizer1.zero_grad()
            q_value_loss1.backward()
            self.q_optimizer1.step()
        if torch.isnan(q_value_loss2): # error captur
            print('Error: q loss 2 value is nan')
            # breakpoint()
        else:
            self.q_optimizer2.zero_grad()
            q_value_loss2.backward()
            self.q_optimizer2.step()

        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()
            if torch.isnan(policy_loss): # error capture
                print('Error: policy loss value is nan')
                breakpoint()
            else:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
            
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return predicted_q_value1.mean()

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        device = 'cuda:0' if torch.cuda.is_available() and self.machine_type == 'gpu' else 'cpu'
        self.q_net1.load_state_dict(torch.load(path+'_q1', map_location=device))
        self.q_net2.load_state_dict(torch.load(path+'_q2', map_location=device))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=device))
        # self.q_net1.eval()
        # self.q_net2.eval()
        # self.policy_net.eval()

    def share_memory(self):
        self.q_net1.share_memory()
        self.q_net2.share_memory()
        self.target_q_net1.share_memory()
        self.target_q_net2.share_memory()
        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        ShareParameters(self.q_optimizer1)
        ShareParameters(self.q_optimizer2)
        ShareParameters(self.policy_optimizer)


def worker(id, td3_trainer, envs, env_name, rewards_queue, eval_rewards_queue, success_queue,\
        eval_success_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size,\
        explore_steps, noise_decay, update_itr, explore_noise_scale, eval_noise_scale, reward_scale,\
        gamma, soft_tau, DETERMINISTIC, hidden_dim, model_path, render, randomized_params, seed=1):
    '''
    the function for sampling with multi-processing
    '''
    with torch.cuda.device(id % torch.cuda.device_count()):
        td3_trainer.to_cuda()
        print(td3_trainer, replay_buffer)
        try:
            env = gym.make(envs[env_name])  # mujoco env
        except:
            env = envs[env_name]()  # robot env
        frame_idx=0
        rewards=[]
        current_explore_noise_scale = explore_noise_scale
        last_savepoint = 0
        for eps in range(max_episodes): # training loop
            episode_reward = 0
            if randomized_params:
                state = env.reset(**(rand_params(env, params=randomized_params)[0]))
            else:
                state = env.reset()
            current_explore_noise_scale = current_explore_noise_scale*noise_decay
            
            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = td3_trainer.policy_net.get_action(state, noise_scale=current_explore_noise_scale)
                else:
                    action = td3_trainer.policy_net.sample_action()
                try:
                    next_state, reward, done, info = env.step(action)
                    if render: 
                        env.render()   
                except KeyboardInterrupt:
                    print('Finished')
                    td3_trainer.save_model(model_path)
                except MujocoException:
                    print('MujocoException')
                    # recreate an env, since sometimes reset not works, the env might be broken
                    try:
                        env = gym.make(env_name)  # mujoco env
                    except:
                        env = envs[env_name]()  # robot env

                    # td3_trainer.policy_net = td3_trainer.target_ini(td3_trainer.target_policy_net, td3_trainer.policy_net)  # reset policy net as target net
                    try:  # recover the policy from last savepoint
                        td3_trainer.load_model(model_path+'/{}_td3'.format(last_savepoint))
                    except:
                        print('Error: no last savepoint: ', last_savepoint)
                    break

                if np.isnan(np.sum([np.sum(state), np.sum(action), reward, np.sum(next_state), done])): # prevent nan in data 
                    print('Nan in data')
                    # print(state, action, reward, next_state, done)
                else: # prevent nan in data 
                    replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                frame_idx += 1
                
                if done:
                    break
            print('Worker: ', id, '|Episode: ', eps, '| Episode Reward: ', episode_reward, '| Step: ', step)
            rewards_queue.put(episode_reward)

            if eps % eval_interval == 0 and eps>0:  # only one process update
                td3_trainer.save_model(model_path+'/{}_td3'.format(eps))
                last_savepoint = eps

            if replay_buffer.get_length() > batch_size:
                for i in range(update_itr):
                    _=td3_trainer.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale, \
                        gamma=gamma, soft_tau=soft_tau)
        td3_trainer.save_model(model_path+'/{}_td3'.format(eps))


def cpu_worker(id, td3_trainer, envs, env_name, rewards_queue, eval_rewards_queue, success_queue,\
        eval_success_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size,\
        explore_steps, noise_decay, update_itr, explore_noise_scale, eval_noise_scale, reward_scale,\
        gamma, soft_tau, DETERMINISTIC, hidden_dim, model_path, render, randomized_params, seed=1):
    '''
    the function for sampling with multi-processing
    '''
    # td3_trainer.to_cuda()
    print(td3_trainer, replay_buffer)
    try:
        env = gym.make(env_name)  # mujoco env
    except:
        env = envs[env_name]()  # robot env
    frame_idx=0
    rewards=[]
    current_explore_noise_scale = explore_noise_scale
    last_savepoint = 0
    for eps in range(max_episodes): # training loop
        episode_reward = 0
        if randomized_params:
            state = env.reset(**(rand_params(env, params=randomized_params)[0]))
        else:
            state = env.reset()
        current_explore_noise_scale = current_explore_noise_scale*noise_decay
        
        for step in range(max_steps):
            if frame_idx > explore_steps:
                action = td3_trainer.policy_net.get_action(state, noise_scale=current_explore_noise_scale)
            else:
                action = td3_trainer.policy_net.sample_action()
            try:
                next_state, reward, done, info = env.step(action)
                if render: 
                    env.render()   
            except KeyboardInterrupt:
                print('Finished')
                td3_trainer.save_model(model_path)
            except MujocoException:
                print('MujocoException')
                # recreate an env, since sometimes reset not works, the env might be broken
                try:  
                    env = gym.make(env_name)  # mujoco env
                except:
                    env = envs[env_name]()  # robot env

                # td3_trainer.policy_net = td3_trainer.target_ini(td3_trainer.target_policy_net, td3_trainer.policy_net)  # reset policy net as target net
                try: # recover the policy from last savepoint
                    td3_trainer.load_model(model_path+'/{}_td3'.format(last_savepoint))
                except:
                    print('Error: no last savepoint: ', last_savepoint)
                break

            if np.isnan(np.sum([np.sum(state), np.sum(action), reward, np.sum(next_state), done])): # prevent nan in data 
                print('Nan in data')
                # print(state, action, reward, next_state, done)
            else: # prevent nan in data 
                replay_buffer.push(state, action, reward, next_state, done)    

            state = next_state
            episode_reward += reward
            frame_idx += 1
            if done:
                break
        print('Worker: ', id, '|Episode: ', eps, '| Episode Reward: ', episode_reward, '| Step: ', step)
        rewards_queue.put(episode_reward)

        if eps % eval_interval == 0 and eps>0:
            td3_trainer.save_model(model_path+'/{}_td3'.format(eps))
            last_savepoint = eps

        if replay_buffer.get_length() > batch_size:
            for i in range(update_itr):
                _=td3_trainer.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale, \
                    gamma=gamma, soft_tau=soft_tau)
    
    td3_trainer.save_model(model_path+'/{}_td3'.format(eps))

import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import time
import queue
import math
import random
import datetime
import os
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from rl.buffers import ReplayBuffer
from utils.load_params import load_params
from utils.common_func import rand_params
from rl.td3.td3 import TD3_Trainer, worker, cpu_worker
# from rl.td3.td3_test import TD3_Trainer, worker, cpu_worker


def train_td3(env, envs, train, test, finetune, path, model_id, render, process, seed):
    torch.manual_seed(seed)  # Reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # hyper-parameters for RL training
    try: # custom env
        env_name = env.name 
    except: # gym env
        env_name = env.spec.id

    num_workers = process # or: mp.cpu_count()
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = './data/weights/{}{}'.format(prefix, seed)
    if not os.path.exists(model_path) and train:
        os.makedirs(model_path)
    print('Model Path: ', model_path)

    # load other default parameters
    [max_steps, max_episodes, action_range, batch_size, explore_steps, update_itr, eval_interval, explore_noise_scale, eval_noise_scale, reward_scale, \
        gamma, soft_tau, hidden_dim, noise_decay, policy_target_update_interval, q_lr, policy_lr, replay_buffer_size, randomized_params, DETERMINISTIC] = \
        load_params('td3', env_name, ['max_steps', 'max_episodes', 'action_range', 'batch_size', 'explore_steps', 'update_itr', 'eval_interval', \
        'explore_noise_scale', 'eval_noise_scale', 'reward_scale', 'gamma', 'soft_tau', 'hidden_dim', 'noise_decay', 'policy_target_update_interval', \
        'q_lr', 'policy_lr','replay_buffer_size', 'randomized_params', 'deterministic'] )
    if not action_range:
        action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric
    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager

    action_space = env.action_space
    state_space = env.observation_space

    machine_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    worker_ = worker if torch.cuda.is_available() else cpu_worker
    print(machine_type, worker)
    td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        policy_target_update_interval=policy_target_update_interval, action_range=action_range, machine_type=machine_type )

    if train: 
        if finetune is True:
            td3_trainer.load_model('./data/weights/'+ path +'/{}_td3'.format(model_id))
        td3_trainer.share_memory()

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()

        processes=[]
        rewards=[]
        success = []
        eval_rewards = []
        eval_success = []

        for i in range(num_workers):
            process = Process(target=worker_, args=(i+4, td3_trainer, envs, env_name, rewards_queue, eval_rewards_queue, success_queue, eval_success_queue, \
            eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay, update_itr, explore_noise_scale, eval_noise_scale, \
            reward_scale, gamma, soft_tau,  DETERMINISTIC, hidden_dim, model_path, render, randomized_params))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep getting the episode reward from the queue
            r = rewards_queue.get()
            rewards.append(r)

            if len(rewards)%20==0 and len(rewards)>0:
                # plot(rewards)
                np.save('log/'+prefix+'td3_rewards', rewards)

        [p.join() for p in processes]  # finished at the same time

        td3_trainer.save_model(model_path)
        
    if test:
        import time
        model_path = './data/weights/'+ path +'/{}_td3'.format(str(model_id))
        print('Load model from: ', model_path)
        td3_trainer.load_model(model_path)
        # td3_trainer.to_cuda()
        # print(env.action_space.high, env.action_space.low)

        no_DR = False
        dist_threshold = 0.02
        dist_threshold_max = 0.07
        if no_DR:
            randomized_params=None
        print(randomized_params)
        for eps in range(10):
            if not no_DR:
                param_dict, param_vec = rand_params(env, randomized_params)
                state = env.reset(**param_dict)
                print('Randomized parameters value: ', param_dict)
            else:
                state = env.reset()
            env.render()   
            episode_reward = 0
            import time
            time.sleep(1)
            s_list = []
            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, noise_scale=0.0)

                next_state, reward, done, _ = env.step(action)
                env.render() 
                # time.sleep(0.1)
                episode_reward += reward
                state=next_state
                s_list.append(state)
                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            # np.save('data/s.npy', s_list)

def get_hyperparams(env_name):
    if 'pandaopendoorfk' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        # 'max_steps': 300,
        'max_steps': 1000,
        'max_episodes': 10000,
        'action_range': 0.05,  # on joint
        # 'action_range': 0.1,  # on joint
        'batch_size': 640,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 500, # evaluate the model and save it
        'explore_noise_scale': 0.02, 
        'eval_noise_scale': 0.02,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization in a batch
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e6,
        'randomized_params': ['knob_friction', 'hinge_stiffness', 'hinge_damping', 'hinge_frictionloss', 'door_mass', 'knob_mass', 'table_position_offset_x', 'table_position_offset_y'],  # choose in: 'all', None, or a list of parameter keys
        'deterministic': True,
        }

    else:
        raise NotImplementedError

    print('Hyperparameters: ', hyperparams_dict)
    return hyperparams_dict

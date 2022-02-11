import numpy as np

def action_noise_wrapper(Env, add_range, add_default, mul_range, mul_default, sys_range, sys_default, dim=None):
    def add_noise_to(env, action, dim=dim):
        if dim:
            add = np.random.uniform(-1, 1, action[dim].shape) * env.action_noise_additive
            mul = 1.0 + np.random.uniform(-1, 1, action[dim].shape) * env.action_noise_multiplicative
            add_sys = env.action_noise_systematic
            action_ = action.copy()
            action_[dim] = (action[dim] + add + add_sys) * mul
        else:
            add = np.random.uniform(-1, 1, action.shape) * env.action_noise_additive
            mul = 1.0 + np.random.uniform(-1, 1, action.shape) * env.action_noise_multiplicative
            add_sys = env.action_noise_systematic
            action_ = (action + add + add_sys) * mul
        return action_
    
    class ActionNoise(Env):
        parameters_spec = {
            **Env.parameters_spec,
            'action_noise_additive': add_range,
            'action_noise_multiplicative': mul_range,
            'action_noise_systematic': sys_range,
        }

        def reset_props(self, action_noise_additive=add_default, action_noise_multiplicative=mul_default, action_noise_systematic=sys_default, **kwargs):
            self.action_noise_additive = action_noise_additive
            self.action_noise_multiplicative = action_noise_multiplicative
            self.action_noise_systematic = action_noise_systematic
            super().reset_props(**kwargs)
            self.params_dict.update({
                'action_noise_additive': action_noise_additive,
                'action_noise_multiplicative': action_noise_multiplicative,
                'action_noise_systematic': action_noise_systematic,
            })
        
        def step(self, action):
            return super().step(add_noise_to(self, action))
        
    return ActionNoise

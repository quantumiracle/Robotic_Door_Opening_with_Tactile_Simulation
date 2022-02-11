import numpy as np

def obs_noise_wrapper(Env, names, range_var, default):
    def add_noise_to(env, obs):
        assert(all(name in obs for name in names))
        for name in names:
            var = getattr(env, 'obs_noise_' + name, default)
            noise = np.random.normal(scale=var, size=np.array(obs[name]).shape)
            obs[name] = obs[name] + noise
        return obs

    
    class ObservationNoise(Env):
        parameters_spec = {
            **Env.parameters_spec,
            **{'obs_noise_' + name: range_var for name in names},
        }

        def reset_props(self, **kwargs):
            r = {'obs_noise_' + name: default for name in names}
            p2 = {}
            for k in kwargs:
                if k in r: r[k] = kwargs[k]
                else: p2[k] = kwargs[k]
            for k in r:
                setattr(self, k, r[k])
            super().reset_props(**p2)
            self.params_dict.update(r)
        
        def step(self, action):
            obs, reward, done, info = super().step(action)
            return add_noise_to(self, obs), reward, done, info
        
        def reset(self, **kwargs):
            return add_noise_to(self, super().reset(**kwargs))

    return ObservationNoise

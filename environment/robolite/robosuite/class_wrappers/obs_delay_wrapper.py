import copy
from collections import deque

def obs_delay_wrapper(Env, names, max_delay):
    def add_delay_to(env, obs):
        assert(all(name in obs for name in names))
        for name in names:
            delay = getattr(env, 'obs_delay_' + name)
            queue = getattr(env, 'obs_delay_queue_' + name)
            queue.appendleft(copy.deepcopy(obs[name]))
            if len(queue) == delay + 1:   # else keep it as-is
                obs[name] = queue.pop()
        return obs
    
    class ObservationDelay(Env):
        parameters_spec = {
            **Env.parameters_spec,
            **{'obs_delay_' + name: max_delay + 1 for name in names},
        }

        def reset_props(self, **kwargs):
            tkeys = ['obs_delay_' + name for name in names]
            delays = {name: 0 for name in names}
            for name in names:
                if ('obs_delay_' + name) in kwargs:
                    delays[name] = kwargs['obs_delay_' + name]
            for name in names:
                setattr(self, 'obs_delay_' + name, delays[name])
                setattr(self, 'obs_delay_queue_' + name, deque(maxlen=max_delay + 1))
                
            p2 = {}
            for k in kwargs:
                if k not in tkeys:
                    p2[k] = kwargs[k]
            super().reset_props(**p2)
            self.params_dict.update(delays)

        
        def step(self, action):
            obs, reward, done, info = super().step(action)
            return add_delay_to(self, obs), reward, done, info

        def reset(self, **kwargs):
            return add_delay_to(self, super().reset(**kwargs))

    return ObservationDelay

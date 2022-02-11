import numpy as np

def latent_dynamics_provider(Env, params_to_attach='all'):  
    class DynamicsParams(Env):
        if params_to_attach=='all':
            _params_dim = len(Env.params_dict.keys().tolist())
        else:
            _params_dim = len(params_to_attach)
        _params_to_attach = None

        def reset(self, **kwargs):
            obs = super().reset(**kwargs)
            if params_to_attach == 'all':
                self._params_to_attach = self.params_dict
            else:
                self._params_to_attach = {k:v for k, v in self.params_dict.items() if k in params_to_attach}
            # print('reset: ', self.params_to_encode)
            info={}
            if self._params_to_attach:
                info['dynamics_params'] = list(self._params_to_attach.values())
            else:
                info['dynamics_params'] = np.zeros(self._params_dim) 
            return obs, info

        def step(self, action):
            obs, reward, done, info = super().step(action)
            if self._params_to_attach:
                info['dynamics_params'] = list(self._params_to_attach.values())
            else:
                info['dynamics_params'] = np.zeros(self._params_dim)           
            return obs, reward, done, info

    return DynamicsParams


def change_dof(Env, new_dof, old_dof=None):
    if old_dof is None:
        old_dof = Env.dof
    class DOFChangedEnv(Env):
        dof = new_dof
        
        def __init__(self, **kwargs):
            self.dof = old_dof
            super().__init__(**kwargs)
            self.dof = new_dof

        def step(self, action):
            self.dof = old_dof
            ret = super().step(action)
            self.dof = new_dof
            return ret

    return DOFChangedEnv

# from .panda_ik_wrapper import panda_ik_wrapper
from .panda_ik_simple_wrapper import panda_ik_simple_wrapper
from .obs_noise_wrapper import obs_noise_wrapper
from .obs_delay_wrapper import obs_delay_wrapper
from .action_noise_wrapper import action_noise_wrapper
from .latent_dynamics_provider import latent_dynamics_provider


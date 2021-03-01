# for consistence in comparison among different methods
from importlib import import_module  # dynamic module importing

def load_params(alg_name, env_name, parmas_list):
    """ load default parameter values """
    # module = import_module('.'.join(['sim2real_policies', alg_name, 'default_params']))
    if alg_name=='td3':
        module = import_module('default_params')
    elif alg_name=='ppo':
        module = import_module('default_params_ppo')
    print(module)
    default_params = getattr(module, 'get_hyperparams')(env_name)
    params_value_list = []
    for param in parmas_list:
        assert param in default_params.keys(), "No param {} in default dictionary".format(param)
        params_value_list.append(default_params[param])
    return params_value_list
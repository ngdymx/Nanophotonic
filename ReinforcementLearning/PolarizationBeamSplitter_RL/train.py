from random import triangular
import ray
from ray.rllib.agents.ppo import PPOTrainer
import os
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger
from PBS_env import PolarizationSplitter_env
import numpy as np
from alive_progress import alive_bar
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models import ModelCatalog     
import torch.nn as nn                
import gym
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
from ray.rllib.utils.spaces.space_utils import flatten_space
class MPCActionMaskingModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name,  *args, **kwargs):
        nn.Module.__init__(self)
        # self.original_space = (
        #     observation_space.original_space
        #     if hasattr(observation_space, "original_space")
        #     else observation_space
        # )
        super(MPCActionMaskingModel, self).__init__(observation_space,
            action_space, num_outputs, {}, name, 
            *args, **kwargs)
        orig_space = getattr(observation_space, "original_space", observation_space)
        
        # Observation of the dopping pattern, M * Nself.
        net_config = {
            "fcnet_hiddens": [512,512,512],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "fcnet_activation": "relu",
            "dim" : 20,
            "grayscale": True,
            "conv_filters": [
                [16, [5, 5], 1], 
                [32, [4, 4], 1], 
                [64, [4, 4], 1], 
                [128, [3, 3], 1], 
            ],
            "conv_activation": "relu",
        }
        self.action_embed_model = ComplexInputNetwork(orig_space['observation'], action_space, num_outputs, net_config, name + 'embed_network')
        #self.register_variables(self.action_embed_model.variables())
    
    def forward(self, input_dict, state, seq_lens):
        # print(input_dict['obs'])
        # print(state)
        # print(seq_lens)
        #print(input_dict['obs'])
        action_mask = input_dict['obs']['action_mask']
        original_space = getattr(self.obs_space, "original_space", None)
        #print(original_space)
        bsize = input_dict['obs_flat'].shape[0]
        actual_obs_dict = input_dict['obs']['observation']
        actual_obs_list = [actual_obs_dict[i].view(bsize, -1) for i in range(len(original_space['observation']))]
        actual_obs_flat = torch.cat(actual_obs_list, -1)
        internal_model_input = {'obs': actual_obs_flat}
        action_embed = self.action_embed_model(internal_model_input)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        final_logit = action_embed[0] + inf_mask
        return final_logit, state
    
    def value_function(self):
        return self.action_embed_model.value_function()
    
ModelCatalog.register_custom_model("maskedModel", MPCActionMaskingModel)

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

config_train = {
    "num_workers": 1,
    "num_gpus": 0,
    "gamma": 0.8,
    "lr": 0.0001,
#    "lr_schedule": [
#        [0, 0.0001],
#        [40000, 0.00005],
#    ],
    "train_batch_size": 500,
    "rollout_fragment_length": 500,
    "sgd_minibatch_size": 500,
    "num_sgd_iter" : 2,
    # Change the following line to `“framework”: “tf”` to use tensorflow
    "framework": "torch",
    "explore": True,
    "env": PolarizationSplitter_env,
    "model": {
        "custom_model": "maskedModel",
    },
    "env_config": {
        "M":    20,
        "N":    20,
        "neff1": 3.03321,
        "neff2": 2.56056,
        "neff":  1,
        "wl": 1550e-9,
        "TEz_xclbin": "FDTD_TEz_new.awsxclbin",
        "TMz_xclbin": "FDTD_TMz_new.awsxclbin",
        "Nx": 150,
        "Ny":   400,
        "Nt":   8000,
        "Debug": False,
        "max_episode_steps": 45,
    },
}



trainer = PPOTrainer(config=config_train,logger_creator=custom_log_creator(os.path.expanduser("~/PBS_ML/logs"), 'PBS'))
N = 600
result = []
total_reward = []
with alive_bar(N) as bar:
    for i in range(N):
        results = trainer.train()
        result.append(results)
        msg = f"Iter: {i}; avg. reward={results['episode_reward_mean']}"
        total_reward.append(results['episode_reward_mean'])
        bar.title(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
        if i == 0:
            os.system("echo \"" + msg + "\" > current_train.log")
        else:
            os.system("echo \"" + msg + "\" >> current_train.log")
            if (i % 50  == 0):
                trainer.save('./checkpoints/')
        bar()
   
# np.savetxt('train_Curve.txt',np.array(total_reward))
    

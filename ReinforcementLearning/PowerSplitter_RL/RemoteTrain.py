# This can be called locally or remotely depending on your settings
# Basic supports
import os
import numpy as np
from alive_progress import alive_bar
import time

# For taining system
import ray
from ray.rllib.agents.ppo import PPOTrainer
import gym

# For custome model
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn
from ray.rllib.utils.torch_utils import FLOAT_MIN

# For custome logger
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger

# For FPGA driver
from pynq import Device
from pynq import Overlay
from pynq import allocate

# It turned out we have to write everythong in one file using ray remote cluster, so allow this chaotic file

#######################################################################################################################
# Global Variables and Settings

# Design Target
# 1,2,3,4
# 1: 0.1/0.9
TARGET = 1

# Physical constant
C = float(299792458.0)

# Ray cluster entry
RAY_CLUSTER_HEAD_IP = '52.3.167.65'
# RAY_CLUSTER_HEAD_IP = ''
RAY_CLUSTER_HEAD_PORT = '10001'

# FPGA kernel ID doesn't work well on the remote rollout worker
if RAY_CLUSTER_HEAD_IP != '':
    REMOTE_WORKER = True
else:
    REMOTE_WORKER = False
    
# Rollout worker setup
NUM_OF_WORKERS = 8
NUM_OF_FPGAS_PER_WORKER = 2

# Training parameters
NUM_OF_ITERATION = 100 # Number of iterations (weight updates)
GAMMA = 0.9 # Discount factor, 0-1, larger means longer vision, 0 means fully greedy
LR = 0.0005 # Learining rate, default is 0.00005. To some extent, larger LR learns faster but may be not stable.
TRAIN_BATCH_SIZE = 512 # How many samples are requried to do each train
ROLLOUT_FRAGMENT_LENGTH = int(np.ceil(TRAIN_BATCH_SIZE / NUM_OF_WORKERS)) # How many samples one worker is required to provide to do on training.
SGD_MINIBATCH_SIZE = TRAIN_BATCH_SIZE # The data are rare and expensive, so we don't really use SGD, all samples are used.
NUM_SGD_ITER = 2 # How many time the samples are passed through the network

# Net config
NET_CONFIG = {
    "fcnet_hiddens": [512,512,512],
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    "fcnet_activation": "relu",
    "dim" : 44,
    "grayscale": True,
    "conv_filters": [
        [16, [4, 4], 1],
        [32, [3, 3], 1],
        [64, [3, 3], 1],
        [128, [2, 2], 1],
    ],
    "conv_activation": "relu",
}

# FPGA Kernels to be use
DEBUGING_MODE = False # debug mode doesn't use FPGA, only fake data are provided, just for code checking
FPGA_KERNEL_PATH = '/home/centos/'
FPGA_KERNEL = 'FDTD_TEz_Triple.awsxclbin'

# Monitoring s
LOG_PATH = './logs'
LOG_DISCRYPTION = 'FINAL'
CHECK_POINT_PATH = './checkpoint/'
CHECK_POINT_PERIOD = 50

# Number of GPU in the training machine
NUM_OF_GPUS = 1

# End of global variables and settings
#######################################################################################################################


#######################################################################################################################
# Initialize ray
if RAY_CLUSTER_HEAD_IP == '': # No cluster used locally
    ray.init(resources = {"FPGA": (NUM_OF_WORKERS * NUM_OF_FPGAS_PER_WORKER)})
else:
    ray.init("ray://" + RAY_CLUSTER_HEAD_IP + ":" + RAY_CLUSTER_HEAD_PORT)

#######################################################################################################################



#######################################################################################################################
# PowerSplitter supporting APIs
# Create the space and generating Ca, C, etc.
class PowerSplitter1_3:
    def __init__(self, Nx, Ny, wavelength = 1550e-9, neff1 = 3.03221, neff = 1):
        self.Nx = Nx
        self.Ny = Ny
        self.c = 299792458
        self.wavelength = wavelength
        self.mu0 = 4 * np.pi *1e-7
        self.ep0 = 1 / (4 * np.pi * 1e-7) / self.c / self.c
        self.neff1 = neff1
        self.neff = neff
        self.ep1 = self.ep0 * self.neff1 * self.neff1 #TEz
        self.ep2 = self.ep0 * self.neff * self.neff   #pertubation

        dx = 30e-9
        dy = 30e-9
        dt = (dx**-2+dy**-2)**-.5/self.c*0.99

        ep = np.zeros((Nx, Ny))
        mu = np.zeros((Nx, Ny))
        self.C_std = np.zeros((Nx, Ny))
        self.Ca_std = np.zeros((Nx, Ny))
        mu[:, :] = self.mu0
        ep[:, :] = self.ep0
        Ox= int(Nx / 2)
        Oy = int(Ny / 2)

        ep[(Ox - 40): (Ox + 40), (Oy - 40): (Oy + 40)] = self.ep1
                    
        for i in range(98, 115):
            for j in range(160):
                ep[i,j] = self.ep1
                
        for i in range(35, 52):
            for j in range(160):
                ep[i, 398-j] = self.ep1
                ep[i+31, 398-j] = self.ep1
                ep[i+63, 398-j] = self.ep1

        #PML
        d = 20
        R0 = 1e-16
        m = 3
        sigmax = np.zeros((self.Nx, self.Ny))
        sigmay = np.zeros((self.Nx, self.Ny))
        Pright = np.zeros((d))
        Ptop = np.zeros((d))
        sigmax_max = -np.log(R0) * (m+1) * self.ep0 * self.c / 2 / d /dx
        sigmay_max = -np.log(R0) * (m+1) * self.ep0 * self.c / 2 / d /dy
        for i in range(d):
            Pright[i]= np.power((i / d), m) * sigmax_max
            Ptop[i]= np.power((i / d), m) * sigmay_max
        for col in range(Ny):
            sigmax[Nx-d:Nx,col] = Pright
            sigmax[0:d,col] = np.flip(Pright)
        for j in range(150):
            sigmax[120: 140, j] = Pright.T
            sigmax[76: 96,j] = np.flip(Pright).T
        for row in range(Nx):
            sigmay[row,Ny-d:Ny] = Ptop 
            sigmay[row,0:d] = np.flip(Ptop)

        sigma = np.sqrt((np.power(sigmax, 2) + np.power(sigmay, 2))/2) 
        for row in range(Nx):
            for col in range(Ny):
                self.C_std[row, col] = ep[row, col] / (ep[row, col] + 0.5 * dt * sigma[row, col])
                self.Ca_std[row, col] = dt / dy / mu[row, col] * dt / dy / ep[row, col] * self.C_std[row, col]

        self.Ca_dopped_val = dt/dy/self.mu0 * dt/dy/self.ep2
        self.dt = dt
        self.Ca_dopped = self.Ca_std.copy()

    def reset(self):
        self.Ca_dopped = self.Ca_std.copy()

    def PatternDopping(self, row, col):
        W = 4
        Offset_W = 35
        Offset_L = 159
        ow = np.int(np.floor(W * row))
        owu = np.int(np.ceil(ow + W))
        ol = np.int(np.floor(W * col))
        olu = np.int(np.ceil(ol + W))
        self.Ca_dopped[Offset_W + ow:Offset_W + owu,Offset_L + ol:Offset_L + olu] = self.Ca_dopped_val

    def export(self,Ca_buffer,C_buffer):
        Ca = self.Ca_dopped.copy()
        Ca_buffer[:,:] = Ca
        C_buffer[:,:] = self.C_std.copy()
        
# end of PowerSplitter APIs  
#######################################################################################################################



#######################################################################################################################
# FPGA kernel driver
# APIs for running the fpga kernel        
class FDTD_kernel():
    def __init__(self, xclbin, Nx, Ny, Nt, dt, amp, wavelength = 1550e-9, device_id = 0):
        self.xclbin = xclbin
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.dt = dt
        self.amp = amp
        self.wavelength = wavelength
        
        self.overlay = Overlay(self.xclbin, device = Device.devices[device_id])

        self.C = allocate((Nx, Ny), dtype = 'float32', target = self.overlay.bank0)
        self.Ca = allocate((Nx, Ny), dtype = 'float32', target = self.overlay.bank0)
        self.source = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)
        self.out_f1 = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)
        self.out_f2 = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)
        self.out_f3 = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)

        self.f0 = C / wavelength
        self.t_index = np.arange(0, self.Nt)
        self.source[:] = self.amp * np.sin(2 * np.pi * self.f0 * self.t_index * self.dt)
        self.source.sync_to_device()
        self.task_on = None

    def run(self, src_row, src_col, det_f1_row, det_f1_col, det_f2_row, det_f2_col, det_f3_row, det_f3_col):
        self.C.sync_to_device()
        self.Ca.sync_to_device()
        self.task_on = self.overlay.FDTD_Kernel_1.start(self.out_f1, self.out_f2, self.out_f3,self.source, self.C, self.Ca, self.Nt, self.Nx, src_row, src_col, det_f1_row, det_f1_col, det_f2_row, det_f2_col, det_f3_row, det_f3_col)

    def join(self):
        self.task_on.wait()
        self.out_f1.sync_from_device()
        self.out_f2.sync_from_device()
        self.out_f3.sync_from_device()
        
# end of FPGA driver
#######################################################################################################################


#######################################################################################################################
# PowerSpitter gym compatiable environment
# reset, step, etc.

class PowerSplitter1_3_env(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self,config):
        # Load all configurations
        isRemote = config.get("Remote")
        if not isRemote: # Local: the fpga id depends on the worker id
            self.worker_index = config.worker_index
        else: # Remote: uncertainty, only use first FPGA
            self.worker_index = 1
        self.M      = config.get("M")
        self.N      = config.get("N")
        self.neff1   = config.get("neff1")
        self.neff  = config.get("neff")
        self.wl     = config.get("wl")
        self.xclbin = config.get("xclbin")
        self.Nx     = config.get("Nx")
        self.Ny     = config.get("Ny")
        self.Nt     = config.get("Nt")
        self.Debuging = config.get("Debug")

        # Observation for the power transmission, two port, from 0 to 1
        PowerTrans_ob = gym.spaces.Box(
            low = np.array([0, 0, 0, 0, 0, 0]),
            high = np.array([1, 1, 1, 1, 1, 1]),
        )
        # Observation of the dopping pattern, M * Nself.
        
        self.pattern_shape = (self.M, self.N, 3)
        Pattern_ob = gym.spaces.Box(low = np.zeros(self.pattern_shape), 
                                    high = np.ones(self.pattern_shape),
                                    dtype = np.float16)

        # Define the final observation, combine the Power Transmission and the Pattern
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Tuple((PowerTrans_ob,Pattern_ob)),
                'action_mask': gym.spaces.MultiBinary(self.M * self.N)
            }
        )

        self.action_mask = np.array([1] * self.M * self.N)

        # Action space, each action determines one column, so M * 1
        self.action_space = gym.spaces.Discrete(self.M * self.N)

        # Init PolarizationSplitter
        self.PowerSplitter1_3_inst = PowerSplitter1_3(self.Nx, self.Ny, self.wl, self.neff1, self.neff)

        # Create driver
        if not self.Debuging:
            self.simulator = FDTD_kernel(self.xclbin, self.Nx, self.Ny, self.Nt, self.PowerSplitter1_3_inst.dt, 2, self.wl, device_id = (self.worker_index - 1))

        # Declare some variables
        self.T1_target = 1/3
        self.T2_target = 1/3
        self.T3_target = 1/3
        self.training_step = 0
        self.current_pattern = np.zeros(shape=Pattern_ob.sample().shape, dtype=np.int32)
        self.reward_max = -100000
        self.E_total = 0

    def fitness_function(self, E1, E2, E3):
        T1 = E1 / self.E_total
        T2 = E2 / self.E_total
        T3 = E3 / self.E_total
        T_total = T1 + T2 + T3
        fitness = 0.2/(0.2 + (T1 - self.T1_target * T_total) ** 2 + (T2 - self.T2_target * T_total) ** 2 + (T3 - self.T3_target * T_total) ** 2 + (1 - T_total) ** 2)
        # bond the transfer
        if T1 > 1:
            T1 = 1
        elif T1 < 0:
            T1= 0

        if T2 > 1:
            T2= 1
        elif T2 < 0:
            T2 = 0

        if T3 > 1:
            T3= 1
        elif T3 < 0:
            T3 = 0
        return T1, T2, T3, fitness

    def observe(self):
        E1 = np.var(self.simulator.out_f1[-2000:])
        E2 = np.var(self.simulator.out_f2[-2000:])
        E3 = np.var(self.simulator.out_f3[-2000:])
        return E1, E2, E3

    def reset(self, rand = False, target = 1/3):
        if rand:
            self.T1_target = np.random.randint(1, 4)
            self.T1_target = self.T1_target / 10
            self.T2_target = self.T1_target
            self.T3_target = self.T1_target
        else:
            self.T1_target = target
            self.T2_target = self.T1_target
            self.T3_target = self.T1_target

        self.training_step = 0
        self.TotalDoppingPoints = 0

        self.reward_max = -100000
        self.action_mask = np.array([1] * (self.M * self.N))
        # reset current pattern
        for i in range(self.M):
            for j in range(self.N):
                self.current_pattern[i][j][:] = 0

        self.PowerSplitter1_3_inst.reset()

        print("Env: New episode, new target is %.2f" % self.T1_target)
        if not self.Debuging:
            self.PowerSplitter1_3_inst.export(self.simulator.Ca, self.simulator.C)
            self.simulator.run(107, 39, 43, 370, 74, 370, 107, 370)
            self.simulator.join()
            E1, E2, E3 = self.observe()
            self.E_total = E1 + E2 + E3
            self.E_total = self.E_total / 0.18 * 0.84

            T1, T2, T3, _ = self.fitness_function(E1, E2, E3)
            PowerTrans_ob = np.array([T1, T2, T3, self.T1_target, self.T2_target, self.T3_target])
            Pattern_ob = self.current_pattern.copy()
            print("\tT1 = %.2f, T2 = %.2f, T3 = %.2f" % (T1, T2, T3))
            observation = {
                "observation" :(PowerTrans_ob, Pattern_ob),
                "action_mask" :self.action_mask
            }
            return observation
        else:
            E1 = 0.2856
            E2 = 0.5323
            E3 = 0.1821
            PowerTrans_ob = np.array([T1, T2, T3, self.T1_target, self.T2_target, self.T3_target])
            Pattern_ob = self.current_pattern.copy()
            return (PowerTrans_ob, Pattern_ob)
        
    def step(self, action):
        #action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        action = int(action)
        row = int(action / self.N)
        col = int(action % self.N)
        
        print("--Pertubation point: ", (row,col))
        self.current_pattern[row][col][:] = 1
        self.action_mask[action] = 0
        
        self.PowerSplitter1_3_inst.PatternDopping(row, col)

        self.training_step += 1
    
        reward = -1

        if not self.Debuging:
            self.PowerSplitter1_3_inst.export(self.simulator.Ca, self.simulator.C)
            self.simulator.run(107, 39, 43, 370, 74, 370, 107, 370)
            self.simulator.join()
            E1, E2, E3 = self.observe()
            T1, T2, T3, fitness = self.fitness_function(E1, E2, E3)
            PowerTrans_ob = np.array([T1, T2, T3, self.T1_target, self.T2_target, self.T3_target])
            Pattern_ob = self.current_pattern.copy()
            loss = 1 - T1 - T2 - T3
            print("\tT1 = %.2f, T2 = %.2f, T3 = %.2f" % (T1, T2, T3))
            observation = {
                "observation" :(PowerTrans_ob, Pattern_ob),
                "action_mask" :self.action_mask
            }
            reward += fitness
        else:
            E1 = 0.2856
            E2 = 0.5323
            E2 = 0.1821
            PowerTrans_ob = np.array([T1, T2, T3, self.T1_target, self.T2_target, self.T3_target])
            Pattern_ob = self.current_pattern.copy()
        reward = reward * 5
        
        if reward > self.reward_max:
            self.reward_max = reward
        
        
        if self.training_step == 120:
            done = True
        else:
            done = False
        if  (T1 + T2 + T3) < 0.18:
            reward -= 5 * (121 - self.training_step)
            done = True

        if done:
            print("Env: episode finished, maximum reward is %.2f" % self.reward_max)
            print("Final pattern:")            

            print("----------------------")
            for i in range(self.M):
                base = list("|********************|")
                for j in range(self.N):
                    if self.current_pattern[i][j][0] == 0:
                        base[j + 1] = ' '
                print("".join(base))
            print("----------------------")
        else:
            print("Env: step %02d, action is %d, reward is %.2f" % (self.training_step, action, reward))

        # print(self.current_pattern)
        return observation, reward, done, {}


# end of PowerSpitter MPC environment
#######################################################################################################################



#######################################################################################################################
# Custome model with action mask
# Based on ray.rllib.model.torch.complex_input_net
# Due to some BUGs, the network configurations is specified inside this model rather then the training configuration.

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
        net_config = NET_CONFIG
        self.action_embed_model = ComplexInputNetwork(orig_space['observation'], action_space, num_outputs, net_config, name + 'embed_network')
        # self.register_variables(self.action_embed_model.variables())

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

# End of custome model with action mask
#######################################################################################################################



#######################################################################################################################
# Custome log creater
# It is used to better record the information
# Ray hasn't optimize this. All logs will be created under ~/ray_results and hard to track.
def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

# End of custome log creater
#######################################################################################################################



#######################################################################################################################
# Train configuration

config_train = {
    "num_workers": NUM_OF_WORKERS,
    # Specify how many FPGAs are requried for each worker, the head of ray cluster needs it to do scheduling.
    "custom_resources_per_worker":{
            'FPGA': NUM_OF_FPGAS_PER_WORKER,
        },
    "num_gpus": NUM_OF_GPUS,
    "gamma": GAMMA,
    "lr": LR,
#    "lr_schedule": [
#        [0, 0.0001],
#        [40000, 0.00005],
#    ],
    "train_batch_size": TRAIN_BATCH_SIZE,
    "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
    "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
    "num_sgd_iter" : NUM_SGD_ITER,
    # Change the following line to `“framework”: “tf”` to use tensorflow
    "framework": "torch",
    "explore": True,
    "env": PowerSplitter1_3_env,
    "model": {
        "custom_model": "maskedModel",
    },
    "env_config": {
        "M":    20,
        "N":    20,
        "neff1": 3.03321,
        "neff":  1,
        "wl": 1550e-9,
        "xclbin": FPGA_KERNEL_PATH + FPGA_KERNEL,
        "Nx": 150,
        "Ny":   399,
        "Nt":   4000,
        "Debug": False,
        "Remote": REMOTE_WORKER,
        "max_episode_steps": 45,
    },
}
# end of train configuration
#######################################################################################################################



#######################################################################################################################
# main
# the __main__ function is not used as we may want to get access to train for debuging after the training.

# Check if the log path exist, if not make it. 
ret = os.system('ls ' + LOG_PATH)
if ret != 0: # 0 means the path exists
    os.system('mkdir -p ' + LOG_PATH)

# Check if the checkpoint path exist, if not make it. 
ret = os.system('ls ' + CHECK_POINT_PATH)
if ret != 0:
    os.system('mkdir -p ' + CHECK_POINT_PATH)
    
trainer = PPOTrainer(config=config_train,logger_creator=custom_log_creator(os.path.expanduser(LOG_PATH), 'PowerSplitter_' + LOG_DISCRYPTION))

N = NUM_OF_ITERATION

result = []

total_reward = []

with alive_bar(N) as bar:
    for i in range(N):
        results = trainer.train()
        result.append(results)
        msg = f"Iter: {i}; avg. reward={results['episode_reward_mean']}"
        total_reward.append(results['episode_reward_mean'])
        if (results["episode_reward_mean"] / results["episode_len_mean"]) >= -0.15:
            print("Current Batches: %d" % i)
            os.system("echo \"" + msg + "\" >> current_train.log")
            trainer.save(CHECK_POINT_PATH)
            break
            
        if i == 0:
            os.system("echo \"" + msg + "\" > current_train.log")
        elif i == (N - 1):
            os.system("echo \"" + msg + "\" >> current_train.log")
            trainer.save(CHECK_POINT_PATH)
        else:
            os.system("echo \"" + msg + "\" >> current_train.log")
            if (i % CHECK_POINT_PERIOD  == 0):
                trainer.save(CHECK_POINT_PATH)
        bar()


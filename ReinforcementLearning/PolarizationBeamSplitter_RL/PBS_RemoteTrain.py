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
GAMMA = 0.5 # Discount factor, 0-1, larger means longer vision, 0 means fully greedy
LR = 0.0001 # Learining rate, default is 0.00005. To some extent, larger LR learns faster but may be not stable.
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

# FPGA Kernels to be use
DEBUGING_MODE = False # debug mode doesn't use FPGA, only fake data are provided, just for code checking
FPGA_KERNEL_PATH = '/home/centos/'
FPGA_KERNEL_TMz = 'FDTD_TMz_new.awsxclbin'
FPGA_KERNEL_TEz = 'FDTD_TEz_new.awsxclbin'

# Monitoring setup
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
# PBS supporting APIs
# Create the space and generating Ca, C, etc.
class PolarizationSplitter:
    def __init__(self, Nx, Ny, wavelength = 1550e-9, neff1 = 3.03221, neff2 = 2.56056, neff = 1):
        self.Nx = Nx
        self.Ny = Ny
        self.c = 299792458
        self.wavelength = wavelength
        self.mu0 = 4 * np.pi *1e-7
        self.ep0 = 1 / (4 * np.pi * 1e-7) / self.c / self.c
        self.neff1 = neff1
        self.neff2 = neff2
        self.neff = neff
        self.ep1 = self.ep0 * self.neff1 * self.neff1 #TEz
        self.ep2 = self.ep0 * self.neff2 * self.neff2 #TMz
        self.ep3 = self.ep0 * self.neff * self.neff   #pertubation
        
        dx = 30e-9
        dy = 30e-9
        dt = (dx**-2+dy**-2)**-.5/self.c*0.99
        
        ep_TEz = np.zeros((Nx, Ny))
        ep_TMz = np.zeros((Nx, Ny))
        mu = np.zeros((Nx, Ny))
        self.C_std_TEz = np.zeros((Nx, Ny))
        self.Ca_std_TEz = np.zeros((Nx, Ny))
        self.C_std_TMz = np.zeros((Nx, Ny))
        self.Ca_std_TMz = np.zeros((Nx, Ny))
        mu[:, :] = self.mu0
        ep_TEz[:, :] = self.ep0
        ep_TMz[:, :] = self.ep0
        Ox = int(Nx / 2)  - 1
        Oy = int(Ny / 2) - 1

        ep_TEz[(Ox - 40): (Ox + 40 + 1), (Oy - 40): (Oy + 40 + 1)] = self.ep1
        ep_TMz[(Ox - 40): (Ox + 40 + 1), (Oy - 40): (Oy + 40 + 1)] = self.ep2
                    
        for i in range(67, 82):
            for j in range(160):
                ep_TEz[i,j] = self.ep1
                ep_TMz[i,j] = self.ep2
                
        for i in range(67, 82):
            for j in range(160):
                ep_TEz[i-16, 399-j] = self.ep1
                ep_TEz[i+16, 399-j] = self.ep1
                ep_TMz[i-16, 399-j] = self.ep2
                ep_TMz[i+16, 399-j] = self.ep2
        
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
            sigmax[87: 107, j] = Pright.T
            sigmax[43: 63,j] = np.flip(Pright).T
        for row in range(Nx):
            sigmay[row,Ny-d:Ny] = Ptop 
            sigmay[row,0:d] = np.flip(Ptop)

        sigma = np.sqrt((np.power(sigmax, 2) + np.power(sigmay, 2))/2) 
        for row in range(Nx):
            for col in range(Ny):
                self.C_std_TEz[row, col] = ep_TEz[row, col] / (ep_TEz[row, col] + 0.5 * dt * sigma[row, col])
                self.Ca_std_TEz[row, col] = dt / dy / mu[row, col] * dt / dy / ep_TEz[row, col] * self.C_std_TEz[row, col]
                self.C_std_TMz[row, col] = ep_TMz[row, col] / (ep_TMz[row, col] + 0.5 * dt * sigma[row, col])
                self.Ca_std_TMz[row, col] = dt / dy / mu[row, col] * dt / dy / ep_TMz[row, col] * self.C_std_TMz[row, col]
                
        self.Ca_dopped_val = dt/dy/self.mu0 * dt/dy/self.ep3
        self.dt = dt
        self.Ca_dopped_TEz = self.Ca_std_TEz.copy()
        self.Ca_dopped_TMz = self.Ca_std_TMz.copy()

    def reset(self):
        self.Ca_dopped_TEz = self.Ca_std_TEz.copy()
        self.Ca_dopped_TMz = self.Ca_std_TMz.copy()
                    
    def PatternDopping(self, row, col):
        W = 4
        Offset_W = 36
        Offset_L = 161
        ow = int(np.floor(W * row))
        owu = int(np.ceil(ow + W))
        ol = int(np.floor(W * col))
        olu = int(np.ceil(ol + W))
        self.Ca_dopped_TEz[Offset_W + ow:Offset_W + owu,Offset_L + ol:Offset_L + olu] = self.Ca_dopped_val
        self.Ca_dopped_TMz[Offset_W + ow:Offset_W + owu,Offset_L + ol:Offset_L + olu] = self.Ca_dopped_val
        
    def exportTEz(self,Ca_buffer_TEz,C_buffer_TEz):
        Ca_TEz = self.Ca_dopped_TEz.copy()
        Ca_buffer_TEz[:,:] = Ca_TEz
        C_buffer_TEz[:,:] = self.C_std_TEz.copy()
        
    def exportTMz(self,Ca_buffer_TMz,C_buffer_TMz):
        Ca_TMz = self.Ca_dopped_TMz.copy()
        Ca_buffer_TMz[:,:] = Ca_TMz
        C_buffer_TMz[:,:] = self.C_std_TMz.copy()


# end of PBS APIs  
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
        self.wavelength = wavelength

        self.overlay = Overlay(self.xclbin, device = Device.devices[device_id])

        self.C = allocate((Nx, Ny), dtype = 'float32', target = self.overlay.bank0)
        self.Ca = allocate((Nx, Ny), dtype = 'float32', target = self.overlay.bank0)
        self.source = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)
        self.out_f1 = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)
        self.out_f2 = allocate((Nt,), dtype = 'float32', target = self.overlay.bank0)

        self.f0 = C / self.wavelength
        self.t_index = np.arange(0,self.Nt)
        self.source[:] = amp * np.sin(2 * np.pi * self.f0 * self.t_index * self.dt)
        self.source.sync_to_device()
        self.task_on = None

    def apply_source(self, src):
        self.source[:] = src
        self.source.sync_to_device()

    def run(self, src_row, src_col, det_f1_row, det_f1_col, det_f2_row, det_f2_col):
        self.C.sync_to_device()
        self.Ca.sync_to_device()
        # No perturbations
        self.task_on = self.overlay.FDTD_Kernel_1.start(self.out_f1, self.out_f2, self.source, self.C, self.Ca, self.Nt, self.Nx, src_row, src_col, det_f1_row, det_f1_col, det_f2_row, det_f2_col)

    def join(self):
        self.task_on.wait()
        self.out_f1.sync_from_device()
        self.out_f2.sync_from_device()

        
# end of FPGA driver
#######################################################################################################################


#######################################################################################################################
# MPC gym compatiable environment
# reset, step, etc.

class PolarizationSplitter_env(gym.Env):
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
        self.neff2   = config.get("neff2")
        self.neff  = config.get("neff")
        self.wl     = config.get("wl")
        self.TEz_xclbin = config.get("TEz_xclbin")
        self.TMz_xclbin = config.get("TMz_xclbin")
        self.Nx     = config.get("Nx")
        self.Ny     = config.get("Ny")
        self.Nt     = config.get("Nt")
        self.Debuging = config.get("Debug")
        # Observation for the power transmission, two port, from 0 to 1
        PowerTrans_ob = gym.spaces.Box(
            low = np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high = np.array([1, 1, 1, 1, 1, 1, 1, 1]),
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
        self.PolarizationSplitter_inst = PolarizationSplitter(self.Nx, self.Ny, self.wl, self.neff1, self.neff2, self.neff)

        # Create driver
        if not self.Debuging:
            self.simulator_TMz = FDTD_kernel(self.TMz_xclbin, self.Nx, self.Ny, self.Nt, self.PolarizationSplitter_inst.dt, 0.5, self.wl, device_id = (2 * self.worker_index - 2))
            self.simulator_TEz = FDTD_kernel(self.TEz_xclbin, self.Nx, self.Ny, self.Nt, self.PolarizationSplitter_inst.dt, 1.5, self.wl, device_id = (2 * self.worker_index - 1))

        # Declare some variables
        self.T1_TEz_target = 0
        self.T2_TEz_target = 1
        self.T1_TMz_target = 1
        self.T2_TMz_target = 0
        self.training_step = 0
        self.current_pattern = np.zeros(shape=Pattern_ob.sample().shape, dtype=np.int32)
        self.reward_max = -100000
        self.E_TEz_total = 0
        self.E_TMz_total = 0


    def fitness_function(self, E1_TEz, E2_TEz, E1_TMz, E2_TMz):
        T1_TEz = E1_TEz / self.E_TEz_total
        T2_TEz = E2_TEz / self.E_TEz_total
        T1_TMz = E1_TMz / self.E_TMz_total
        T2_TMz = E2_TMz / self.E_TMz_total
        fitness = 0.1/(0.1 + (T1_TEz - self.T1_TEz_target) ** 2 + (T2_TEz - self.T2_TEz_target) ** 2 + (T1_TMz - self.T1_TMz_target) ** 2 + (T2_TMz - self.T2_TMz_target) ** 2)
        # bond the transfer
        if T1_TEz > 1:
            T1_TEz = 1
        elif T1_TEz < 0:
            T1_TEz = 0

        if T2_TEz > 1:
            T2_TEz = 1
        elif T2_TEz < 0:
            T2_TEz = 0

        if T1_TMz > 1:
            T1_TMz = 1
        elif T1_TMz < 0:
            T1_TMz = 0

        if T2_TMz > 1:
            T2_TMz = 1
        elif T2_TMz < 0:
            T2_TMz = 0
        return T1_TEz, T2_TEz, T1_TMz, T2_TMz, fitness
        
    def observe(self):

        E1_TEz = np.var(self.simulator_TEz.out_f1[-2000:])
        E2_TEz = np.var(self.simulator_TEz.out_f2[-2000:])
        E1_TMz = np.var(self.simulator_TMz.out_f1[-2000:])
        E2_TMz = np.var(self.simulator_TMz.out_f2[-2000:])
        return E1_TEz, E2_TEz, E1_TMz, E2_TMz

    def reset(self, rand = False, target = 0.0):
        if rand:
            self.T1_TEz_target = np.random.randint(1, 4)
            self.T1_TEz_target = self.T1_TEz_target / 10
            self.T2_TEz_target = 1 - self.T1_TEz_target
            self.T1_TMz_target = np.random.randint(1, 4)
            self.T1_TMz_target = self.T1_TMz_target / 10
            self.T2_TMz_target = 1 - self.T1_TMz_target
        else:
            self.T1_TEz_target = target
            self.T2_TEz_target = 0.65 - self.T1_TEz_target
            self.T2_TMz_target = target
            self.T1_TMz_target = 0.55 - self.T2_TMz_target

        self.training_step = 0
        self.TotalDoppingPoints = 0

        self.reward_max = -100000
        self.action_mask = np.array([1] * (self.M * self.N))
        # reset current pattern
        for i in range(self.M):
            for j in range(self.N):
                self.current_pattern[i][j][:] = 0

        self.PolarizationSplitter_inst.reset()

        print("Env: New episode!")
        
        if not self.Debuging:
            self.PolarizationSplitter_inst.exportTEz(self.simulator_TEz.Ca, self.simulator_TEz.C)
            self.PolarizationSplitter_inst.exportTMz(self.simulator_TMz.Ca, self.simulator_TMz.C)
            self.simulator_TEz.run(75,30,59,302,91,302)
            self.simulator_TMz.run(75,30,59,302,91,302)
            self.simulator_TEz.join()
            self.simulator_TMz.join()
            E1_TEz, E2_TEz, E1_TMz, E2_TMz = self.observe()
            self.E_TEz_total = E1_TEz + E2_TEz
            self.E_TEz_total = self.E_TEz_total / 0.25
            self.E_TMz_total = E1_TMz + E2_TMz
            self.E_TMz_total = self.E_TMz_total / 0.8
            T1_TEz, T2_TEz, T1_TMz, T2_TMz, _ = self.fitness_function(E1_TEz, E2_TEz, E1_TMz, E2_TMz)
            PowerTrans_ob = np.array([T1_TEz, T2_TEz, T1_TMz, T2_TMz, self.T1_TEz_target, self.T2_TEz_target, self.T1_TMz_target, self.T2_TMz_target])
            Pattern_ob = self.current_pattern.copy()
            print("\tT1_TEz = %.2f, T2_TEz = %.2f, T1_TMz = %.2f, T2_TMz = %.2f" % (T1_TEz, T2_TEz, T1_TMz, T2_TMz))
            observation = {
                "observation" :(PowerTrans_ob, Pattern_ob),
                "action_mask" :self.action_mask
            }
            return observation
        else:
            E1_TEz = 0.5
            E2_TEz = 0.5
            E1_TMz = 0.5
            E2_TMz = 0.5
            PowerTrans_ob = np.array([E1_TEz / self.E_TEz_total, E2_TEz / self.E_TEz_total, E1_TMz / self.E_TMz_total, E2_TMz / self.E_TMz_total, 
            self.T1_TEz_target, self.T2_TEz_target, self.T1_TMz_target, self.T2_TMz_target])
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
        
        self.PolarizationSplitter_inst.PatternDopping(row, col)

        self.training_step += 1
    
        
        reward = -1

        if not self.Debuging:
            self.PolarizationSplitter_inst.exportTEz(self.simulator_TEz.Ca, self.simulator_TEz.C)
            self.PolarizationSplitter_inst.exportTMz(self.simulator_TMz.Ca, self.simulator_TMz.C)
            self.simulator_TEz.run(75,30,59,302,91,302)
            self.simulator_TMz.run(75,30,59,302,91,302)
            self.simulator_TEz.join()
            self.simulator_TMz.join()
            E1_TEz, E2_TEz, E1_TMz, E2_TMz = self.observe()
            T1_TEz, T2_TEz, T1_TMz, T2_TMz, fitness = self.fitness_function(E1_TEz, E2_TEz, E1_TMz, E2_TMz)
            PowerTrans_ob = np.array([T1_TEz, T2_TEz, T1_TMz, T2_TMz, self.T1_TEz_target, self.T2_TEz_target, self.T1_TMz_target, self.T2_TMz_target])
            Pattern_ob = self.current_pattern.copy()
            loss = 2 - T1_TEz - T2_TEz - T1_TMz - T2_TMz
            print("\tT1_TEz = %.2f, T2_TEz = %.2f, T1_TMz = %.2f, T2_TMz = %.2f" % (T1_TEz, T2_TEz, T1_TMz, T2_TMz))
            observation = {
                "observation" :(PowerTrans_ob, Pattern_ob),
                "action_mask" :self.action_mask
            }
            reward += fitness

            
        reward = reward * 5
        
        if reward > self.reward_max:
            self.reward_max = reward
        
        
        if self.training_step == 150:
            done = True
        else:
            done = False
        if  (T1_TEz + T2_TEz) < 0.15 or (T1_TMz + T2_TMz) < 0.3:
            reward -= 5 * (151 - self.training_step)
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


# end of gym MPC environment
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
        "TEz_xclbin": FPGA_KERNEL_PATH + FPGA_KERNEL_TEz,
        "TMz_xclbin": FPGA_KERNEL_PATH + FPGA_KERNEL_TMz,
        "Nx": 150,
        "Ny":   400,
        "Nt":   8000,
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
    
trainer = PPOTrainer(config=config_train,logger_creator=custom_log_creator(os.path.expanduser(LOG_PATH), 'PBS_' + LOG_DISCRYPTION))
trainer.restore('./checkpoint/checkpoint_000301')

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


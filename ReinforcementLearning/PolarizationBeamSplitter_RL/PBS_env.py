import re
import matplotlib.pyplot as plt
import numpy as np
import gym
from zmq import device
from PBS_api import PolarizationSplitter
from FDTD_kernel_Driver import FDTD_kernel
# Configurations:
# M :       Number of rows of the polarizationsplitter, 20
# N :       Number of columns of the polarizationsplitter, 20. Also defines the steps of one eposode
# neff1:    Effective index of the material of TEz
# neff2:    Effective index of the material of TMz
# neff:     neff of the dopping part
# wl:       wavelength of the light
# xclbin:   Awsxclbin to use in the system
# Nx:       Horizontal Grid, for N, determined by xclbin
# Ny:       Vertical Grid, for M, determined by xclbin
# Nt:       Total simulation steps

# Debug:    If FPGA doesn't exist, set to 1

class PolarizationSplitter_env(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self,config):
        # Load all configurations
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
            self.simulator_TMz = FDTD_kernel(self.TMz_xclbin, self.Nx, self.Ny, self.Nt, self.PolarizationSplitter_inst.dt, 0.5, self.wl, device_id=0)
            self.simulator_TEz = FDTD_kernel(self.TEz_xclbin, self.Nx, self.Ny, self.Nt, self.PolarizationSplitter_inst.dt, 1.5, self.wl, device_id=1)

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
            self.T2_TEz_target = 0.5 - self.T1_TEz_target
            self.T2_TMz_target = target
            self.T1_TMz_target = 0.40 - self.T2_TMz_target

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
            self.simulator_TEz.run(75,30,59,350,91,350)
            self.simulator_TMz.run(75,30,59,350,91,350)
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
        
        
        if self.training_step == 100:
            done = True
        else:
            done = False
        if  (T1_TEz + T2_TEz) < 0.15 or (T1_TMz + T2_TMz) < 0.3:
            reward -= 5 * (101 - self.training_step)
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

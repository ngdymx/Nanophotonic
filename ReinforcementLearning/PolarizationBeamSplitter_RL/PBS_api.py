import numpy as np
import random

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
#         for j in range(150):
#             sigmax[87: 107, j] = Pright.T
#             sigmax[43: 63,j] = np.flip(Pright).T
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

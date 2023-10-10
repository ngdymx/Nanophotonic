from pynq import Device
from pynq import Overlay
from pynq import allocate
import numpy as np

C = 299792458
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

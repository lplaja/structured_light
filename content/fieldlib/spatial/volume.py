# fieldlib/spatial/volume.py

import numpy as np

class SpatialVolume:
    def __init__(self, xlim, ylim, zlim, nx, ny, nz):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.x = np.linspace(*xlim, nx)
        self.y = np.linspace(*ylim, ny)
        self.z = np.linspace(*zlim, nz)

        self.XY, self.YX = np.meshgrid(self.x, self.y, indexing='xy')
        self.ZY, self.YZ = np.meshgrid(self.z, self.y, indexing='xy')

    def get_xyz_axis(self):
        return self.x, self.y, self.z
    def get_xy_grid(self):
        return self.XY, self.YX

    def get_zy_grid(self):
        return self.ZY, self.YZ

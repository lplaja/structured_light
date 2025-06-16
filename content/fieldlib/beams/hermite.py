# fieldlib/beams/hermite.py

import numpy as np
from scipy.special import hermite
from fieldlib.spatial.volume import SpatialVolume

class HermiteGaussBeam:
    def __init__(self, amplitude, waist, wavelength, mode_indices=(0, 0), volume: SpatialVolume = None):
        self.amplitude = amplitude
        self.waist = waist
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.n, self.m = mode_indices
        self.volume = volume

    def evaluate(self, x, y, z=0):
        z_R = np.pi * self.waist**2 / self.wavelength  # Rayleigh range
        w_z = self.waist * np.sqrt(1 + (z / z_R)**2)    # beam radius
        with np.errstate(divide='ignore', invalid='ignore'):
            R_z = np.where(z == 0, np.inf, z * (1 + (z_R / z)**2))
        gouy = np.arctan(z / z_R)                       # Gouy phase

        Hn = hermite(self.n)
        Hm = hermite(self.m)

        x_norm = np.sqrt(2) * x / w_z
        y_norm = np.sqrt(2) * y / w_z

        envelope = np.exp(-(x**2 + y**2) / w_z**2)
        phase = np.exp(-1j * (self.k * (x**2 + y**2) / (2 * R_z) - (self.n + self.m + 1) * gouy))

        field = (self.waist / w_z) * Hn(x_norm) * Hm(y_norm) * envelope * phase
        return self.amplitude * field

    def evaluate_on_volume(self, plane='xy', index=0):
        if self.volume is None:
            raise ValueError("SpatialVolume must be provided to use evaluate_on_volume")

        if plane == 'xy':
            X, Y = self.volume.get_xy_grid()
            z = self.volume.z[index]
            return self.evaluate(X, Y, z)

        elif plane == 'zy':
            Z, Y = self.volume.get_zy_grid()
            x = self.volume.x[index]
            return self.evaluate(x, Y, Z)

        else:
            raise ValueError("Unsupported plane. Use 'xy' or 'zy'.")

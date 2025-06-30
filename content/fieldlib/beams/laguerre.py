# fieldlib/beams/laguerre.py

import numpy as np
from scipy.special import genlaguerre
from fieldlib.spatial.volume import SpatialVolume

class LaguerreGaussBeam:
    def __init__(self, amplitude, waist, wavelength, phase0=0, mode_indices=(0, 0), volume: SpatialVolume = None):
        self.amplitude = amplitude
        self.waist = waist
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.p, self.l = mode_indices
        self.phase0=phase0
        self.volume = volume

    def evaluate(self, x, y, z=0):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z_R = np.pi * self.waist**2 / self.wavelength  # Rayleigh range

        w_z = self.waist * np.sqrt(1 + (z / z_R)**2)  # beam radius
        with np.errstate(divide='ignore', invalid='ignore'):
            R_z = np.where(z == 0, np.inf, z * (1 + (z_R / z)**2))
        gouy = np.arctan(z / z_R)  # Gouy phase

        rho = np.sqrt(2) * r / w_z
        laguerre = genlaguerre(self.p, np.abs(self.l))(rho**2)

        amp = (self.waist / w_z) * np.exp(-r**2 / w_z**2) * (rho**np.abs(self.l)) * laguerre
        phase = np.exp(-1j * (self.k * r**2 / (2 * R_z) - self.l * phi + (2*self.p + np.abs(self.l) + 1) * gouy+self.phase0))

        return self.amplitude * amp * phase

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

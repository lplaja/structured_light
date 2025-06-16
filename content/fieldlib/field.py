# fieldlib/field.py

from fieldlib.beams.laguerre import LaguerreGaussBeam
from fieldlib.beams.hermite import HermiteGaussBeam
from fieldlib.utils.plotting import density_plot
from fieldlib.utils.plotting import density_plot_with_polarization
from fieldlib.spatial.volume import SpatialVolume
import numpy as np

BEAM_REGISTRY = {
    'LaguerreGauss': LaguerreGaussBeam,
    'HermiteGauss': HermiteGaussBeam,
}

POLARIZATION_VECTORS = {
    'x': np.array([1, 0], dtype=complex),
    'y': np.array([0, 1], dtype=complex),
    'R': (np.array([1, 0]) - 1j * np.array([0, 1])) / np.sqrt(2),
    'L': (np.array([1, 0]) + 1j * np.array([0, 1])) / np.sqrt(2),
}

class Field:
    def __init__(self, beam_type, wavelength, amplitude, waist, volume: SpatialVolume,
                 polarization='x', **beam_params):
        self.beam_type = beam_type
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.waist = waist
        self.volume = volume
        self.beam_params = beam_params
        self.polarization = polarization
        self._beam = self._create_beam()

    def _create_beam(self):
        BeamClass = BEAM_REGISTRY.get(self.beam_type)
        if BeamClass is None:
            raise ValueError(f"Unknown beam type: {self.beam_type}")
        return BeamClass(
            amplitude=self.amplitude,
            waist=self.waist,
            wavelength=self.wavelength,
            volume=self.volume,
            **self.beam_params
        )

    def evaluate(self, x, y, z=0):
        scalar_field = self._beam.evaluate(x, y, z)
        polarization_vector = POLARIZATION_VECTORS[self.polarization]
        return polarization_vector[:, np.newaxis, np.newaxis] * scalar_field

    def evaluate_on_volume(self, plane='xy', index=0):
        scalar_field = self._beam.evaluate_on_volume(plane=plane, index=index)
        polarization_vector = POLARIZATION_VECTORS[self.polarization]
        return polarization_vector[:, np.newaxis, np.newaxis] * scalar_field

    def plot(self, plane='xy', index=0, kind='amplitude'):
        density_plot(self, plane=plane, index=index, kind=kind)

    def get_polarization_vector(self):
        return POLARIZATION_VECTORS[self.polarization]


class FieldSuperposition:
    def __init__(self, fields):
        if not fields:
            raise ValueError("FieldSuperposition requires a non-empty list of fields.")
        self.fields = fields
        self.volume = fields[0].volume

    def evaluate(self, x, y, z=0):
        return sum(field.evaluate(x, y, z) for field in self.fields)

    def evaluate_on_volume(self, plane='xy', index=0):
        return sum(field.evaluate_on_volume(plane=plane, index=index) for field in self.fields)

    def plot(self, plane='xy', index=0, kind='amplitude'):
        density_plot_with_polarization(self, plane=plane, index=index, kind=kind, interactive=True)
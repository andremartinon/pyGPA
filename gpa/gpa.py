import matplotlib.pyplot as plt
import numpy as np

from typing import List


class GPA:

    def __init__(self, lattice: np.ndarray, modulus_tolerance: float = 0,
                 phases_tolerance: float = 0):
        self.n = lattice.size

        self.shape = lattice.shape
        self.lattice = lattice
        self.modulus_tolerance = modulus_tolerance
        self.phases_tolerance = phases_tolerance

        self.v, self.u = np.gradient(np.flip(lattice, 0))
        self.modulus = np.sqrt(self.u ** 2 + self.v ** 2)
        self.phases = np.arctan2(self.v, self.u)

        self._asymmetric_lattice()
        self._symmetric_lattice()

    def _check_modulus(self, nan_value: int = 0):
        half_size = int(self.n / 2)
        odd = int(self.n % 2)

        modulus = np.ravel(self.modulus)
        modulus_first_half = modulus[0:half_size + odd]
        modulus_second_half = np.flip(modulus[half_size:])
        modulus = np.abs(modulus_first_half - modulus_second_half)
        modulus[modulus <= self.modulus_tolerance] = 0
        modulus[np.isnan(modulus)] = nan_value
        modulus = np.concatenate((modulus, np.flip(modulus[:half_size])))
        return modulus.reshape(self.shape)

    def _check_phases(self, nan_value: int = 0):
        half_size = int(self.n / 2)
        odd = int(self.n % 2)

        phases = np.ravel(self.phases)
        phases_first_half = phases[0:half_size + odd]
        phases_second_half = np.flip(phases[half_size:])
        phases = np.abs(phases_first_half) + np.abs(phases_second_half)
        phases[phases >= (np.pi - self.phases_tolerance)] = 0
        phases[np.isnan(phases)] = nan_value
        phases = np.concatenate((phases, np.flip(phases[:half_size])))
        return phases.reshape(self.shape)

    def _asymmetric_lattice(self):
        modulus = self._check_modulus()
        phases = self._check_phases()

        self.asymmetric_lattice = (modulus + phases).astype('int64')
        self.asymmetric_lattice[self.asymmetric_lattice > 0] = 1
        self.asymmetric_lattice[self.asymmetric_lattice < 0] = 0

        self.asymmetric_va = np.sum(self.asymmetric_lattice)

    def _symmetric_lattice(self):
        modulus = self._check_modulus(nan_value=-1)
        phases = self._check_phases(nan_value=-1)

        self.symmetric_lattice = (modulus + phases).astype('int64')
        self.symmetric_lattice[self.symmetric_lattice != 0] = 1
        self.symmetric_lattice = 1 - self.symmetric_lattice

        self.symmetric_va = np.sum(self.symmetric_lattice)

    def _asymmetrical_magnitude_coefficient(self) -> float:
        asymmetric_u = self.u * self.asymmetric_lattice
        asymmetric_v = self.v * self.asymmetric_lattice
        asymmetric_modulus = np.sqrt(asymmetric_u ** 2 + asymmetric_v ** 2)

        modulus_of_sum = np.sqrt((np.sum(asymmetric_u))**2 +
                                 (np.sum(asymmetric_v))**2)
        sum_of_modulus = np.sum(asymmetric_modulus)

        with np.errstate(divide='ignore', invalid='ignore'):
            confluence = np.nan_to_num(modulus_of_sum/sum_of_modulus)

        return self.asymmetric_va / self.n * (2 - confluence)

    def _symmetrical_magnitude_coefficient(self) -> float:
        symmetric_u = self.u * self.symmetric_lattice
        symmetric_v = self.v * self.symmetric_lattice
        symmetric_modulus = np.sqrt(symmetric_u ** 2 + symmetric_v ** 2)

        modulus_of_sum = np.sqrt((np.sum(symmetric_u))**2 +
                                 (np.sum(symmetric_v))**2)
        sum_of_modulus = np.sum(symmetric_modulus)

        with np.errstate(divide='ignore', invalid='ignore'):
            confluence = np.nan_to_num(modulus_of_sum/sum_of_modulus)

        return self.symmetric_va / self.n * (2 - confluence)

    def evaluate(self) -> List[float]:
        return [self._symmetrical_magnitude_coefficient(),
                self._asymmetrical_magnitude_coefficient()]

    def plot(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        x, y = np.meshgrid(np.arange(0, self.u.shape[1], 1),
                           np.arange(0, self.v.shape[0], 1))

        ax.quiver(x, y, self.u, self.v)

        plt.show()

    def plot_asymmetric(self):
        x, y = np.meshgrid(np.arange(0, self.u.shape[1], 1),
                           np.arange(0, self.v.shape[0], 1))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.quiver(x, y, self.u * self.asymmetric_lattice,
                  self.v * self.asymmetric_lattice)

        plt.show()

    def plot_symmetric(self):
        x, y = np.meshgrid(np.arange(0, self.u.shape[1], 1),
                           np.arange(0, self.v.shape[0], 1))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.quiver(x, y, self.u * self.symmetric_lattice,
                  self.v * self.symmetric_lattice)

        plt.show()

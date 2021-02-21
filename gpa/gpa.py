import matplotlib.pyplot as plt
import numpy as np


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

    def _asymmetric_lattice(self):
        half_size = int(self.n / 2)
        odd = int(self.n % 2)

        modulus = np.ravel(self.modulus)
        modulus_first_half = modulus[0:half_size+odd]
        modulus_second_half = np.flip(modulus[half_size:])
        modulus = np.abs(modulus_first_half - modulus_second_half)
        modulus[modulus <= self.modulus_tolerance] = 0
        modulus[np.isnan(modulus)] = 0

        phases = np.ravel(self.phases)
        phases_first_half = phases[0:half_size+odd]
        phases_second_half = np.flip(phases[half_size:])
        phases = np.abs(phases_first_half) + np.abs(phases_second_half)
        phases[phases >= (np.pi - self.phases_tolerance)] = 0
        phases[np.isnan(phases)] = 0

        asymmetry_matrix = modulus + phases
        asymmetry_matrix = np.concatenate((
            asymmetry_matrix, np.flip(asymmetry_matrix[:half_size])))
        asymmetry_matrix = asymmetry_matrix.reshape(self.shape)

        asymmetry_matrix[asymmetry_matrix != 0] = 1.0

        self.va = np.sum(asymmetry_matrix)

        self.asymmetric_u = self.u * asymmetry_matrix
        self.asymmetric_u[self.asymmetric_u == -0.0] = 0

        self.asymmetric_v = self.v * asymmetry_matrix
        self.asymmetric_v[self.asymmetric_v == -0.0] = 0

        self.asymmetric_modulus = np.sqrt(
            self.asymmetric_u ** 2 + self.asymmetric_v ** 2)
        self.asymmetric_phases = np.arctan2(
            self.asymmetric_v, self.asymmetric_u)

    def _asymmetrical_magnitude_coefficient(self):
        modulus_of_sum = np.sqrt((np.sum(self.asymmetric_u))**2 +
                                 (np.sum(self.asymmetric_v))**2)
        sum_of_modulus = np.sum(self.asymmetric_modulus)

        with np.errstate(divide='ignore', invalid='ignore'):
            confluence = np.nan_to_num(modulus_of_sum/sum_of_modulus)

        return self.va / self.n * (2 - confluence)

    def evaluate(self):
        return self._asymmetrical_magnitude_coefficient()

    def plot(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        x, y = np.meshgrid(np.arange(0, self.u.shape[1], 1),
                           np.arange(0, self.v.shape[0], 1))

        ax.quiver(x, y, self.u, self.v)

        plt.show()

    def plot_asymmetric(self):
        x, y = np.meshgrid(np.arange(0, self.asymmetric_u.shape[1], 1),
                           np.arange(0, self.asymmetric_v.shape[0], 1))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.quiver(x, y, self.asymmetric_u, self.asymmetric_v)

        plt.show()

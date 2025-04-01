from dataclasses import dataclass
from functools import cached_property
from . import constants


@dataclass(frozen=True)
class Atom:
    """
    Data class for an atom.

    Attributes:
        mass (float): Mass of the atom in kg.
        gF (float): Effective g-factor (gF), describing the magnetic moment.
        mF (float): Magnetic quantum number (mF).
        s (float): s-wave scattering length in meters. The scattering length is a measure of the effective range
                   of the interaction between two atoms in a cold atomic gas.
    """

    mass: float  # kg
    gF: float  # Effective g-factor for the full hyperfine state (orbital, spin, and nuclear spin).
    mF: float  # Magnetic quantum number
    s: float  # s-wave scattering length (m)

    @cached_property
    def mu(self) -> float:
        """
        Calculate the effective magnetic moment of the atom.

        Returns:
            float: Effective magnetic moment in J/T.
        """
        muB = constants.muB
        return self.gF * self.mF * muB

    def potential_energy(self, B_mag: float, z: float) -> float:
        """
        Calculate the potential energy of the atom in a magnetic field.

        Zeeman Effect: This is an approximation where the energy difference induced by
        the external magnetic field is small compared with the hyperfine splitting.
        In this case, the energy level splitting of each hyperfine state is proportional
        to the quantum number mF and the effective g-factor gF.

        Gravitational Potential Energy: The potential energy of the atom due to gravity,
        where the atom chip is oriented vertically in the upside-down configuration.

        Assumptions:
            - The magnetic field and the atom's magnetic moment are aligned.
            - The magnetic field strength is non-zero everywhere in the trapping region.
            - The atom's kinetic energy is negligible (cold atoms).
            - The atom is treated as a point particle.

        Args:
            B_mag (float): Magnitude of the magnetic field in Gauss.
            z (float): Height of the atom in millimeters.

        Returns:
            float: Potential energy of the atom in J.
        """

        # Zeeman effect and gravitational potential energy
        # - Convert B_mag from Gauss to Tesla
        # - Convert z from mm to m
        g = constants.g
        return self.mu * B_mag * 1e-4 - self.mass * g * z * 1e-3


# Define the properties of the rubidium-87 atom
# fmt: off
rb87 = Atom(
    mass = 1.44316e-25,  # Mass (kg)
    gF   = 0.5,          # Landé g-factor (F=2 state)
    mF   = 2,            # Magnetic quantum number
    s    = 5.2e-9,       # s-wave scattering length
)
# fmt: on

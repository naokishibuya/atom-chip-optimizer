from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import constants


class Atom(NamedTuple):
    mass: float  # Mass (kg)
    gF: float  # Landé g-factor
    mF: float  # Magnetic quantum number
    a_s: float  # s-wave scattering length (m): an effective interaction range between two atoms in a cold atomic gas.


# Define the properties of the rubidium-87 atom
rb87 = Atom(mass=1.44316e-25, gF=0.5, mF=2, a_s=5.2e-9)


@jax.jit
def magnetic_moment(atom: Atom) -> float:
    return atom.gF * atom.mF * constants.muB


@jax.jit
def trap_potential_energy(atom: Atom, B_mag: jnp.ndarray, z: float) -> jnp.ndarray:
    """
    Computes the total potential energy of an atom in a magnetic trap.

    The energy includes:
      - Zeeman potential: μ = gF * mF * μB * B
      - Gravitational potential: U = m * g * z

    Assumes:
      - Magnetic field is aligned with the atomic moment.
      - B_mag is in Gauss; z is in mm.
      - Energy is returned in joules.

    Args:
        atom (Atom): Atom parameters (mass, gF, mF).
        B_mag (jnp.ndarray): Magnetic field magnitude [Gauss].
        z (float): Height above chip center [mm].

    Returns:
        jnp.ndarray: Potential energy [J].
    """
    mu = magnetic_moment(atom)
    return mu * B_mag * 1e-4 - atom.mass * constants.g * z * 1e-3


@jax.jit
def gravity_equivalent_field(atom: Atom, z: float) -> float:
    """
    Calculate the gravitational equivalent of the magnetic field.
    """
    mu = magnetic_moment(atom)
    return atom.mass * constants.g * z * 1e-3 / mu * 1e4


@jax.jit
def gravity_equivalent_potential(atom: Atom, z: float) -> float:
    """
    Calculate the gravitational equivalent potential energy.
    """
    return atom.mass * constants.g * z * 1e-3

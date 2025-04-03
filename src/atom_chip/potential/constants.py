from typing import Final


# Bohr magneton, J/T (Joules per Tesla)
# ----------------------------------------------------
# It is the unit of magnetic moment in atomic physics
# defined as $\mu_B = \frac{e\hbar}{2m_e}$, where:
# - $e$ is the elementary charge,
# - $\hbar$ is the reduced Planck constant,
# - $m_e$ is the electron rest mass.
# ----------------------------------------------------
muB: Final[float] = 9.274009994e-24  # J/T

# Gravitational acceleration
g: Final[float] = 9.81  # m/s²

# Boltzmann constant, J/K
kB: Final[float] = 1.38065e-23

# Planck constant, J·s
h: Final[float] = 6.62607015e-34

# Reduced Planck constant, J·s
hbar: Final[float] = 1.0545718e-34


# Utility functions
def joule_to_microKelvin(E: float) -> float:
    """
    Convert energy from Joules to microKelvin.
    Args:
        E (float): Energy in Joules.
    Returns:
        float: Energy in microKelvin.
    """
    return E * 1e6 / kB

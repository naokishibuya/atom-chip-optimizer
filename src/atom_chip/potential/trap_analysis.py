"""
This module analyzes magnetic trapping potentials to derive relevant BEC properties under different physical regimes.

Analysis is divided into three parts:

Part 1 — Trap Characterization:
- Locate trap minimum (potential or magnetic field)
- Compute Hessian and extract trap geometry
- Calculate trap frequencies (ωₓ, ωᵧ, ω_z)
- Evaluate Larmor frequency for adiabaticity checks

Part 2 — BEC Analysis (Non-Interacting Model):
- Estimate spatial extent (harmonic oscillator radii)
- Compute average harmonic oscillator length and frequency
- Calculate critical temperature and non-interacting chemical potential

Part 3 — BEC Analysis (Interacting / Thomas-Fermi Limit):
- Compute Thomas-Fermi radii and chemical potential
- Valid for high atom numbers or strong interactions (mean-field regime)

Notes:
- Use Part 2 for small atom numbers and weak interactions (Gaussian ground state)
- Use Part 3 for large condensed atom numbers (interaction-dominated regime)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from . import constants
from .atom import Atom
from .minimum import search_minimum, MinimumResult
from .hessian import hessian_at_minimum, Hessian


logging.basicConfig(level=logging.INFO, format="")
logger = logging.getLogger(__name__)


# fmt: off
@register_dataclass
@dataclass
class Frequency:
    frequency: jnp.ndarray = field(default_factory=lambda: jnp.nan)  # [Hz]
    angular  : jnp.ndarray = field(default_factory=lambda: jnp.nan)  # [rad/s]
# fmt: on


# Non-interacting BEC properties based on quantum harmonic oscillator model (ideal gas)
# Assumptions:
#  - Atom number is low
#  - Scattering length is small (weak interactions)
#  - We're mostly interested in qualitative behavior or tight traps
#  - We want fast estimates without solving the GPE
# then,
#  - Gaussian wavefunction
#  - Radii ≈ harmonic oscillator lengths
#  - Chemical potential ~ zero-point energy
# fmt: off
@register_dataclass
@dataclass
class BECAnalysis:
    total_atoms: int         = field(default_factory=lambda: 0)        # Total number of atoms in the trap
    a_ho       : float       = field(default_factory=lambda: jnp.nan)  # H.O. length [m]
    w_ho       : float       = field(default_factory=lambda: jnp.nan)  # Geometric mean trap frequency (angular) [rad/s]
    mu_0       : float       = field(default_factory=lambda: jnp.nan)  # Ground level chemical potential limit [J]
    radii      : jnp.ndarray = field(default_factory=lambda: None)     # Harmonic oscillator length [m]
    T_c        : float       = field(default_factory=lambda: jnp.nan)  # Critical temperature [K]
# fmt: on


# Interacting BEC properties in the Thomas-Fermi approximation (mean-field limit)
# Assumptions:
#  - Condensed atom number is large
#  - Interactions dominate over kinetic energy
#  - We're deep in the BEC regime and want quantitative accuracy
#  - The trap is relatively loose (large spatial extent)
# then,
#  - Inverted-parabola density profile
#  - Much larger radii (TF radii)
#  - Chemical potential dominated by interaction energy
#  - Kinetic energy negligible
# fmt: off
@register_dataclass
@dataclass
class TFAnalysis:
    condensed_atoms: int         = field(default_factory=lambda: 0)       # Number of condensed atoms in the trap
    mu             : float       = field(default_factory=lambda: jnp.nan) # Thomas-Fermi chemical potential [J]
    radii          : jnp.ndarray = field(default_factory=lambda: None)    # Thomas-Fermi radii in each direction [m]
# fmt: on


@dataclass
class AnalysisOptions:
    search: dict
    hessian: dict
    total_atoms: int = 1e5
    condensed_atoms: int = 1e3

    def __post_init__(self):
        if self.condensed_atoms < 0:
            raise ValueError("Condensed atoms must be 0 or greater.")
        if self.total_atoms < self.condensed_atoms:
            raise ValueError("Total atoms must be greater than condensed atoms.")


# fmt: off
@dataclass
class FieldAnalysis:
    minimum: MinimumResult
    hessian: Hessian   = field(default_factory=lambda: Hessian())
    trap   : Frequency = field(default_factory=lambda: Frequency())
    larmor : Frequency = field(default_factory=lambda: Frequency())
# fmt: on


# fmt: off
@dataclass
class PotentialAnalysis:
    minimum: MinimumResult
    hessian: Hessian     = field(default_factory=lambda: Hessian())
    trap   : Frequency   = field(default_factory=lambda: Frequency())
    larmor : Frequency   = field(default_factory=lambda: Frequency())
    bec    : BECAnalysis = field(default_factory=lambda: BECAnalysis())
    tf     : TFAnalysis  = field(default_factory=lambda: TFAnalysis())
# fmt: on


# === Field Analysis is done mostly for debugging purposes ===


def analyze_field(
    atom: Atom,
    field_function: Callable[[jnp.ndarray], float],
    options: AnalysisOptions,
) -> FieldAnalysis:
    """
    This analyzes the magnetic field and related characteristics.
    """

    logger.info("-" * 100)
    logger.info("Analyzing magnetic field [G] ...")

    # Define the objective function for the field analysis
    def objective(point: jnp.array) -> float:
        B_mag, _ = field_function(point)
        return B_mag[0]

    # Step 1: Potential Minimum
    logger.info(options.search)
    minimum = search_minimum(objective, **options.search)
    if minimum.found:
        logger.info(
            "Minimum {:.10g}G @ x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(minimum.value, *minimum.position)
        )
    else:
        logger.info(f"Optimization failed: {minimum.message}")
        return PotentialAnalysis(minimum)

    # Step 2: Hessian
    hessian = hessian_at_minimum(objective, minimum.position, **options.hessian)
    logger.info(f"Hessian by {options.hessian}")
    logger.info(hessian.eigenvalues)
    logger.info(hessian.eigenvectors)
    logger.debug(hessian.matrix)

    # Step 3: Trap Frequencies (1e-4 for conversion from G to T)
    eigenvalues = atom.mu * hessian.eigenvalues * 1e-4  # don't modify the hessian matrix!
    trap = trap_frequencies(eigenvalues, atom.mass)
    logger.info(f"Trap frequencies (Hz) : {trap.frequency}")

    # Step 4: Larmor Frequency
    larmor = larmor_frequency(atom, minimum.value)
    logger.info(f"Larmor frequency (MHz): {larmor.frequency * 1e-6}")

    return FieldAnalysis(
        minimum=minimum,
        hessian=hessian,
        trap=trap,
        larmor=larmor,
    )


# === Trap Analysis is the main purpose ===


def analyze_trap(
    atom: Atom,
    potential_function: Callable[[jnp.ndarray], float],
    options: AnalysisOptions,
) -> PotentialAnalysis:
    """
    Finds the minimum of the trap potential, calculates relevant properties,
    and returns a TrapAnalysis object containing the results.

    Args:
        atom (Atom): The atom object containing the properties of the atom.
        potential_function (Callable): Function to compute the potential energy at given points.
        options (AnalysisOptions): Options for the trap analysis.

    Returns:
        TrapAnalysis: An object containing the results of the trap analysis.
    """

    # ----------------------------------------------------------------------
    # Part 1: Trap Characterization
    # ----------------------------------------------------------------------

    logger.info("-" * 100)
    logger.info("Analyzing trap potential [J] ...")

    # Define the objective function for the trap analysis
    def objective(point: jnp.array) -> float:
        E, _, _ = potential_function(point)
        return E[0]

    # Step 1: Potential Minimum
    logger.info(options.search)
    minimum = search_minimum(objective, **options.search)
    if minimum.found:
        logger.info(
            "Minimum {:.10g}J @ x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(minimum.value, *minimum.position)
        )
    else:
        logger.info(f"Optimization failed: {minimum.message}")
        return PotentialAnalysis(minimum)

    # Step 2: Hessian
    hessian = hessian_at_minimum(objective, minimum.position, **options.hessian)
    logger.info(f"Hessian by {options.hessian}")
    logger.info(hessian.eigenvalues)
    logger.info(hessian.eigenvectors)
    logger.debug(hessian.matrix)

    # Step 3: Trap Frequencies
    trap = trap_frequencies(hessian.eigenvalues, atom.mass)
    logger.info(f"Trap frequencies (Hz)   : {trap.frequency}")

    # Step 4: Larmor Frequency
    _, B_mag, _ = potential_function(minimum.position)
    larmor = larmor_frequency(atom, B_mag[0])
    logger.info(f"Larmor frequency (MHz)  : {larmor.frequency * 1e-6}")
    logger.info(f"Larmor frequency (rad/s): {larmor.angular}")

    # -----------------------------------------------------------------------
    # Part 2: BEC Analysis (Non-Interacting Limit)
    # -----------------------------------------------------------------------

    # Step 5: BEC Analysis (Non-Interacting Limit)
    bec_analysis = analyze_bec_non_interacting(
        atom=atom,
        trap_frequency=trap,
        total_atoms=options.total_atoms,
    )
    logger.info("BEC Analysis (Non-Interacting Limit):")
    logger.info(f"  a_ho (m)    : {bec_analysis.a_ho}")
    logger.info(f"  w_ho (rad/s): {bec_analysis.w_ho}")
    logger.info(f"  mu_0 (J)    : {bec_analysis.mu_0}")
    logger.info(f"  Radii (m)   : {bec_analysis.radii}")
    logger.info(f"  T_c (K)     : {bec_analysis.T_c}")

    # Step 6: BEC Analysis (Thomas-Fermi Limit)
    tf_analysis = analyze_bec_thomas_fermi(
        atom=atom,
        trap_frequency=trap,
        condensed_atoms=options.condensed_atoms,
    )
    logger.info("BEC Analysis (Thomas-Fermi Limit):")
    logger.info(f"  mu (J)      : {tf_analysis.mu}")
    logger.info(f"  Radii (m)   : {tf_analysis.radii}")

    return PotentialAnalysis(
        minimum=minimum,
        hessian=hessian,
        trap=trap,
        larmor=larmor,
        bec=bec_analysis,
        tf=tf_analysis,
    )


# === Part 1: Trap Characterization ===


@jax.jit
def trap_frequencies(eigenvalues: jnp.ndarray, mass: float):
    """
    Calculates trap frequencies from Hessian eigenvalues.

    Oscillation frequencies of atoms in the trap.
    (how fast atoms oscillate in the trap)
    """
    eigenvalues = eigenvalues * 1e6  # J/mm^2 -> J/m^2
    angular = jnp.sqrt(eigenvalues / mass)
    frequency = angular / (2 * jnp.pi)
    return Frequency(frequency, angular)


@jax.jit
def larmor_frequency(atom: Atom, B_mag: jnp.ndarray) -> Frequency:
    """
    Calculates Larmor frequency from magnetic field strength.

    Precession rate of an atom's magnetic moment in an external magnetic field.
    (how fast tiny internal magnets wobble in a magnetic field)
    """
    # B_mag is in [G]: convert to [T] by 1e-4
    # mu B_mag is in [J/T] where mu = gF mF muB
    angular = atom.mu * B_mag * 1e-4 / constants.hbar
    frequency = angular / (2 * jnp.pi)
    return Frequency(frequency, angular)


# === Part 2: BEC Analysis (Non-Interacting Limit) ===
@jax.jit
def analyze_bec_non_interacting(
    atom: Atom,
    trap_frequency: Frequency,
    total_atoms: int,
) -> BECAnalysis:
    """
    Analyzes the non-interacting BEC properties based on the harmonic oscillator model.
    Assumes low atom number and weak interactions.

    Harmonic oscillator length (a_ho) gives a sense of the spatial extent of the ground state wavefunction of
    the harmonic oscillator.

    Chemical potential at T=0 (mu_0) is the energy of the ground state of the harmonic oscillator.
    At absolute zero, all particles in a non-interacting BEC will occupy the lowest energy state (the ground state).
    """
    omega = trap_frequency.angular
    w_ho = jnp.prod(omega) ** (1 / 3)  # geometric mean of trap frequencies
    a_ho = jnp.sqrt(constants.hbar / (atom.mass * w_ho))
    mu_0 = 0.5 * constants.hbar * jnp.sum(omega)  # ground state energy of the harmonic oscillator
    radii = jnp.sqrt(constants.hbar / (atom.mass * omega))
    T_c = 0.94 * constants.hbar / constants.kB * w_ho * total_atoms ** (1 / 3)
    return BECAnalysis(
        total_atoms=total_atoms,
        a_ho=a_ho,
        w_ho=w_ho,
        mu_0=mu_0,
        radii=radii,
        T_c=T_c,
    )


# === Part 3: BEC Analysis (Thomas-Fermi / Interacting Limit) ===
@jax.jit
def analyze_bec_thomas_fermi(
    atom: Atom,
    trap_frequency: Frequency,
    condensed_atoms: int,
) -> TFAnalysis:
    """
    Analyzes the Thomas-Fermi BEC properties based on the mean-field limit.
    Assumes large condensed atom number and strong interactions.
    (Thomas-Fermi approximation neglects kinetic energy)

    The strong repulsive interactions cause the condensate to expand and its density profile becomes
    inverted parabolic or a lens shape within the spatial extent of the trap.

    The density is nearly uniform in the center and drops to zero at the edges, defined by the Thomas-Fermi radius.
    This shape arises because the interaction energy dominates over the kinetic energy, and the condensate
    expands until the outward pressure from interactions balances the inward trapping potential.
    """
    omega = trap_frequency.angular
    w_ho = jnp.prod(omega) ** (1 / 3)  # geometric mean
    a_ho = jnp.sqrt(constants.hbar / (atom.mass * w_ho))
    mu = 0.5 * constants.hbar * w_ho * (15 * atom.a_s * condensed_atoms / a_ho) ** (2 / 5)
    radii = jnp.sqrt(2 * mu / atom.mass) / omega
    return TFAnalysis(
        condensed_atoms=condensed_atoms,
        mu=mu,
        radii=radii,
    )

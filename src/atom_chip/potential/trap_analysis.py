from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
from .atom import Atom
from .minimum import search_minimum, MinimumResult
from .hessian import hessian_at_minimum, Hessian


@dataclass
class AnalysisOptions:
    search: dict
    hessian: dict
    verbose: bool = False


@dataclass
class TrapFrequency:
    frequency: jnp.ndarray  # [Hz]
    angular: jnp.ndarray  # [rad/s]


@dataclass
class FieldAnalysis:
    minimum: MinimumResult
    hessian: Optional[Hessian] = None
    trap_frequency: Optional[TrapFrequency] = None


@dataclass
class TrapAnalysis:
    minimum: MinimumResult
    hessian: Optional[Hessian] = None
    trap_frequency: Optional[TrapFrequency] = None


def analyze_field(atom: Atom, function: Callable[[jnp.ndarray], float], options: AnalysisOptions) -> TrapAnalysis:
    # search for the minimum
    print("-" * 100)
    print("Searching for field minimum...[G]")
    print(options.search)
    minimum = search_minimum(function, **options.search)
    if not minimum.found:
        return TrapAnalysis(minimum)

    if options.verbose:
        print("Minimum {:.10g} found @ x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(minimum.value, *minimum.point))

    # compute the hessian at the minimum
    hessian = hessian_at_minimum(function, minimum.point, **options.hessian)
    if options.verbose:
        print(f"Hessian: {options.hessian}")
        print(hessian.eigenvalues)
        print(hessian.eigenvectors)
        # print(hessian.matrix)

    # compute the trap frequencies
    # convert from [G/mm^2] to [J/m^2]
    # 100 = 1e-4 * 1e6 ([G to T] * [/mm^2 to /m^2])
    eigenvalues = atom.mu * hessian.eigenvalues * 100  # don't modify the hessian matrix!
    trap_frequency = _trap_frequency(atom, eigenvalues)
    if options.verbose:
        print("Trap frequencies (Hz):", trap_frequency.frequency)

    return FieldAnalysis(minimum, hessian, trap_frequency)


def analyze_trap(atom: Atom, function: Callable[[jnp.ndarray], float], options: AnalysisOptions) -> TrapAnalysis:
    # search for the minimum
    print("-" * 100)
    print("Searching for potential minimum...[J]")
    print(options.search)
    minimum = search_minimum(function, **options.search)
    if not minimum.found:
        return TrapAnalysis(minimum)

    if options.verbose:
        print("Minimum {:.10g} found @ x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(minimum.value, *minimum.point))

    # compute the hessian at the minimum
    hessian = hessian_at_minimum(function, minimum.point, **options.hessian)
    if options.verbose:
        print(f"Hessian: {options.hessian}")
        print(hessian.eigenvalues)
        print(hessian.eigenvectors)
        # print(hessian.matrix)

    # compute the trap frequencies
    # convert from [J/mm^2] to [J/m^2] by 1/(1e-3 m)^2 = 1e6 m^2
    eigenvalues = hessian.eigenvalues * 1e6  # don't modify the hessian matrix!
    trap_frequency = _trap_frequency(atom, eigenvalues)
    if options.verbose:
        print("Trap frequencies (Hz):", trap_frequency.frequency)

    return TrapAnalysis(minimum, hessian, trap_frequency)


def _trap_frequency(atom: Atom, eigenvalues: jnp.ndarray) -> TrapFrequency:
    """
    The eigenvalues of the Hessian matrix at the minimum are the trap frequencies.
    The eigenvalues are expected to be in J/m^2.
    """

    # compute the trap frequencies
    omega = jnp.sqrt(eigenvalues / atom.mass)
    frequency = omega / (2 * jnp.pi)  # convert to Hz
    angular = omega  # convert to rad/s
    return TrapFrequency(frequency, angular)

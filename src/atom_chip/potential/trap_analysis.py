from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
from .minimum import search_minimum, MinimumResult
from .hessian import hessian_at_minimum, Hessian


@dataclass
class AnalysisOptions:
    search: dict
    hessian: dict
    verbose: bool = False


@dataclass
class FieldAnalysis:
    minimum: MinimumResult
    hessian: Optional[Hessian] = None


@dataclass
class TrapAnalysis:
    minimum: MinimumResult
    hessian: Optional[Hessian] = None


def analyze_field(function: Callable[[jnp.ndarray], float], options: AnalysisOptions) -> TrapAnalysis:
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

    return FieldAnalysis(minimum, hessian)


def analyze_trap(function: Callable[[jnp.ndarray], float], options: AnalysisOptions) -> TrapAnalysis:
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

    return TrapAnalysis(minimum, hessian)

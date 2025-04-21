from dataclasses import dataclass
import logging
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from ..atom_chip import AtomChip


@dataclass
class EvaluatorResult:
    success: bool
    E: float
    B_mag: float
    trap_depth: float
    grad_val: float
    hessian: jnp.ndarray
    x: float
    y: float
    z: float


class Evaluator:
    def __init__(self, atom_chip_maker: Callable, search_options: dict):
        self.atom_chip_maker = atom_chip_maker
        self.search_options = search_options

        self.atom_chip = None
        self._x_last = None
        self._val_last = None

    def _as_hashable(self, params):
        # Convert to a hashable key for caching
        return np.asarray(params).tobytes()

    def evaluate(self, params: jnp.ndarray):
        atom_chip = self.atom_chip_maker(params)
        self.result = _evaluate(atom_chip, self.search_options)
        return self.result


def _evaluate(atom_chip: AtomChip, search_options: dict) -> EvaluatorResult:
    """
    Evaluator the atom chip potential and its properties.
    Args:
        atom_chip (AtomChip): The atom chip object.
        search_options (dict): The search options for optimization.
    Returns:
        EvaluatorResult: The result of the evaluation.
    """

    # objective function
    def get_potential_energy(pos):
        E, _, _ = atom_chip.get_potentials(pos)
        return E[0]

    # minimize the potential energy
    result = minimize(get_potential_energy, **search_options)
    if not result.success:
        logging.error(f"Optimization failed: {result.message}")
        return EvaluatorResult(False, np.nan, np.nan, np.nan, np.nan, None)

    # extract the results
    potential_minimum = result.x

    # gradient
    grad_fn = jax.grad(get_potential_energy)
    grad_val = jnp.linalg.norm(grad_fn(potential_minimum))

    # hessian
    hessian_fn = jax.hessian(get_potential_energy)
    hessian = hessian_fn(potential_minimum)

    # magnetic field
    E, B_mag, _ = atom_chip.get_potentials(potential_minimum)
    E, B_mag = E[0], B_mag[0]

    # trap depth
    E_far = get_potential_energy(jnp.array([0.0, 0.0, 2.0]))
    trap_depth = E_far - E

    x, y, z = result.x

    return EvaluatorResult(
        success=True,
        E=E,
        B_mag=B_mag,
        trap_depth=trap_depth,
        grad_val=grad_val,
        hessian=hessian,
        x=x,
        y=y,
        z=z,
    )

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize


@dataclass
class Hessian:
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    matrix: jnp.ndarray


@dataclass
class PotentialMinimum:
    """
    Result of the minimum search.
    """

    found: bool
    value: jnp.ndarray  # Value of the function at the minimum
    point: jnp.ndarray  # Position of the minimum [x, y, z] in mm
    grads: jnp.ndarray  # Gradient at the minimum
    hessian: Hessian


def search_minimum_potential(
    func,
    initial_guess,
    max_iterations=int(1e5),
    learning_rate=1e-2,
    tolerance=1e-6,
) -> PotentialMinimum:
    """
    Perform gradient descent to find the minimum of a function.

    Args:
        func: The function to minimize. It should take a single argument (the parameters) and return a scalar value.
        initial_guess: Initial guess for the minimum point.
        max_iterations: Number of iterations to perform.
        learning_rate: Learning rate for gradient descent.
        tolerance: Tolerance for convergence.

    Returns:
        The parameters that minimize the function and the minimum value of the function.
    """

    def objective_function(point):
        value = func(point)[0]
        return value[0]

    optres = minimize(
        objective_function,
        x0=initial_guess.flatten(),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 10000, "maxfev": 10000, "disp": True},
        tol=1e-6,
    )
    if optres.success:
        initial_guess = jnp.array(optres.x).reshape(initial_guess.shape)

    # Initialize the optimizer
    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.99,
    )
    optimizer = optax.sgd(learning_rate=lr_schedule)
    opt_state = optimizer.init(initial_guess)
    params = initial_guess

    found = False
    for i in range(max_iterations):
        # Compute the gradients
        grads = jax.grad(objective_function)(params)

        # Update the parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Check for convergence
        if jnp.linalg.norm(grads) < tolerance:
            found = True
            break

    # TODO
    # hessian_func = jax.hessian(objective_function)
    # hessian_matrix = hessian_func(params)[0]
    # print(hessian_matrix.shape)
    # eigenvalues, eigvectors = jnp.linalg.eig(hessian_matrix)

    result = PotentialMinimum(
        found=found,
        value=objective_function(params),
        point=params[0],
        grads=grads[0],
        hessian=None,
        # hessian=Hessian(
        #     eigenvalues=eigenvalues,
        #     eigenvectors=eigvectors,
        #     matrix=hessian_matrix,
        # ),
    )

    return result

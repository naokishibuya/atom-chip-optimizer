import os
from typing import Callable
import numpy as np
from jax import numpy as jnp
import atom_chip as ac
from atom_chip.components import RectangularConductor


# fmt: off
def make_chip_z(
    material     : str = "copper",
    current      : float = 1.0,
    center_length: float = 10.0,
    center_width : float = 0.3,
    center_height: float = 0.07,
    leg_length   : float = 30.0,
    leg_width    : float = 0.3,
    leg_height   : float = 0.07,
    stray_x      : float = 1.0,
    stray_y      : float = 1.0,
    z_offset     : float = 0.485,
):
    # Convention
    # x-axis: wire length (initial directional axis)
    # y-axis: wire width
    # z-axis: wire height

    cl, ll = center_length, leg_length
    cw, lw = center_width , leg_width
    ch, lh = center_height, leg_height

    segments = [
        [(-cl/2 + lw/2, -ll - cw/2, 0), (-cl/2 + lw/2,     -cw/2, 0), lw, lh],
        [(-cl/2       ,          0, 0), ( cl/2       ,         0, 0), cw, ch],
        [( cl/2 - lw/2,       cw/2, 0), ( cl/2 - lw/2, ll + cw/2, 0), lw, lh],
    ]

    for i in range(len(segments)):
        segments[i][0] = (segments[i][0][0], segments[i][0][1], z_offset)
        segments[i][1] = (segments[i][1][0], segments[i][1][1], z_offset)

    component = RectangularConductor(material, current, segments)

    bisa_fiels = ac.field.BiasFields(
        coil_factors=(-1.068, 1.8, 3.0),
        currents=(0, 0, 0.0),
        stray_fields=(stray_x, stray_y, 0.0)
    )

    atom_chip = ac.AtomChip(
        name        = "Chip-Z",
        atom        = ac.rb87,
        components  = [component],
        bias_fields = bisa_fiels,
    )
    return atom_chip
# fmt: on


def main():
    # fmt: off
    options = ac.potential.AnalysisOptions(
        search = dict(
            x0      = [0.0, 0.0, 0.5],  # Initial guess
            bounds  = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)],
            method  = "Nelder-Mead",
            options = dict(
                xatol   = 1e-10,
                fatol   = 1e-10,
                maxiter = int(1e5),
                maxfev  = int(1e5),
                disp    = False,
            ),
        ),
        hessian = dict(
            method = "jax",
            # method = "finite-difference",
            # hessian_step = 1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
        verbose = True,
    )
    # fmt: on

    # objective
    def atom_chip_maker(params: jnp.ndarray) -> Callable:
        center_length = params[0]
        stray_x = params[1]
        return make_chip_z(
            center_length=center_length,
            stray_x=stray_x,
            stray_y=5.0,
        )

    evaluator = ac.search.Evaluator(
        atom_chip_maker=atom_chip_maker,
        search_options=options.search,
    )

    def objective(params: jnp.ndarray) -> float:
        result = evaluator.evaluate(params)
        if not result.success:
            return np.inf
        hessian = result.hessian
        eigvals, eigvecs = jnp.linalg.eigh(hessian)
        x_axis = jnp.array([1.0, 0.0, 0.0])
        projections = jnp.abs(jnp.dot(eigvecs.T, x_axis))
        x_idx = jnp.argmax(projections)
        return float(eigvals[x_idx])  # minimize x-curvature

    def contraints(params: jnp.ndarray) -> jnp.ndarray:
        result = evaluator.evaluate(params)
        if not result.success:
            return jnp.array([1.0, 1.0, 1.0])
        return jnp.array(
            [
                result.grad_val,  # ≈ 0
                result.B_mag - 0.5,  # ≥ 0
                result.trap_depth - 1e-30,  # ≥ 0
            ]
        )

    params = jnp.array([10.0, 5.0])  # center_length, stray_x

    plotter = ac.search.CallbackPlotter(evaluator)

    result = ac.search.optimize(
        objective=objective,
        constraints=contraints,
        lower_bounds=jnp.array([0.0, 0.0, 0.0]),
        upper_bounds=jnp.array([1e-6, 10.0, np.inf]),  # some tiny wiggle room for gradient ≈ 0
        params=params,
        callback=plotter.callback,
    )
    if not result.success:
        return

    # Extract the optimized parameters
    center_length, stray_x = result.params

    print("\n=== Optimization Complete ===")
    print(f"center_length: {center_length:.4f} mm")
    print(f"stray_x      : {stray_x:.4f} G")

    atom_chip = atom_chip_maker(result.params)
    atom_chip.analyze(options)

    # Export the atom chip to JSON
    directory = os.path.dirname(__file__)
    atom_chip.save(os.path.join(directory, "chip_z.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    main()

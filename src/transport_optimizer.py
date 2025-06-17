import argparse
from typing import Callable, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np  # formatting arrays only
from scipy.optimize import minimize
import atom_chip as ac
import transport_initializer
from transport_reporter import attrdict, report_results


# Precompute the wire layout and segment counts
SHIFTING_WIRES, GUIDING_WIRES = transport_initializer.setup_wire_layout()
GUIDING_WIRE_SEGMENT_COUNTS = jnp.array([len(wire) for wire in GUIDING_WIRES], dtype=jnp.int32)


# ----------------------------------------------------------------------------------------------------
# Setup the wire configuration for the atom chip.
#
# This can be JIT-compiled but it's not necessary since it runs only once.
# ----------------------------------------------------------------------------------------------------
def setup_wire_config() -> ac.atom_chip.WireConfig:
    """
    Build the wire configuration for the atom chip.
    """
    starts = []
    ends = []
    widths = []
    heights = []

    # Collect segments for shifting and guiding wires
    for segment in SHIFTING_WIRES:
        s, e, w, h = segment
        starts.append(jnp.array(s, dtype=jnp.float64))
        ends.append(jnp.array(e, dtype=jnp.float64))
        widths.append(jnp.array(w, dtype=jnp.float64))
        heights.append(jnp.array(h, dtype=jnp.float64))

    for wire in GUIDING_WIRES:
        for segment in wire:
            s, e, w, h = segment
            starts.append(jnp.array(s, dtype=jnp.float64))
            ends.append(jnp.array(e, dtype=jnp.float64))
            widths.append(jnp.array(w, dtype=jnp.float64))
            heights.append(jnp.array(h, dtype=jnp.float64))

    # Convert to WireConfig
    wire_config = ac.atom_chip.WireConfig(
        starts=jnp.stack(starts),
        ends=jnp.stack(ends),
        widths=jnp.stack(widths),
        heights=jnp.stack(heights),
    )
    return wire_config


# ----------------------------------------------------------------------------------------------------
# Calculate the target position of the trap.
#
# This can be JIT-compiled but it's not necessary since it runs only once.
# ----------------------------------------------------------------------------------------------------
def calculate_destination(
    r0: jnp.ndarray,
    num_shifts: int,
    shifting_wire_distance: float,  # mm
) -> jnp.ndarray:
    """
    Calculate the target position of the trap after a number of shifts.
    """
    distance = num_shifts * shifting_wire_distance
    return r0 + distance * jnp.array([1.0, 0.0, 0.0])


# ----------------------------------------------------------------------------------------------------
# Find the local minimum of the magnetic potential energy.
#
# This is not a JIT function because it uses `minimize` from SciPy
# ----------------------------------------------------------------------------------------------------
def find_trap_minimum(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasConfig,
    wire_currents: jnp.ndarray,
    guess: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find the local minimum of the magnetic potential energy.
    """

    def objective(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    # scipy minimize
    result = minimize(objective, **make_search_options(initial_guess=guess))
    if not result.success:
        print(f"Minimization failed: {result.message} result.x={result.x} guess={guess}")
    return result.x  # shape (3,)


def make_search_options(initial_guess) -> ac.potential.AnalysisOptions:
    """
    Create analysis options for the trap potential search.
    """
    x, y, _ = initial_guess
    bounds = [
        (x - 0.5, x + 0.5),  # x bounds
        (y - 0.5, y + 0.5),  # y bounds
        (0.0, 1.0),  # z bounds
    ]
    return dict(
        x0=initial_guess,  # Initial guess
        bounds=bounds,
        method="Nelder-Mead",
        options=dict(
            xatol=1e-10,
            fatol=1e-10,
            maxiter=int(1e5),
            maxfev=int(1e5),
            disp=False,
        ),
    )


def optimize_initial_currents(
    params,
    *,
    target_U=7.5e-28,
    z_min=0.35,
    lr=1e-2,
    iters=1000,
    w_U=1.0,
    w_x=10.0,
    w_y=20.0,
    w_z=1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adjust the static wire currents so that
      • U0 ≃ target_U,
      • (x0,y0,z0) ≃ (0,0,≥z_min),
      • with soft penalties in y and z (x is loose).
    """
    atom = transport_initializer.ATOM
    wire_cfg = setup_wire_config()
    bias_cfg = ac.field.ZERO_BIAS_CONFIG

    # initial logical currents
    I_shift = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guide = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_vec = jnp.concatenate([I_shift, I_guide])  # shape (N,)

    n_shift = I_shift.shape[0]
    n_total = I_vec.shape[0]
    I_limits = jnp.concatenate(
        [
            jnp.ones(n_shift) * params.I_max_shifting,
            jnp.ones(n_total - n_shift) * params.I_max_guiding,
        ]
    )

    # --- JAX jitted helpers -----------------------------------------
    @jax.jit
    def trap_U(r, I_wires):
        # r: (3,), I: (N,)
        wire_currents = distribute_currents_to_wires(I_wires)
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_cfg, wire_currents, bias_cfg)
        return U[0]

    # ∂U/∂I  at fixed r
    @jax.jit
    def dU_dI(r, I_wires):
        return jax.grad(lambda I_wires: trap_U(r, I_wires))(I_wires)

    # ∂²U/∂r²  at fixed I
    @jax.jit
    def H_r(r, I_wires):
        return jax.hessian(lambda r: trap_U(r, I_wires))(r)

    # ∂²U/(∂r ∂I)  at fixed r
    @jax.jit
    def g_rI(r, I_wires):
        return jax.jacobian(lambda I_wires: jax.grad(lambda r: trap_U(r, I_wires))(r))(I_wires)

    # ----------------------------------------------------------------

    # initial trap minimum (uses SciPy under the hood)
    I_phys = distribute_currents_to_wires(I_vec)
    r0 = find_trap_minimum(atom, wire_cfg, bias_cfg, np.array(I_phys), np.array([0.0, 0.0, 0.5]))

    for i in range(1, iters + 1):
        # 1) metrics at (r0, I_vec)
        U0 = float(trap_U(r0, I_vec))

        # 2) direct gradient ∂U0/∂I
        gradU = dU_dI(r0, I_vec)  # (N,)

        # 3) implicit dr/dI
        H = H_r(r0, I_vec)  # (3,3)
        J = g_rI(r0, I_vec)  # (3,N)
        drdI = -jnp.linalg.solve(H, J)  # (3,N)

        # 4) position‐penalty gradient
        x, y, z = r0
        dx_dI, dy_dI, dz_dI = drdI[0], drdI[1], drdI[2]

        grad_pos = 2 * w_x * x * dx_dI + 2 * w_y * y * dy_dI - 2 * w_z * jnp.maximum(z_min - z, 0.0) * dz_dI  # (N,)

        # 5) assemble total gradient of L
        rel_err = (U0 - target_U) / target_U
        grad_U_term = 2 * w_U * rel_err / target_U * gradU
        grad_total = grad_U_term + grad_pos  # (N,)

        # 6) gradient step + clipping
        I_vec = I_vec - lr * grad_total
        I_vec = jnp.clip(I_vec, -I_limits, I_limits)

        # 7) re-find the trap minimum under new currents
        I_phys = distribute_currents_to_wires(I_vec)
        r0 = find_trap_minimum(atom, wire_cfg, bias_cfg, np.array(I_phys), np.array(r0))

        if i % 10 == 0:
            # format current values
            shifting_currents = np.array2string(I_vec[:n_shift], precision=2, separator=", ", suppress_small=True)
            guiding_currents = np.array2string(I_vec[n_shift:], precision=2, separator=", ", suppress_small=True)
            print(
                f"[{i:3d}] U0={U0:9.3e}  "
                f"r0={r0[0]:6.2f} {r0[1]:6.2f} {r0[2]:6.2f}  "
                f"rel_err={rel_err:6.2e}  "
                f"|gradU|={jnp.linalg.norm(gradU):6.2e}  "
                f"|gradU_term|={jnp.linalg.norm(grad_U_term):6.2e}  "
                f"|step|={(lr * jnp.linalg.norm(grad_U_term)):6.2e}  "
                f"shifting={shifting_currents}  "
                f"guiding={guiding_currents}"
            )

    # split back into shifting / guiding
    I_opt_shift = I_vec[:n_shift]
    I_opt_guide = I_vec[n_shift:]
    return I_opt_shift, I_opt_guide


# ----------------------------------------------------------------------------------------------------
# Optimize the transport of the atom chip trap.
# ----------------------------------------------------------------------------------------------------
def optimize_transport(params: attrdict):
    # 1. Simulation parameters
    # select which wires to optimize
    wire_ids = jnp.array(params.wire_ids, dtype=jnp.int32)
    mask = jnp.zeros((15,)).at[wire_ids].set(1.0)
    print(f"Transport optimization settings: {params}, mask={mask}")

    # 1st and 2nd derivatives are 0 at s = 0 and s = 1
    def smoothstep_quintic(s: float) -> jnp.ndarray:
        return 10 * s**3 - 15 * s**4 + 6 * s**5  # C² at both ends

    # Cosine schedule (range [0, 1] over finite time steps t over duration T)
    def cosine_schedule(s: float) -> jnp.ndarray:
        return 0.5 * (1 - jnp.cos(jnp.pi * s))

    schedule_func = {
        "smoothstep": smoothstep_quintic,
        "cosine": cosine_schedule,
    }[params.scheduler]

    # 2. Transport initial setup
    wire_config = setup_wire_config()
    bias_config = transport_initializer.setup_bias_config()

    I_shifting_wires = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])
    I_limits = jnp.concatenate(
        [
            jnp.ones_like(I_shifting_wires) * params.I_max_shifting,  # Shifting wires limits
            jnp.ones_like(I_guiding_wires) * params.I_max_guiding,  # Guiding wires limits
        ]
    )  # shape: (15,)

    # 3. Reference values for the trap
    atom_chip = transport_initializer.build_atom_chip()
    analysis = transport_initializer.analyze_atom_chip(atom_chip)
    r0_ref = analysis.potential.minimum.position
    U0_ref = analysis.potential.minimum.value
    omega_ref = analysis.potential.trap.frequency
    BEC_radii_ref = analysis.potential.bec.radii  # non-interacting BEC radii
    TF_radii_ref = analysis.potential.tf.radii  # Thomas-Fermi radii
    mu_ref = analysis.potential.tf.mu  # Chemical potential
    destination_r = calculate_destination(r0_ref, params.num_shifts, shifting_wire_distance=0.4)  # mm

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}")
    print(f"Trap frequency: {omega_ref} (Hz) radii: {BEC_radii_ref} (m) TF: {TF_radii_ref} (m) mu: {mu_ref:.4g} (J)")
    print(f"Desired final position: {destination_r}")

    # 4. Optimize the transport
    results, error_log = generate_schedule(
        atom=atom_chip.atom,
        wire_config=wire_config,
        bias_config=bias_config,
        I_start=I_start,
        I_limits=I_limits,
        mask=mask,
        r0_ref=r0_ref,
        U0_ref=U0_ref,
        omega_ref=omega_ref,
        BEC_radii_ref=BEC_radii_ref,
        TF_radii_ref=TF_radii_ref,
        mu_ref=mu_ref,
        destination_r=destination_r,
        schedule_func=schedule_func,
        T=params.T,
        reg=params.reg,
        n_atoms=params.n_atoms,
    )
    report_results(params, results, error_log)


# ----------------------------------------------------------------------------------------------------
# Generate the transport schedule.
# ----------------------------------------------------------------------------------------------------
# fmt: off
def generate_schedule(
    atom           : ac.Atom,
    wire_config    : ac.atom_chip.WireConfig,
    bias_config    : ac.field.BiasConfig,
    I_start        : jnp.ndarray,  # Initial currents for the wires (shape: (15,))
    I_limits       : jnp.ndarray,  # Optional limits for the currents (shape: (15,))
    mask           : jnp.ndarray,  # Mask to restrict current updates (shape: (15,))
    r0_ref         : jnp.ndarray,  # Reference position of the trap at t=0 (shape: (3,))
    U0_ref         : float,        # Reference potential energy of the trap
    omega_ref      : jnp.ndarray,  # Reference trap frequencies (shape: (3,))
    BEC_radii_ref  : jnp.ndarray,  # Reference BEC radii (shape: (3,))
    TF_radii_ref   : jnp.ndarray,  # Reference Thomas-Fermi radii (shape: (3,))
    mu_ref         : float,        # Reference chemical potential
    destination_r  : jnp.ndarray,  # Desired final position of the trap (shape: (3,))
    schedule_func  : Callable[[float], jnp.ndarray],  # Scheduler function
    T              : int,          # Number of time steps
    reg            : float,        # Regularization parameter
    n_atoms        : int,          # Number of atoms in the BEC (for chemical potential calculation)
) -> Tuple[attrdict, List]:
# fmt: on
    @jax.jit
    def r_target(t: int, T: int) -> jnp.ndarray:
        """ Calculate the target position of the trap at time t """
        s = t / T  # Normalize time step to [0, 1]
        return r0_ref + schedule_func(s) * (destination_r - r0_ref)

    @jax.jit
    def trap_U(r: jnp.ndarray, I_wires: jnp.ndarray) -> float:
        """ Compute the potential energy at position r for given wire currents I_wires """
        wire_currents = distribute_currents_to_wires(I_wires)
        U, _, _ = ac.atom_chip.trap_potential_energies(
            jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config
        )
        return U[0]

    # Trap gradient
    # fmt: off
    grad_U_r  = jax.grad(trap_U, argnums=0)      # Gradient of potential energy w.r.t. position r
    hess_U_r  = jax.jacfwd(grad_U_r, argnums=0)  # Hessian of potential energy w.r.t. position r
    cross_jac = jax.jacfwd(grad_U_r, argnums=1)  # Cross Jacobian of potential energy w.r.t. wire currents I_wires
    # fmt: on

    @jax.jit
    def calc_dr_dI(r0, I_wires):
        """ Implicit gradient dr/dI = -H^{-1} @ dgrad_U_r/dI """
        H = hess_U_r(r0, I_wires)
        J = cross_jac(r0, I_wires)
        return -jnp.linalg.solve(H, J)

    @jax.jit
    def solve_for_delta_I_from_delta_r(I_wires: jnp.ndarray, r_now: jnp.ndarray, r_next: jnp.ndarray) -> jnp.ndarray:
        """
        Solve for the change in wire currents (delta_I) given the current (r_now) and target (r_next) positions.
        """
        delta_r = r_next - r_now
        # make sure non-negative displacement in x but allow negative in y and z
        delta_r = jnp.array([jnp.maximum(delta_r[0], 0.0), delta_r[1], delta_r[2]])
        dr_dI = calc_dr_dI(r_now, I_wires)  # shape: (3, 15)  Jacobian of trap position w.r.t. wire currents
        condition_number = jnp.linalg.cond(dr_dI)  # max singular value / min singular value
        alpha = reg * (1 + condition_number)  # Regularization term
        delta_I = jnp.linalg.solve(dr_dI.T @ dr_dI + alpha * jnp.eye(dr_dI.shape[1]), dr_dI.T @ delta_r)
        return delta_I

    # fmt: off
    I_wires     = I_start
    trajectory  = [ r0_ref    ]
    target_rs   = [ r0_ref    ]
    current_log = [ I_wires   ]
    U0s         = [ U0_ref    ]
    omegas      = [ omega_ref ]
    BEC_radii   = [ BEC_radii_ref ]   # Non-interacting BEC radii
    TF_radii    = [ TF_radii_ref  ]  # Thomas-Fermi radii
    mu_vals     = [ mu_ref    ]  # Chemical potential
    error_log   = []
    # fmt: on

    # Control loop
    for t in range(T):
        # Compute the current target position
        r_now = trajectory[-1]
        r_next = r_target(t + 1, T)

        # Compute the implicit gradient and update currents
        delta_I = solve_for_delta_I_from_delta_r(I_wires, r_now, r_next)
        I_wires = I_wires + delta_I * mask  # Apply mask to restrict current updates
        I_wires = jnp.clip(I_wires, -I_limits, I_limits)

        # Find the minimum trap position and evaluate the trap
        wire_currents = distribute_currents_to_wires(I_wires)
        r_min = find_trap_minimum(atom, wire_config, bias_config, wire_currents, r_now)

        # Evaluate the trap
        U0, omega, eigenvalues, bec_radii, tf_radii, mu = evaluate_trap(
            atom, wire_config, bias_config, wire_currents, r_min, n_atoms)

        # Check for NaN or negative eigenvalues in omega
        message = " ".join([
            f"Step {t + 1:4d}:",
            f"r_min={format_array(r_min)}",
            f"U0={U0:10.4g}",
            f"omega={format_array(omega)}",
            f"BEC-radii={format_array(bec_radii)}",
            f"TF-radii={format_array(tf_radii)}",
            f"mu={mu:10.4g}",
        ])
        print(message)

        if jnp.any(jnp.isnan(omega)) or jnp.any(eigenvalues < 0):
            error_log.append(f"{message}: NaN or negative eigenvalues!!!")
            continue

        # Log the results
        trajectory.append(r_min)
        target_rs.append(r_next)
        current_log.append(I_wires)
        U0s.append(U0)
        omegas.append(omega)
        BEC_radii.append(bec_radii)
        TF_radii.append(tf_radii)
        mu_vals.append(mu)

    if error_log:
        print(f"{len(error_log)} errors encountered during optimization!")
    else:
        print("Optimization completed successfully without errors.")

    # fmt: off
    results = attrdict(
        trajectory  = jnp.stack(trajectory),
        target_rs   = jnp.stack(target_rs),
        current_log = jnp.stack(current_log),
        U0s         = jnp.array(U0s),
        omegas      = jnp.stack(omegas),
        BEC_radii   = jnp.stack(BEC_radii),
        TF_radii    = jnp.stack(TF_radii),
        mu_vals     = jnp.array(mu_vals),
    )
    # fmt: on
    return results, error_log


def format_array(array: jnp.ndarray) -> str:
    return np.array2string(
        np.array(array),
        formatter={"float_kind": lambda x: f"{x: 10.4g}"},
        separator=" ",
    )


# ----------------------------------------------------------------------------------------------------
# Evaluate the potential energy and curvature at a fixed position.
#
# This function is JIT-compiled for performance.
# ----------------------------------------------------------------------------------------------------
@jax.jit
def evaluate_trap(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasConfig,
    wire_currents: jnp.ndarray,
    trap_position: jnp.ndarray,
    n_atoms: int,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Evaluate potential energy and curvature at a fixed position."""

    def objective(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    U0 = objective(trap_position)
    # H = ac.potential.hessian_by_finite_difference(
    #     objective,
    #     jnp.ravel(trap_position),
    #     step=1e-3,
    # )
    # eigenvalues = H.eigenvalues * 1e6  # # J/mm^2 -> J/m^2
    # omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)
    H = jax.hessian(objective)(trap_position.ravel())  # .ravel() ensures it's 1D
    eigenvalues = jnp.linalg.eigvalsh(H) * 1e6  # Use eigvalsh for symmetric matrix, more stable
    omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)

    # Non-interacting BEC radii
    angular_freq = 2 * jnp.pi * omega
    bec_radii = jnp.sqrt(ac.constants.hbar / (atom.mass * angular_freq))

    # Thomas-Fermi radii
    w_ho = jnp.prod(angular_freq) ** (1 / 3)  # geometric mean
    a_ho = jnp.sqrt(ac.constants.hbar / (atom.mass * w_ho))
    mu = 0.5 * ac.constants.hbar * w_ho * (15 * atom.a_s * n_atoms / a_ho) ** (2 / 5)
    tf_radii = jnp.sqrt(2 * mu / atom.mass) / angular_freq

    return U0, omega, eigenvalues, bec_radii, tf_radii, mu


# ----------------------------------------------------------------------------------------------------
# Distribute logical wire currents to physical wire segments.
# ----------------------------------------------------------------------------------------------------
@jax.jit
def distribute_currents_to_wires(I_wires: jnp.ndarray) -> jnp.ndarray:
    """
    Distribute logical wire currents to physical wire segments.
    Input shape: (15,) = 6 shifting + 9 guiding
    Output shape: (73,) = expanded full wire layout
    """
    I_shifting_wires = I_wires[:6]  # shape: (6,)
    I_guiding_wires = I_wires[6:]  # shape: (9,)

    # Expand shifting wires
    total_shifting = jnp.concatenate(
        [I_shifting_wires, -I_shifting_wires, I_shifting_wires, -I_shifting_wires, I_shifting_wires]
    )  # (30,)

    # Expand guiding wires
    total_guiding = jnp.repeat(I_guiding_wires, GUIDING_WIRE_SEGMENT_COUNTS)  # (43,)

    return jnp.concatenate([total_shifting, total_guiding])  # (73,)


# ----------------------------------------------------------------------------------------------------
# Main entry point for the script.
# ----------------------------------------------------------------------------------------------------
def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Transport Optimizer for Atom Chip")
    parser.add_argument("--init", action="store_true", help="Optimize initial transport currents")
    parser.add_argument("--T",              type=int,   default=50000,    help="Number of time steps")
    parser.add_argument("--num_shifts",     type=int,   default=6,        help="Number of shifts to apply to the trap")
    parser.add_argument("--reg",            type=float, default=1e-2,     help="Regularization parameter")
    parser.add_argument("--n_atoms",        type=int,   default=int(1e5), help="Number of atoms in the BEC")
    parser.add_argument("--I_max_shifting", type=float, default=3.5,      help="Max current for shifting wires (A)")
    parser.add_argument("--I_max_guiding",  type=float, default=70.0,     help="Max current for guiding wires (A)")
    parser.add_argument("--wire_ids",       type=str, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 14],
                        help="List of wire IDs to optimize (default: all shifting wires and outermost guiding wires)")
    parser.add_argument("--scheduler",      type=str, default="smoothstep",
                        choices=["smoothstep", "cosine"],
                        help="Scheduler function to normalize time")
    parser.add_argument("--transport_time", type=float, default=3.0, help="Transport time in seconds")
    args = parser.parse_args()
    # fmt: on

    if args.init:
        # Initialize the transport optimizer with default parameters
        I_shifting_wires, I_guiding_wires = optimize_initial_currents(attrdict(**vars(args)))
        # format the results into string with 2 decimal places
        I_shifting_wires = np.array2string(I_shifting_wires, precision=2, separator=', ', suppress_small=True)
        I_guiding_wires = np.array2string(I_guiding_wires, precision=2, separator=', ', suppress_small=True)
        print(f"Optimized shifting wire currents: {I_shifting_wires}")
        print(f"Optimized guiding wire currents: {I_guiding_wires}")
        print("Update the transport_initializer.py file with these values.")
    else:
        # Run the transport optimization
        optimize_transport(attrdict(**vars(args)))


if __name__ == "__main__":
    main()

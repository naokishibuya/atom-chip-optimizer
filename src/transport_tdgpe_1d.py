"""
1-D TDGPE transport
"""

import argparse
import numpy as np
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import atom_chip as ac
from transport_initializer import ATOM as atom
from transport_reporter import load_results


# ----------------------------------------------------------------------------------------------------
# Harmonic oscillator ground state
# ----------------------------------------------------------------------------------------------------
def ho_ground_state(x_um, centre_um, m, omega):
    """1-D harmonic-oscillator ground state on µm grid."""
    hbar = ac.constants.hbar  # J·s
    x_m = (x_um - centre_um) * 1e-6
    a_ho = jnp.sqrt(hbar / (m * omega))
    psi = jnp.exp(-(x_m**2) / (2 * a_ho**2)) / (jnp.pi**0.25 * jnp.sqrt(a_ho))
    return normalize(psi, x_um)


# ----------------------------------------------------------------------------------------------------
# Normalize wavefunction
# ----------------------------------------------------------------------------------------------------
def normalize(psi, x_um):
    norm = np.sqrt(np.trapezoid(jnp.abs(psi) ** 2, x_um))
    return psi / norm if norm > 0 else psi


# ----------------------------------------------------------------------------------------------------
# Harmonic oscillator energy
# ----------------------------------------------------------------------------------------------------
def ho_energy(psi, x_um, m, omega):
    hbar = ac.constants.hbar  # J·s
    dx_m = (x_um[1] - x_um[0]) * 1e-6
    x_m = x_um * 1e-6
    grad = (jnp.roll(psi, -1) - jnp.roll(psi, 1)) / (2 * dx_m)
    T = (hbar**2 / (2 * m)) * jnp.abs(grad) ** 2
    V = 0.5 * m * omega**2 * x_m**2 * jnp.abs(psi) ** 2
    return float(np.trapezoid(T + V, x_m))


# ----------------------------------------------------------------------------------------------------
# Time-dependent split-step
# ----------------------------------------------------------------------------------------------------
@jax.jit
def split_step_td_scan(psi0_cplx, V_ts, g1d, dt, dx_m, m):
    """Time-dependent split-step.  V_ts shape = (T, N)."""
    hbar = ac.constants.hbar  # J·s
    k = 2 * jnp.pi * jnp.fft.fftfreq(V_ts.shape[1], d=dx_m)

    def body(psi, Vt):
        nlin = g1d * jnp.abs(psi) ** 2
        psi_h = psi * jnp.exp(-0.5j * dt / hbar * (Vt + nlin))
        psi_k = jnp.fft.fft(psi_h) * jnp.exp(-0.5j * dt * hbar * k**2 / m)
        psi_f = jnp.fft.ifft(psi_k)
        nlin = g1d * jnp.abs(psi_f) ** 2
        psi_next = psi_f * jnp.exp(-0.5j * dt / hbar * (Vt + nlin))
        return psi_next, None

    psiT, _ = jax.lax.scan(body, psi0_cplx, V_ts[:-1])
    return psiT


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="1-D TDGPE transport simulation using JAX.")
    parser.add_argument("--results_dir", type=str, help="Path to the results directory")
    parser.add_argument("--upsample", type=int, default=1, help="Upsampling multiplier")
    parser.add_argument("--transport_time", type=float, default=3, help="transport seconds")
    args = parser.parse_args()

    # optimiser output
    params, results = load_results(args.results_dir)
    traj = results.trajectory  # mm
    omegas = results.omegas  # Hz
    n_atoms = float(params.n_atoms)

    # physical constants
    hbar = ac.constants.hbar  # J·s
    m_rb = atom.mass  # kg
    a_s = atom.a_s  # s-wave scattering length (m)

    # initial & final trap parameters
    r0_um, rT_um = traj[[0, -1], 0] * 1e3
    w0, wT = omegas[[0, -1], 0] * 2 * np.pi
    wyT, wzT = omegas[-1, 1:] * 2 * np.pi

    # spatial grid
    x = jnp.linspace(min(r0_um, rT_um) - 10, max(r0_um, rT_um) + 10, 4096)
    dx_m = float((x[1] - x[0]) * 1e-6)

    # states
    psi0 = ho_ground_state(x, r0_um, m_rb, w0).astype(jnp.complex128)
    phi0 = ho_ground_state(x, rT_um, m_rb, wT)

    # interaction strength g1d
    omega_perp = np.sqrt(wyT * wzT)
    a_perp = np.sqrt(hbar / (m_rb * omega_perp))
    g1d = 2 * hbar**2 * a_s * n_atoms / (m_rb * a_perp**2)

    t_idx = np.arange(len(traj))
    t_fine = np.linspace(0, len(traj) - 1, args.upsample * len(traj))

    s = t_idx / t_idx[-1]
    s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5  # C² smooth-step
    r_smooth = r0_um + s_smooth * (rT_um - r0_um)
    w_smooth = w0 + s_smooth * (wT - w0)

    r_path = np.interp(t_fine, t_idx, r_smooth)  # µm
    w_path = np.interp(t_fine, t_idx, w_smooth)  # rad/s

    V_ts = 0.5 * m_rb * (w_path[:, None] ** 2) * ((x[None, :] - r_path[:, None]) * 1e-6) ** 2
    dt = args.transport_time / (V_ts.shape[0] - 1)
    print(f"dt = {dt:.2e} s  |  steps = {V_ts.shape[0]}")

    # time evolution
    psiT = split_step_td_scan(psi0, V_ts, g1d, dt, dx_m, m_rb)
    psiT = normalize(psiT, x)

    # diagnostics
    fidelity = float(np.abs(np.trapezoid(jnp.conj(phi0) * psiT, x)) ** 2)
    E_exc = ho_energy(psiT, x, m_rb, wT) - 0.5 * hbar * wT
    print(f"Fidelity F = {fidelity:.6e}\nExcitation ΔE = {E_exc:.3e} J")

    # plot results
    # plot_density(x, psi0, phi0, psiT, zoom=False)  # full view
    # plot_density(x, psi0, phi0, psiT, zoom=True)  # zoomed on final state
    plot_1d_tdgpe(x, phi0, psiT, args)


def plot_density(x, psi0, phi0, psiT, zoom=False):
    plt.figure(figsize=(8, 5))
    if not zoom:
        plt.plot(x, jnp.abs(psi0) ** 2, label=r"$|\psi_0|^2$", alpha=0.4)
    plt.plot(x, jnp.abs(phi0) ** 2, "--", label=r"$|\phi_0|^2$ target")
    plt.plot(x, jnp.abs(psiT) ** 2, label=r"$|\psi_T|^2$")
    plt.xlabel("x (µm)")
    plt.ylabel("Probability density")
    plt.title("TDGPE transport — fully-JAX scan (smooth-step path)")
    if zoom:
        # where does ψT live?  (threshold = 1e-3 of max density)
        th = 1e-3 * jnp.max(jnp.abs(psiT) ** 2)
        support_mask = jnp.abs(psiT) ** 2 > th
        x_min = float(x[support_mask][0])
        x_max = float(x[support_mask][-1])
        pad = 2.0  # µm margin
        plt.xlim(x_min - pad, x_max + pad)
        plt.title("TDGPE transport (zoom on final trap)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    tag = "zoom" if zoom else "full"
    plt.savefig(f"tdgpe_probabilities_{tag}.png", dpi=150)
    plt.show()


def plot_1d_tdgpe(x, phi0, psiT, args):
    # find the first and last non-zero points
    first_nonzero = np.argmax(jnp.abs(psiT) > 0.01)
    last_nonzero = len(psiT) - np.argmax(jnp.abs(psiT[::-1]) > 0) - 1
    x = x[first_nonzero : last_nonzero + 1]
    psiT = psiT[first_nonzero : last_nonzero + 1]
    phi0 = phi0[first_nonzero : last_nonzero + 1]

    print(f"First non-zero at x = {x[first_nonzero]:.2f} µm")
    print(f"Last non-zero at x = {x[last_nonzero]:.2f} µm")
    print(f"Range of non-zero values: {x[first_nonzero]:.2f} µm to {x[last_nonzero]:.2f} µm")

    # plot
    plt.figure(figsize=(8, 5))
    # plt.plot(x, jnp.abs(psi0) ** 2, label="|ψ₀|²")
    plt.plot(x, jnp.abs(phi0) ** 2, label="|φ₀|² target", linestyle="--")
    plt.plot(x, jnp.abs(psiT) ** 2, label="|ψ_T|²", alpha=0.7)
    plt.xlabel("x (µm)")
    plt.ylabel("Probability density")
    plt.title("1D TD-GPE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(args.results_dir, "tdgpe-1d.png")
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

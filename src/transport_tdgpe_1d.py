"""
1-D TDGPE transport — fast JAX scan
===================================
Fully-JIT, time-dependent split-step simulation of a BEC in a moving harmonic trap.
Changes in this version
• indentation fixed (block after g1d)
• duplicate `V_ts = ...` removed
• smooth-step up-sampling retained (UPSAMPLE = 20, TRANSPORT_TIME = 10 s)
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from transport_optimizer import load_results

# ————————————————————————————————————————————————
# Helper functions
# ————————————————————————————————————————————————


def normalize(psi, x_um):
    norm = np.sqrt(np.trapezoid(jnp.abs(psi) ** 2, x_um))
    return psi / norm if norm > 0 else psi


def ho_ground_state(x_um, centre_um, m, omega):
    """1-D harmonic-oscillator ground state on µm grid."""
    hbar = 1.054_571_817e-34
    x_m = (x_um - centre_um) * 1e-6
    a_ho = jnp.sqrt(hbar / (m * omega))
    psi = jnp.exp(-(x_m**2) / (2 * a_ho**2)) / (jnp.pi**0.25 * jnp.sqrt(a_ho))
    return normalize(psi, x_um)


# ——— fully-JAX split-step with lax.scan ———
@jax.jit
def split_step_td_scan(psi0_cplx, V_ts, g1d, dt, dx_m, m):
    """Time-dependent split-step.  V_ts shape = (T, N)."""
    hbar = 1.054_571_817e-34
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


def ho_energy(psi, x_um, m, omega):
    hbar = 1.054_571_817e-34
    dx_m = float((x_um[1] - x_um[0]) * 1e-6)
    x_m = x_um * 1e-6
    grad = (jnp.roll(psi, -1) - jnp.roll(psi, 1)) / (2 * dx_m)
    T = (hbar**2 / (2 * m)) * jnp.abs(grad) ** 2
    V = 0.5 * m * omega**2 * x_m**2 * jnp.abs(psi) ** 2
    return float(np.trapezoid(T + V, x_m))


# ————————————————————————————————————————————————
# Main
# ————————————————————————————————————————————————


def main():
    # optimiser output
    data = load_results("transport_results.json")
    traj = data["trajectory"]  # mm
    omegas = data["omegas"]  # Hz
    n_atoms = float(data["n_atoms"])

    # physical constants
    m_rb = 1.443_160_60e-25
    hbar = 1.054_571_817e-34
    a_s = 5.3e-9

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

    # —— up-sample optimiser path with quintic smooth-step ——
    UPSAMPLE = 1
    TRANSPORT_TIME = 3.0  # s total

    t_idx = np.arange(len(traj))
    t_fine = np.linspace(0, len(traj) - 1, UPSAMPLE * len(traj))

    s = t_idx / t_idx[-1]
    s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5  # C² smooth-step
    r_smooth = r0_um + s_smooth * (rT_um - r0_um)
    w_smooth = w0 + s_smooth * (wT - w0)

    r_path = np.interp(t_fine, t_idx, r_smooth)  # µm
    w_path = np.interp(t_fine, t_idx, w_smooth)  # rad/s

    V_ts = 0.5 * m_rb * (w_path[:, None] ** 2) * ((x[None, :] - r_path[:, None]) * 1e-6) ** 2
    dt = TRANSPORT_TIME / (V_ts.shape[0] - 1)
    print(f"dt = {dt:.2e} s  |  steps = {V_ts.shape[0]}")

    # time evolution
    psiT = split_step_td_scan(psi0, V_ts, g1d, dt, dx_m, m_rb)
    psiT = normalize(psiT, x)

    # diagnostics
    fidelity = float(np.abs(np.trapezoid(jnp.conj(phi0) * psiT, x)) ** 2)
    E_exc = ho_energy(psiT, x, m_rb, wT) - 0.5 * hbar * wT
    print(f"Fidelity F = {fidelity:.6e}\nExcitation ΔE = {E_exc:.3e} J")

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, jnp.abs(psi0) ** 2, label="|ψ₀|²")
    plt.plot(x, jnp.abs(psiT) ** 2, label="|ψ_T|²")
    plt.plot(x, jnp.abs(phi0) ** 2, label="|φ₀|² target")
    plt.xlabel("x (µm)")
    plt.ylabel("Probability density")
    plt.title("TDGPE transport — fully-JAX scan (smooth-step path)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tdgpe_probabilities.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

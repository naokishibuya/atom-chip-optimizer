from typing import List, Tuple
import matplotlib.pyplot as plt
from atom_chip.atom_chip import AtomChip


def show_summary(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    font_family: str = "monospace",
    font_size: int = 9,
    box_style: str = "round,pad=0.5",
    face_color: str = "whitesmoke",
    edge_color: str = "gray",
    vertical_alignment: str = "top",
    horizontal_alignment: str = "left",
    fig: plt.Figure = None,
) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()

    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        x=0.01,
        y=0.99,
        s=format_summary(atom_chip),
        fontsize=font_size,
        fontfamily=font_family,
        verticalalignment=vertical_alignment,
        horizontalalignment=horizontal_alignment,
        bbox=dict(
            boxstyle=box_style,
            facecolor=face_color,
            edgecolor=edge_color,
        ),
    )
    fig.tight_layout()
    return fig


def format_summary(atom_chip: AtomChip) -> str:
    # Bias field parameters
    # fmt: off
    bias       = atom_chip.bias_fields
    field      = atom_chip.field
    trap       = atom_chip.trap
    bias_found = bias is not None
    fmin_found = field.minimum.found
    tmin_found = trap.minimum.found
    # fmt: on
    return f"""
Trap Analysis

Bias Field Parameters
-----------------------------------------------------------------
Coil factors                [G/A] : {format_array(bias_found, bias.coil_factors, precision=1)}
Coil currents                 [A] : {format_array(bias_found, bias.currents, precision=1)}
Stray fields                  [G] : {format_array(bias_found, bias.stray_fields, precision=1)}

Magnetic Field Minimum
-----------------------------------------------------------------
Field Minimum                 [G] : {format_value(fmin_found, field.minimum.value)}
Minimum Location             [mm] : {format_array(fmin_found, field.minimum.position)}
Larmor frequency            [MHz] : {format_value(fmin_found, field.larmor.frequency * 1e-6)}
Trap frequencies             [Hz] : {format_array(fmin_found, field.trap.frequency, precision=1)}

Hessian Eigenvalues and Eigenvectors:
{format_array(fmin_found, field.hessian.eigenvalues, pad=" ")}
{format_matrix(fmin_found, field.hessian.eigenvectors)}

Trap Potential Minimum
-----------------------------------------------------------------
Potential Minimum             [J] : {format_value(tmin_found, trap.minimum.value)}
Minimum Location             [mm] : {format_array(tmin_found, trap.minimum.position)}
Larmor frequency            [MHz] : {format_value(tmin_found, trap.larmor.frequency * 1e-6)}
Trap frequencies             [Hz] : {format_array(tmin_found, trap.trap.frequency, precision=1)}

Hessian Eigenvalues and Eigenvectors:
{format_array(tmin_found, trap.hessian.eigenvalues, pad=" ")}
{format_matrix(tmin_found, trap.hessian.eigenvectors)}

BEC Parameters (Harmonic Oscillator Approximation)
-----------------------------------------------------------------
HO Length a_ho               [μm] : {format_value(tmin_found, trap.bec.a_ho * 1e6)}
Trap Frequency G-Avg w_ho [rad/s] : {format_value(tmin_found, trap.bec.w_ho)}

Non-interacting           [atoms] : {format_count(tmin_found, trap.bec.total_atoms)}
Chemical Potential μ0         [J] : {format_value(tmin_found, trap.bec.mu_0)}
Harmonic Oscillator Radii    [μm] : {format_array(tmin_found, trap.bec.radii * 1e6)}
Critical Temperature         [nK] : {format_value(tmin_found, trap.bec.T_c * 1e9)}

Thomas-Fermi              [atoms] : {format_count(tmin_found, trap.tf.condensed_atoms)}
Chemical Potenential μ        [J] : {format_value(tmin_found, trap.tf.mu)}
Harmonic Oscillator Radii    [μm] : {format_array(tmin_found, trap.tf.radii * 1e6)}
"""


def format_matrix(found: bool, eigvecs: List[float], precision: int = 3) -> str:
    if not found or len(eigvecs) == 0:
        return "N/A"
    try:
        return "\n".join("|" + " ".join(f"{float(v): 2.{precision}e}" for v in row) + " |" for row in eigvecs)
    except Exception:
        return "N/A"


def format_array(found: bool, arr: List[float], precision: int = 3, pad: str = " ") -> str:
    if not found or len(arr) == 0:
        return "N/A"
    try:
        return "[" + " ".join(f"{float(v):{pad}2.{precision}e}" for v in arr) + f"{pad}]"
    except Exception:
        return "N/A"


def format_value(found: bool, val: float, precision: int = 3) -> str:
    if not found:
        return "N/A"
    try:
        return f"{float(val):.{precision}e}"
    except Exception:
        return "N/A"


def format_count(found: bool, count: int) -> str:
    if not found:
        return "N/A"
    try:
        return f"{int(count)}"
    except Exception:
        return "N/A"

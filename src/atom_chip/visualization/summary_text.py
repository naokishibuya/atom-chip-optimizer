from typing import Tuple
import platform
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QFont
from atom_chip.atom_chip import AtomChip


def get_monospaced_font() -> str:
    system = platform.system()
    if system == "Windows":
        font_family = "Courier New"
    elif system == "Darwin":
        font_family = "Menlo"
    else:
        font_family = "DejaVu Sans Mono"
    return font_family


def show_summary(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    font_family: str = get_monospaced_font(),
    font_size: int = 10,
    fig: plt.Figure = None,
) -> plt.Figure:
    if fig is None:
        # Create dummy figure and canvas
        fig, ax = plt.subplots(figsize=size)
        ax.set_visible(False)  # Hide plot area

        # Set background and border
        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setFont(QFont(font_family, font_size))
        text_box.setStyleSheet("""
            background-color: whitesmoke;
            border: 1px solid gray;
            border-radius: 5px;
            margin: 1px;
            padding: 3px 15px;
            color: black;
        """)

        # Create a central widget for the text box
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            background-color: #e5e5e5;
            margin: 0;
        """)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)
        layout.addWidget(text_box)
        central_widget.setLayout(layout)

        # update window
        fig.canvas.manager.window.setCentralWidget(central_widget)
        fig.canvas.toolbar.setVisible(False)  # Hide toolbar

        # Store reference to text box
        fig.text_box = text_box

    fig.text_box.setPlainText(format_summary(atom_chip))
    fig.tight_layout()
    return fig


def format_summary(atom_chip: AtomChip) -> str:
    # Bias field parameters
    # fmt: off
    bias       = atom_chip.bias_fields
    field      = atom_chip.field
    trap       = atom_chip.potential
    bias_found = bias is not None
    fmin_found = field.minimum.found
    tmin_found = trap.minimum.found
    # fmt: on
    return f"""
Trap Analysis

Bias Field Parameters
----------------------------------------------------------------------------
Coil factors                [G/A] : {format_array(bias_found, bias.coil_factors)}
Coil currents                 [A] : {format_array(bias_found, bias.currents)}
Stray fields                  [G] : {format_array(bias_found, bias.stray_fields)}

Magnetic Field Minimum
----------------------------------------------------------------------------
Field Minimum                 [G] : {format_value(fmin_found, field.minimum.value)}
Minimum Location             [mm] : {format_array(fmin_found, field.minimum.position)}
Larmor frequency            [MHz] : {format_value(fmin_found, field.larmor.frequency * 1e-6)}
Trap frequencies             [Hz] : {format_array(fmin_found, field.trap.frequency)}

Hessian Eigenvalues and Eigenvectors:
{format_array(fmin_found, field.hessian.eigenvalues)}
{format_matrix(fmin_found, field.hessian.eigenvectors)}

Trap Potential Minimum
----------------------------------------------------------------------------
Potential Minimum             [J] : {format_value(tmin_found, trap.minimum.value)}
Minimum Location             [mm] : {format_array(tmin_found, trap.minimum.position)}
Larmor frequency            [MHz] : {format_value(tmin_found, trap.larmor.frequency * 1e-6)}
Trap frequencies             [Hz] : {format_array(tmin_found, trap.trap.frequency)}

Hessian Eigenvalues and Eigenvectors:
{format_array(tmin_found, trap.hessian.eigenvalues)}
{format_matrix(tmin_found, trap.hessian.eigenvectors)}

BEC Parameters (Harmonic Oscillator Approximation)
----------------------------------------------------------------------------
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


def format_matrix(
    found: bool,
    matrix: jnp.ndarray,
    precision: int = 4,
) -> str:
    if not found or len(matrix) == 0:
        return "N/A"
    return "\n".join(
        format_array(
            found,
            matrix[i],
            precision=precision,
        )
        .replace("[", "|")
        .replace("]", "|")
        for i in range(len(matrix))
    )


def format_array(
    found: bool,
    array: jnp.ndarray,
    precision: int = 4,
) -> str:
    if not found or len(array) == 0:
        return "N/A"
    return np.array2string(
        np.array(array),
        formatter={"float_kind": lambda x: f"{x: {precision + 6}.{precision}f}"},
        separator=" ",
    )


def format_value(found: bool, value: float, precision: int = 4) -> str:
    if not found:
        return "N/A"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def format_count(found: bool, count: int) -> str:
    if not found:
        return "N/A"
    try:
        return f"{int(count):,d}"
    except Exception:
        return "Err"

import os
import logging
import jax.numpy as jnp
import atom_chip as ac


# fmt: off
#-------------------------------------------------------------
# PCB - not used so it gives offsets only
#-------------------------------------------------------------
PCB_LAYER_TOP_HEIGHT    : float = 0.07   # PCB Top Layer tickness (2 oz copper)
PCB_LAYER_BOTTOM_HEIGHT : float = 0.07   # PCB Bottom Layer tickness (2 oz copper)
PCB_CORE_HEIGHT         : float = 0.38   # 0.42; %0.38;

#--------------------------------------------------------------
# Copper Z
#--------------------------------------------------------------
# Gap between top face of Copper Z and top face of PCB (surface after QWP fell)
COOPER_Z_GAP           : float = 0.188 

#-------------------------------------------------------------
# Sidebars
#-------------------------------------------------------------
COPPER_SIDE_BAR_WIDTH  : float = 2.0  # Copper Sidebars width
COPPER_SIDE_BAR_HEIGHT : float = 1.0  # Copper Sidebars thickness

#-------------------------------------------------------------
# Bias fields
#-------------------------------------------------------------
# Coil Current [A] to Field [G] Conversion:
BIAS_X_COIL_FACTOR  : float = -1.068  # [G/A]
BIAS_Y_COIL_FACTOR  : float =  1.8    # [G/A]
BIAS_Z_COIL_FACTOR  : float =  3.0    # [G/A]

# Coil currents [A] to be applied to the external coils
BIAS_X_COIL_CURRENT : float = 17.4
BIAS_Y_COIL_CURRENT : float = 44.3
BIAS_Z_COIL_CURRENT : float = 0.0

# PCB stray fields (G)
BIAS_X_STRAY_FIELD  : float =  3.5
BIAS_Y_STRAY_FIELD  : float = -0.1
BIAS_Z_STRAY_FIELD  : float =  0.0


def build_atom_chip(
    pcb_height       : float = PCB_LAYER_TOP_HEIGHT + PCB_CORE_HEIGHT + PCB_LAYER_BOTTOM_HEIGHT,
    copper_z_gap     : float = COOPER_Z_GAP,
    copper_z_current : float = 85.0,
    sidebar_width    : float = COPPER_SIDE_BAR_WIDTH,
    sidebar_height   : float = COPPER_SIDE_BAR_HEIGHT,
    sizebar_current  : float = 0.0,
    coil_factors     : jnp.ndarray = jnp.array([BIAS_X_COIL_FACTOR , BIAS_Y_COIL_FACTOR , BIAS_Z_COIL_FACTOR]),
    coil_currents    : jnp.ndarray = jnp.array([BIAS_X_COIL_CURRENT, BIAS_Y_COIL_CURRENT, BIAS_Z_COIL_CURRENT]),
    stray_fields     : jnp.ndarray = jnp.array([BIAS_X_STRAY_FIELD , BIAS_Y_STRAY_FIELD , BIAS_Z_STRAY_FIELD]),
) -> ac.AtomChip:
    # offsets
    copper_z_offset = -(pcb_height + copper_z_gap)
    sidebar_offset  = -(pcb_height + copper_z_gap + sidebar_height / 2)

    # Copper Z wires
    copper_z_wires = [
        # [[start point], [end point], width, height]
        [[ 0   , -40.2 , -127.5 ], [ 0   , -40.2 ,  -34.5 ], 3  , 3  ],
        [[ 0   , -37.2 ,  -31.0 ], [ 0   , -25.0 ,  -31.0 ], 5  , 7  ],
        [[ 0   , -22.5 ,  -27.5 ], [ 0   , -22.5 ,   -2.5 ], 3  , 3  ],
        [[-3.25, -17.75,   -1.25], [-3.25,  -6.5 ,   -1.25], 1.5, 2.5],
        [[-3.25,  -6.5 ,   -0.5 ], [-3.25,  -0.15,   -0.5 ], 1.5, 1  ],
        [[-3.0 ,   0.0 ,   -0.5 ], [ 3.0 ,   0.0 ,   -0.5 ], 1.5, 1  ],
        [[ 3.25,   0.15,   -0.5 ], [ 3.25,   6.5 ,   -0.5 ], 1.5, 1  ],
        [[ 3.25,   6.5 ,   -1.25], [ 3.25,  17.75,   -1.25], 1.5, 2.5],
        [[ 0   ,  22.5 ,   -2.5 ], [ 0   ,  22.5 ,  -27.5 ], 3  , 3  ],
        [[ 0   ,  25.0 ,  -31.0 ], [ 0   ,  37.2 ,  -31.0 ], 5  , 7  ],
        [[ 0   ,  40.2 ,  -34.5 ], [ 0   ,  40.2 , -127.5 ], 3  , 3  ],
    ]

    for wire in copper_z_wires:
        wire[0][2] += copper_z_offset  # start point
        wire[1][2] += copper_z_offset  # end point

    copper_z = ac.components.RectangularConductor.create(
        material = "copper",
        current  = copper_z_current,
        segments = copper_z_wires,
    )

    # Sidebars
    sidebar_wires = [
        # [[start point], [end point], width, height]
        [[-10, -17, sidebar_offset], [-10, 17, sidebar_offset], sidebar_width, sidebar_height],
        [[ 10, -17, sidebar_offset], [ 10, 17, sidebar_offset], sidebar_width, sidebar_height],
    ]

    sidebars = ac.components.RectangularConductor.create(
        material = "copper",
        current  = sizebar_current,
        segments = sidebar_wires,
    )

    # Bias fields
    bias_fields = ac.field.BiasFields(
        currents     = coil_currents, # Currents applied to external coids [A]
        coil_factors = coil_factors,  # Current to Field Conversion [G/A]
        stray_fields = stray_fields,  # Stray field offsets [G]
    )

    atom_chip = ac.AtomChip(
        name = "Copper Z",
        atom = ac.rb87,
        components = [copper_z, sidebars],
        bias_fields = bias_fields,
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
                disp    = True,
            ),
        ),
        hessian = dict(
            # method = "jax",
            method = "finite-difference",
            hessian_step = 1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
    )
    # fmt: on

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Build the atom chip
    atom_chip = build_atom_chip()

    # Analyze the atom chip
    analysis = atom_chip.analyze(options)

    # Export the atom chip to JSON
    directory = os.path.dirname(__file__)
    atom_chip.save(os.path.join(directory, "copper_z.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, analysis, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    main()

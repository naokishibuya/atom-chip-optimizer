import os
import logging
from typing import List
import jax.numpy as jnp
import atom_chip as ac


# fmt: off
#-------------------------------------------------------------
# PCB
#-------------------------------------------------------------
PCB_GLASS_HEIGHT        : float = 0.0   # Half-wave plate thickness
PCB_LAYER_TOP_HEIGHT    : float = 0.07  # PCB Top Layer tickness
PCB_CORE_HEIGHT         : float = 0.38  # PCB Core Layer tickness
PCB_LAYER_BOTTOM_HEIGHT : float = 0.07  # PCB Bottom Layer tickness
PCB_Y_OFFSET            : float = 0.0
PCB_X_OFFSET            : float = 0.0
PCB_TOP_OFFSET          : float = 0.0   # PCB top layer vertical offset from reflecting surface (after QWP fell)

#-------------------------------------------------------------
# Shifting wires on chip for Transport
#-------------------------------------------------------------
SHIFTING_WIRE_LENGTH    : float = 16.0  # Shifting wire length
SHIFTING_WIRE_WIDTH     : float = 0.3   # Shifting wire width
SHIFTING_WIRE_CURRENTS  : List[float] = [0.53, 0.89, -0.97, 0.89, 0.53, 0]  # Shifting wire current (A)

#-------------------------------------------------------------
# Guiding wires for Quadrupole fields
#-------------------------------------------------------------
GUIDING_WIRE_LENGTH     : float = 62.0  # Guiding wire length
PBC_PIN_LENGTH          : float = 50.0  # PCB pin length (leg length)
GUIDING__WIRE_CURRENT_0 : float = -3.7  # Central guiding wire current (A)
GUIDING__WIRE_CURRENT_1 : float = 13.8  # Outer guiding wire current (A)
GUIDING__WIRE_CURRENTS  : List[float] = [
                          0, # PCB Wire Q4
    GUIDING__WIRE_CURRENT_1, # PCB Wire Q3
    GUIDING__WIRE_CURRENT_1, # PCB Wire Q2
    GUIDING__WIRE_CURRENT_0, # PCB Wire Q1
    GUIDING__WIRE_CURRENT_0, # PCB Wire Q0
    GUIDING__WIRE_CURRENT_0, # PCB Wire Q1'
    GUIDING__WIRE_CURRENT_1, # PCB Wire Q2'
    GUIDING__WIRE_CURRENT_1, # PCB Wire Q3'
                          0, # PCB Wire Q4'
]

#-------------------------------------------------------------
# Bias fields
#-------------------------------------------------------------
# Coil Current [A] to Field [G] Conversion:
BIAS_X_COIL_FACTOR  : float = -1.068  # [G/A]
BIAS_Y_COIL_FACTOR  : float =  1.8    # [G/A]
BIAS_Z_COIL_FACTOR  : float =  3.0    # [G/A]

# Coil currents [A] to be applied to the external coils
BIAS_X_COIL_CURRENT : float = 0.0
BIAS_Y_COIL_CURRENT : float = 0.0
BIAS_Z_COIL_CURRENT : float = 0.0

# Stray fields (G)
BIAS_X_STRAY_FIELD  : float = 1.0
BIAS_Y_STRAY_FIELD  : float = 1.0
BIAS_Z_STRAY_FIELD  : float = 0.0

# Bias field properties
COIL_FACTORS  = jnp.array([BIAS_X_COIL_FACTOR , BIAS_Y_COIL_FACTOR , BIAS_Z_COIL_FACTOR], dtype=jnp.float64)
COIL_CURRENTS = jnp.array([BIAS_X_COIL_CURRENT, BIAS_Y_COIL_CURRENT, BIAS_Z_COIL_CURRENT], dtype=jnp.float64)
STRAY_FIELDS  = jnp.array([BIAS_X_STRAY_FIELD , BIAS_Y_STRAY_FIELD , BIAS_Z_STRAY_FIELD], dtype=jnp.float64)


# Define the atom chip wire layout
def setup_wire_layout(
    # PCB properties
    top_height             : float = PCB_LAYER_TOP_HEIGHT,
    bottom_height          : float = PCB_LAYER_BOTTOM_HEIGHT,
    x_offset               : float = PCB_X_OFFSET,
    y_offset               : float = PCB_Y_OFFSET,
    top_offset             : float = -(PCB_GLASS_HEIGHT + PCB_TOP_OFFSET + PCB_LAYER_TOP_HEIGHT / 2),
    bottom_offset          : float = -(PCB_GLASS_HEIGHT + PCB_TOP_OFFSET + PCB_LAYER_TOP_HEIGHT +
                                       PCB_CORE_HEIGHT + PCB_LAYER_BOTTOM_HEIGHT / 2),
    pcb_pin_length         : float = PBC_PIN_LENGTH,
    # Shifting wire properties
    shifting_wire_length   : float = SHIFTING_WIRE_LENGTH,
    shifting_wire_width    : float = SHIFTING_WIRE_WIDTH,
    # Guiding wire properties
    guiding_wire_length    : float = GUIDING_WIRE_LENGTH,
) -> ac.AtomChip:
    # Shifting wire length, width, and height
    SL = shifting_wire_length / 2  # half length
    SW = shifting_wire_width       # width
    SH = top_height                # height

    shifting_wires = [
        # [[start point], [end point], width, height]
        # left-most period:
        [[-5.6,  -SL-1.2,  0], [-5.6, SL-1.2,  0], SW, SH],   # WIRE T1
        [[-5.2,  -SL-0.7,  0], [-5.2, SL-0.7,  0], SW, SH],   # WIRE T2
        [[-4.8,  -SL-0.2,  0], [-4.8, SL-0.2,  0], SW, SH],   # WIRE T3
        [[-4.4,  -SL+0.3,  0], [-4.4, SL+0.3,  0], SW, SH],   # WIRE T4
        [[-4.0,  -SL+0.8,  0], [-4.0, SL+0.8,  0], SW, SH],   # WIRE T5
        [[-3.6,  -SL+1.3,  0], [-3.6, SL+1.3,  0], SW, SH],   # WIRE T6
        # left period:
        [[-3.2,  -SL-1.2,  0], [-3.2, SL-1.2,  0], SW, SH],   # WIRE T1 opposite sign
        [[-2.8,  -SL-0.7,  0], [-2.8, SL-0.7,  0], SW, SH],   # WIRE T2 opposite sign
        [[-2.4,  -SL-0.2,  0], [-2.4, SL-0.2,  0], SW, SH],   # WIRE T3 opposite sign
        [[-2.0,  -SL+0.3,  0], [-2.0, SL+0.3,  0], SW, SH],   # WIRE T4 opposite sign
        [[-1.6,  -SL+0.8,  0], [-1.6, SL+0.8,  0], SW, SH],   # WIRE T5 opposite sign
        [[-1.2,  -SL+1.3,  0], [-1.2, SL+1.3,  0], SW, SH],   # WIRE T6 opposite sign
        # center period:
        [[-0.8,  -SL-1.2,  0], [-0.8, SL-1.2,  0], SW, SH],   # WIRE T1
        [[-0.4,  -SL-0.7,  0], [-0.4, SL-0.7,  0], SW, SH],   # WIRE T2
        [[-0.0,  -SL-0.2,  0], [-0.0, SL-0.2,  0], SW, SH],   # WIRE T3
        [[ 0.4,  -SL+0.3,  0], [ 0.4, SL+0.3,  0], SW, SH],   # WIRE T4
        [[ 0.8,  -SL+0.8,  0], [ 0.8, SL+0.8,  0], SW, SH],   # WIRE T5
        [[ 1.2,  -SL+1.3,  0], [ 1.2, SL+1.3,  0], SW, SH],   # WIRE T6
        # right period:
        [[ 1.6,  -SL-1.2,  0], [ 1.6, SL-1.2,  0], SW, SH],   # WIRE T1 opposite sign
        [[ 2.0,  -SL-0.7,  0], [ 2.0, SL-0.7,  0], SW, SH],   # WIRE T2 opposite sign
        [[ 2.4,  -SL-0.2,  0], [ 2.4, SL-0.2,  0], SW, SH],   # WIRE T3 opposite sign
        [[ 2.8,  -SL+0.3,  0], [ 2.8, SL+0.3,  0], SW, SH],   # WIRE T4 opposite sign
        [[ 3.2,  -SL+0.8,  0], [ 3.2, SL+0.8,  0], SW, SH],   # WIRE T5 opposite sign
        [[ 3.6,  -SL+1.3,  0], [ 3.6, SL+1.3,  0], SW, SH],   # WIRE T6 opposite sign
        # right-most period:
        [[ 4.0,  -SL-1.2,  0], [ 4.0, SL-1.2,  0], SW, SH],   # WIRE T1
        [[ 4.4,  -SL-0.7,  0], [ 4.4, SL-0.7,  0], SW, SH],   # WIRE T2
        [[ 4.8,  -SL-0.2,  0], [ 4.8, SL-0.2,  0], SW, SH],   # WIRE T3
        [[ 5.2,  -SL+0.3,  0], [ 5.2, SL+0.3,  0], SW, SH],   # WIRE T4
        [[ 5.6,  -SL+0.8,  0], [ 5.6, SL+0.8,  0], SW, SH],   # WIRE T5
        [[ 6.0,  -SL+1.3,  0], [ 6.0, SL+1.3,  0], SW, SH],   # WIRE T6
    ]

    for wire in shifting_wires:
        # start
        wire[0][0] += x_offset
        wire[0][1] += y_offset
        wire[0][2] += top_offset
        # end
        wire[1][0] += x_offset
        wire[1][1] += y_offset
        wire[1][2] += top_offset

    #----------------------------------------------------------
    # PBC Bottom Layer (Quadrupole wires)
    #----------------------------------------------------------

    # Guiding wire length and height
    GL = guiding_wire_length / 2  # half length
    GH = bottom_height

    # PCB pin length (leg length)
    PL = pcb_pin_length

    # [[start point], [end point], width, height]
    guiding_wires = [[
        # WIRE Q4
        [[-GL             ,  4.9              ,   0], [ GL             ,  4.9              ,   0], 1.5, GH], # bar
    ], [  
        # WIRE Q3
        [[-40.2           , 14                , -PL], [-40.2           , 14                ,   0], 1.0, GH], # leg
        [[-GL-2.03-1.8-6.6,  3.05+1.78+2.6+6.6,   0], [-GL-2.03-1.8    ,  3.05+1.78+2.6    ,   0], 2.0, GH],
        [[-GL-2.03-1.8    ,  3.05+1.78+2.6    ,   0], [-GL-2.03-1.8    ,  3.05+1.78        ,   0], 2.0, GH],
        [[-GL-2.03-1.8    ,  3.05+1.78        ,   0], [-GL-2.03        ,  3.05             ,   0], 2.0, GH],
        [[-GL-2.03        ,  3.05             ,   0], [ GL+2.03        ,  3.05             ,   0], 2.0, GH], # bar
        [[ GL+2.03        ,  3.05             ,   0], [ GL+2.03+1.8    ,  3.05+1.78        ,   0], 2.0, GH],
        [[ GL+2.03+1.8    ,  3.05+1.78        ,   0], [ GL+2.03+1.8    ,  3.05+1.78+2.6    ,   0], 2.0, GH],
        [[ GL+2.03+1.8    ,  3.05+1.78+2.6    ,   0], [ GL+2.03+1.8+6.6,  3.05+1.78+2.6+6.6,   0], 2.0, GH],
        [[ 40.2           , 14                ,   0], [ 40.2           , 14                , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q2
        [[-40.2           ,  8.47             , -PL], [-40.2           ,  8.47             ,   0], 1.0, GH], # leg
        [[-GL-3.2-6.4     ,  1.45+6.65        ,   0], [-GL-3.2         ,  1.45             ,   0], 1.0, GH],
        [[-GL-3.2         ,  1.45             ,   0], [ GL+3.2         ,  1.45             ,   0], 1.0, GH], # bar
        [[ GL+3.2         ,  1.45             ,   0], [ GL+3.2+6.4     ,  1.45+6.65        ,   0], 1.0, GH], 
        [[ 40.2           ,  8.47             ,   0], [ 40.2           ,  8.47             , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q1
        [[-40.2           ,  3.7              , -PL], [-40.2           ,  3.7              ,   0], 1.0, GH], # leg
        [[-GL-3.875-5     ,  0.6+1.8          ,   0], [-GL-3.875       ,  0.6              ,   0], 0.5, GH],
        [[-GL-3.875       ,  0.6              ,   0], [ GL+3.875       ,  0.6              ,   0], 0.5, GH], # bar
        [[ GL+3.875       ,  0.6              ,   0], [ GL+3.875+5     ,  0.6+1.8          ,   0], 0.5, GH],
        [[ 40.2           ,  3.7              ,   0], [ 40.2           ,  3.7              , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q0 (center)
        [[-40.2           ,  0.0              , -PL], [-40.2           ,  0.0              ,   0], 1.0, GH], # leg
        [[-72/2-2.9       ,  0.0              ,   0], [ 72/2+2.9       ,  0.0              ,   0], 0.5, GH], # bar
        [[ 40.2           ,  0.0              ,   0], [ 40.2           ,  0.0              , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q1'
        [[-40.2           , -3.7              , -PL], [-40.2           , -3.7              ,   0], 1.0, GH], # leg
        [[-GL-3.875-5     , -0.6-1.8          ,   0], [-GL-3.875       , -0.6              ,   0], 0.5, GH],
        [[-GL-3.875       , -0.6              ,   0], [ GL+3.875       , -0.6              ,   0], 0.5, GH], # bar
        [[ GL+3.875       , -0.6              ,   0], [ GL+3.875+5     , -0.6-1.8          ,   0], 0.5, GH],
        [[ 40.2           , -3.7              ,   0], [ 40.2           , -3.7              , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q2'
        [[-40.2           , -8.47             , -PL], [-40.2           , -8.47             ,   0], 1.0, GH], # leg
        [[-GL-3.2-6.4     , -1.45-6.65        ,   0], [-GL-3.2         , -1.45             ,   0], 1.0, GH],
        [[-GL-3.2         , -1.45             ,   0], [ GL+3.2         , -1.45             ,   0], 1.0, GH], # bar
        [[ GL+3.2         , -1.45             ,   0], [ GL+3.2+6.4     , -1.45-6.65        ,   0], 1.0, GH],
        [[ 40.2           , -8.47             ,   0], [ 40.2           , -8.47             , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q3'
        [[-40.2           , -14               , -PL], [-40.2           , -14               ,   0], 1.0, GH], # leg
        [[-GL-2.03-1.8-6.6, -3.05-1.78-2.6-6.6,   0], [-GL-2.03-1.8    , -3.05-1.78-2.6    ,   0], 2.0, GH],
        [[-GL-2.03-1.8    , -3.05-1.78-2.6    ,   0], [-GL-2.03-1.8    , -3.05-1.78        ,   0], 2.0, GH],
        [[-GL-2.03-1.8    , -3.05-1.78        ,   0], [-GL-2.03        , -3.05             ,   0], 2.0, GH],
        [[-GL-2.03        , -3.05             ,   0], [ GL+2.03        , -3.05             ,   0], 2.0, GH], # bar
        [[ GL+2.03        , -3.05             ,   0], [ GL+2.03+1.8    , -3.05-1.78        ,   0], 2.0, GH],
        [[ GL+2.03+1.8    , -3.05-1.78        ,   0], [ GL+2.03+1.8    , -3.05-1.78-2.6    ,   0], 2.0, GH],
        [[ GL+2.03+1.8    , -3.05-1.78-2.6    ,   0], [ GL+2.03+1.8+6.6, -3.05-1.78-2.6-6.6,   0], 2.0, GH],
        [[ 40.2           , -14               ,   0], [ 40.2           , -14               , -PL], 1.0, GH], # leg
    ], [
        # WIRE Q4'
        [[-GL             , -4.9              ,   0], [ GL             , -4.9              ,   0], 1.5, GH], # bar
    ]]

    for wires in guiding_wires:
        for wire in wires:
            # start
            wire[0][0] += x_offset
            wire[0][1] += y_offset
            wire[0][2] += bottom_offset
            # end
            wire[1][0] += x_offset
            wire[1][1] += y_offset
            wire[1][2] += bottom_offset

    return shifting_wires, guiding_wires


def make_bias_fields(
    coil_currents: jnp.ndarray = COIL_CURRENTS,
    coil_factors: jnp.ndarray = COIL_FACTORS,
    stray_fields: jnp.ndarray = STRAY_FIELDS,
) -> ac.field.BiasFields:
    return ac.field.BiasFields(
        currents     = coil_currents,  # Currents applied to external coils [A]
        coil_factors = coil_factors,   # Current to Field Conversion [G/A]
        stray_fields = stray_fields,   # Stray field offsets [G]
    )


def build_atom_chip(
    shifting_wires: List[ac.components.RectangularSegmentType],
    shifting_wire_currents: jnp.ndarray,
    guiding_wires: List[ac.components.RectangularSegmentType],
    guiding_wire_currents: jnp.ndarray,
    bias_fields: ac.field.BiasFields,
) -> ac.AtomChip:
    # Define the PCB top layer shifting wires
    top_layer = [
        ac.components.RectangularConductor.create(
                material = "gold",
                current  = current,  # Current (A)
                segments = [wire,],
        ) for wire, current in zip(shifting_wires, shifting_wire_currents)
    ]

    # Define the PCB bottom layer guiding wires
    bottom_layer = [
        ac.components.RectangularConductor.create(
                    material = "copper",
                    current  = current,  # Current (A)
                    segments = wires,
        ) for wires, current in zip(guiding_wires, guiding_wire_currents)
    ]

    atom_chip = ac.AtomChip(
        name        = "BEC Transport",
        atom        = ac.rb87,
        components  = top_layer + bottom_layer,
        bias_fields = bias_fields,
    )
    return atom_chip


def trap_potential_analysis():
    # Define the wire layout
    shifting_wires, guiding_wires = setup_wire_layout()

    # shifting wire currents (A)
    shifting_wire_currents_reversed = [-i for i in SHIFTING_WIRE_CURRENTS]
    shifting_wire_currents = jnp.array(
        SHIFTING_WIRE_CURRENTS +
        shifting_wire_currents_reversed +
        SHIFTING_WIRE_CURRENTS +
        shifting_wire_currents_reversed +
        SHIFTING_WIRE_CURRENTS,
        dtype=jnp.float64,
    )
    guiding_wire_currents = jnp.array(GUIDING__WIRE_CURRENTS, dtype=jnp.float64)

    bias_fields = make_bias_fields()

    atom_chip = build_atom_chip(
        shifting_wires=shifting_wires,
        shifting_wire_currents=shifting_wire_currents,
        guiding_wires=guiding_wires,
        guiding_wire_currents=guiding_wire_currents,
        bias_fields=bias_fields,
    )

    # Define the analysis options
    # fmt: on
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
            method = "jax",
            #method = "finite-difference",
            #hessian_step = 1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
    )
    # fmt: on

    # logging level
    logging.basicConfig(level=logging.INFO)

    # Perform the analysis
    analysis = atom_chip.analyze(options)

    # Save the atom chip layout to a JSON file
    directory = os.path.dirname(__file__)
    atom_chip.save(os.path.join(directory, "transport.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, analysis, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    trap_potential_analysis()

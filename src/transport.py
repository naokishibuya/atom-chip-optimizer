import os
import logging
from typing import List
import jax.numpy as jnp
import atom_chip as ac


# fmt: off
#-------------------------------------------------------------
# PCB
#-------------------------------------------------------------

GLASS_HEIGHT            : float = 0.0   # 0.045; 0.1094; Half-wave plate thickness
PCB_LAYER_TOP_HEIGHT    : float = 0.07  # PCB Top Layer tickness (2 oz copper)
PCB_CORE_HEIGHT         : float = 0.38  # 0.42; %0.38;
PCB_LAYER_BOTTOM_HEIGHT : float = 0.07  # PCB Bottom Layer tickness (2 oz copper)
PCB_Y_OFFSET            : float = 0.0
PCB_X_OFFSET            : float = 0.0
PCB_TOP_OFFSET          : float = 0.0   # PCB top layer vertical offset from reflecting surface (after QWP fell)

#-------------------------------------------------------------
# Transport wires
#-------------------------------------------------------------
PCB_TRANSPORT_WIRE_LENGTH   : float = 16.0  # Transport wire length
PCB_TRANSPORT_WIRE_WIDTH    : float = 0.3   # Transport wire width
PCB_TRANSPORT_WIRE_CURRENTS : List[float] = [0.53, 0.89, -0.97, 0.89, 0.53, 0]  # transport wire current (A)

#-------------------------------------------------------------
# Quadrupole wires
#-------------------------------------------------------------
PCB_QUADRUPOLE_WIRE_LENGTH    : float = 62.0  # Quadrupole wire half length
PBC_PIN_LENGTH                : float = 50.0  # PCB pin length (leg length)
PCB_QUADRUPOLE_WIRE_CURRENT_0 : float = -3.7  # central quadrupole wire current (A)
PCB_QUADRUPOLE_WIRE_CURRENT_1 : float = 13.8  # outer quadrupole wire current (A)
PCB_QUADRUPOLE_WIRE_CURRENTS  : List[float] = [
                                0, # PCB Wire Q4
    PCB_QUADRUPOLE_WIRE_CURRENT_1, # PCB Wire Q3
    PCB_QUADRUPOLE_WIRE_CURRENT_1, # PCB Wire Q2
    PCB_QUADRUPOLE_WIRE_CURRENT_0, # PCB Wire Q1
    PCB_QUADRUPOLE_WIRE_CURRENT_0, # PCB Wire Q0
    PCB_QUADRUPOLE_WIRE_CURRENT_0, # PCB Wire Q1'
    PCB_QUADRUPOLE_WIRE_CURRENT_1, # PCB Wire Q2'
    PCB_QUADRUPOLE_WIRE_CURRENT_1, # PCB Wire Q3'
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

# PCB stray fields (G)
BIAS_X_STRAY_FIELD  : float = 1.0
BIAS_Y_STRAY_FIELD  : float = 1.0
BIAS_Z_STRAY_FIELD  : float = 0.0


# Define the PCB atom chip
def build_atom_chip(
    # PCB properties
    top_height              : float = PCB_LAYER_TOP_HEIGHT,
    bottom_height           : float = PCB_LAYER_BOTTOM_HEIGHT,
    x_offset                : float = PCB_X_OFFSET,
    y_offset                : float = PCB_Y_OFFSET,
    top_offset              : float = -(GLASS_HEIGHT + PCB_TOP_OFFSET + PCB_LAYER_TOP_HEIGHT / 2),
    bottom_offset           : float = -(GLASS_HEIGHT + PCB_TOP_OFFSET + PCB_LAYER_TOP_HEIGHT +
                                      PCB_CORE_HEIGHT + PCB_LAYER_BOTTOM_HEIGHT / 2),
    pcb_pin_length          : float = PBC_PIN_LENGTH,
    # Transport wire properties
    transport_wire_length   : float = PCB_TRANSPORT_WIRE_LENGTH,
    transport_wire_width    : float = PCB_TRANSPORT_WIRE_WIDTH,
    transport_wire_currents : List[float] = PCB_TRANSPORT_WIRE_CURRENTS,
    # Quadrupole wire properties
    quadrupole_wire_length  : float = PCB_QUADRUPOLE_WIRE_LENGTH,
    quadrupole_wire_currents: List[float] = PCB_QUADRUPOLE_WIRE_CURRENTS,
    # Bias field properties
    coil_factors            : jnp.ndarray = jnp.array([BIAS_X_COIL_FACTOR , BIAS_Y_COIL_FACTOR , BIAS_Z_COIL_FACTOR]),
    coil_currents           : jnp.ndarray = jnp.array([BIAS_X_COIL_CURRENT, BIAS_Y_COIL_CURRENT, BIAS_Z_COIL_CURRENT]),
    stray_fields            : jnp.ndarray = jnp.array([BIAS_X_STRAY_FIELD , BIAS_Y_STRAY_FIELD , BIAS_Z_STRAY_FIELD]),
) -> ac.AtomChip:
    # Transport wire length, width, and height
    TL = transport_wire_length / 2  # half length
    TW = transport_wire_width       # width
    TH = top_height                 # height

    transport_wires = [
        # [[start point], [end point], width, height]
        # left-most period:
        [[-5.6,  -TL-1.2,  0], [-5.6, TL-1.2,  0], TW, TH],   # WIRE T1
        [[-5.2,  -TL-0.7,  0], [-5.2, TL-0.7,  0], TW, TH],   # WIRE T2
        [[-4.8,  -TL-0.2,  0], [-4.8, TL-0.2,  0], TW, TH],   # WIRE T3
        [[-4.4,  -TL+0.3,  0], [-4.4, TL+0.3,  0], TW, TH],   # WIRE T4
        [[-4.0,  -TL+0.8,  0], [-4.0, TL+0.8,  0], TW, TH],   # WIRE T5
        [[-3.6,  -TL+1.3,  0], [-3.6, TL+1.3,  0], TW, TH],   # WIRE T6
        # left period:
        [[-3.2,  -TL-1.2,  0], [-3.2, TL-1.2,  0], TW, TH],   # WIRE T1 opposite sign
        [[-2.8,  -TL-0.7,  0], [-2.8, TL-0.7,  0], TW, TH],   # WIRE T2 opposite sign
        [[-2.4,  -TL-0.2,  0], [-2.4, TL-0.2,  0], TW, TH],   # WIRE T3 opposite sign
        [[-2.0,  -TL+0.3,  0], [-2.0, TL+0.3,  0], TW, TH],   # WIRE T4 opposite sign
        [[-1.6,  -TL+0.8,  0], [-1.6, TL+0.8,  0], TW, TH],   # WIRE T5 opposite sign
        [[-1.2,  -TL+1.3,  0], [-1.2, TL+1.3,  0], TW, TH],   # WIRE T6 opposite sign
        # center period:
        [[-0.8,  -TL-1.2,  0], [-0.8, TL-1.2,  0], TW, TH],   # WIRE T1
        [[-0.4,  -TL-0.7,  0], [-0.4, TL-0.7,  0], TW, TH],   # WIRE T2
        [[-0.0,  -TL-0.2,  0], [-0.0, TL-0.2,  0], TW, TH],   # WIRE T3
        [[ 0.4,  -TL+0.3,  0], [ 0.4, TL+0.3,  0], TW, TH],   # WIRE T4
        [[ 0.8,  -TL+0.8,  0], [ 0.8, TL+0.8,  0], TW, TH],   # WIRE T5
        [[ 1.2,  -TL+1.3,  0], [ 1.2, TL+1.3,  0], TW, TH],   # WIRE T6
        # right period:
        [[ 1.6,  -TL-1.2,  0], [ 1.6, TL-1.2,  0], TW, TH],   # WIRE T1 opposite sign
        [[ 2.0,  -TL-0.7,  0], [ 2.0, TL-0.7,  0], TW, TH],   # WIRE T2 opposite sign
        [[ 2.4,  -TL-0.2,  0], [ 2.4, TL-0.2,  0], TW, TH],   # WIRE T3 opposite sign
        [[ 2.8,  -TL+0.3,  0], [ 2.8, TL+0.3,  0], TW, TH],   # WIRE T4 opposite sign
        [[ 3.2,  -TL+0.8,  0], [ 3.2, TL+0.8,  0], TW, TH],   # WIRE T5 opposite sign
        [[ 3.6,  -TL+1.3,  0], [ 3.6, TL+1.3,  0], TW, TH],   # WIRE T6 opposite sign
        # right-most period:
        [[ 4.0,  -TL-1.2,  0], [ 4.0, TL-1.2,  0], TW, TH],   # WIRE T1
        [[ 4.4,  -TL-0.7,  0], [ 4.4, TL-0.7,  0], TW, TH],   # WIRE T2
        [[ 4.8,  -TL-0.2,  0], [ 4.8, TL-0.2,  0], TW, TH],   # WIRE T3
        [[ 5.2,  -TL+0.3,  0], [ 5.2, TL+0.3,  0], TW, TH],   # WIRE T4
        [[ 5.6,  -TL+0.8,  0], [ 5.6, TL+0.8,  0], TW, TH],   # WIRE T5
        [[ 6.0,  -TL+1.3,  0], [ 6.0, TL+1.3,  0], TW, TH],   # WIRE T6
    ]

    for wire in transport_wires:
        # start
        wire[0][0] += x_offset
        wire[0][1] += y_offset
        wire[0][2] += top_offset
        # end
        wire[1][0] += x_offset
        wire[1][1] += y_offset
        wire[1][2] += top_offset

    # transport wire currents (A)
    transport_wire_currents_reversed = [-i for i in transport_wire_currents]
    transport_wire_currents_repeated = (
        transport_wire_currents +
        transport_wire_currents_reversed +
        transport_wire_currents +
        transport_wire_currents_reversed +
        transport_wire_currents 
    )

    # Define the PCB top layer transport wires
    pcblayer_top = []
    for wire, current in zip(transport_wires, transport_wire_currents_repeated):
        pcblayer_top.append(
            ac.components.RectangularConductor.create(
                material = "gold",
                current  = current,  # Current (A)
                segments = [wire,],
            )
        )

    #----------------------------------------------------------
    # PBC Bottom Layer (Quadrupole wires)
    #----------------------------------------------------------

    # Quadrupole wire length and height
    QL = quadrupole_wire_length / 2  # half length
    QH = bottom_height

    # PCB pin length (leg length)
    PL = pcb_pin_length

    # [[start point], [end point], width, height]
    quadrupole_wires = [[
        # WIRE Q4
        [[-QL             ,  4.9              ,   0], [ QL             ,  4.9              ,   0], 1.5, QH], # bar
    ], [  
        # WIRE Q3
        [[-40.2           , 14                , -PL], [-40.2           , 14                ,   0], 1.0, QH], # leg
        [[-QL-2.03-1.8-6.6,  3.05+1.78+2.6+6.6,   0], [-QL-2.03-1.8    ,  3.05+1.78+2.6    ,   0], 2.0, QH],
        [[-QL-2.03-1.8    ,  3.05+1.78+2.6    ,   0], [-QL-2.03-1.8    ,  3.05+1.78        ,   0], 2.0, QH],
        [[-QL-2.03-1.8    ,  3.05+1.78        ,   0], [-QL-2.03        ,  3.05             ,   0], 2.0, QH],
        [[-QL-2.03        ,  3.05             ,   0], [ QL+2.03        ,  3.05             ,   0], 2.0, QH], # bar
        [[ QL+2.03        ,  3.05             ,   0], [ QL+2.03+1.8    ,  3.05+1.78        ,   0], 2.0, QH],
        [[ QL+2.03+1.8    ,  3.05+1.78        ,   0], [ QL+2.03+1.8    ,  3.05+1.78+2.6    ,   0], 2.0, QH],
        [[ QL+2.03+1.8    ,  3.05+1.78+2.6    ,   0], [ QL+2.03+1.8+6.6,  3.05+1.78+2.6+6.6,   0], 2.0, QH],
        [[ 40.2           , 14                ,   0], [ 40.2           , 14                , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q2
        [[-40.2           ,  8.47             , -PL], [-40.2           ,  8.47             ,   0], 1.0, QH], # leg
        [[-QL-3.2-6.4     ,  1.45+6.65        ,   0], [-QL-3.2         ,  1.45             ,   0], 1.0, QH],
        [[-QL-3.2         ,  1.45             ,   0], [ QL+3.2         ,  1.45             ,   0], 1.0, QH], # bar
        [[ QL+3.2         ,  1.45             ,   0], [ QL+3.2+6.4     ,  1.45+6.65        ,   0], 1.0, QH], 
        [[ 40.2           ,  8.47             ,   0], [ 40.2           ,  8.47             , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q1
        [[-40.2           ,  3.7              , -PL], [-40.2           ,  3.7              ,   0], 1.0, QH], # leg
        [[-QL-3.875-5     ,  0.6+1.8          ,   0], [-QL-3.875       ,  0.6              ,   0], 0.5, QH],
        [[-QL-3.875       ,  0.6              ,   0], [ QL+3.875       ,  0.6              ,   0], 0.5, QH], # bar
        [[ QL+3.875       ,  0.6              ,   0], [ QL+3.875+5     ,  0.6+1.8          ,   0], 0.5, QH],
        [[ 40.2           ,  3.7              ,   0], [ 40.2           ,  3.7              , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q0 (center)
        [[-40.2           ,  0.0              , -PL], [-40.2           ,  0.0              ,   0], 1.0, QH], # leg
        [[-72/2-2.9       ,  0.0              ,   0], [ 72/2+2.9       ,  0.0              ,   0], 0.5, QH], # bar
        [[ 40.2           ,  0.0              ,   0], [ 40.2           ,  0.0              , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q1'
        [[-40.2           , -3.7              , -PL], [-40.2           , -3.7              ,   0], 1.0, QH], # leg
        [[-QL-3.875-5     , -0.6-1.8          ,   0], [-QL-3.875       , -0.6              ,   0], 0.5, QH],
        [[-QL-3.875       , -0.6              ,   0], [ QL+3.875       , -0.6              ,   0], 0.5, QH], # bar
        [[ QL+3.875       , -0.6              ,   0], [ QL+3.875+5     , -0.6-1.8          ,   0], 0.5, QH],
        [[ 40.2           , -3.7              ,   0], [ 40.2           , -3.7              , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q2'
        [[-40.2           , -8.47             , -PL], [-40.2           , -8.47             ,   0], 1.0, QH], # leg
        [[-QL-3.2-6.4     , -1.45-6.65        ,   0], [-QL-3.2         , -1.45             ,   0], 1.0, QH],
        [[-QL-3.2         , -1.45             ,   0], [ QL+3.2         , -1.45             ,   0], 1.0, QH], # bar
        [[ QL+3.2         , -1.45             ,   0], [ QL+3.2+6.4     , -1.45-6.65        ,   0], 1.0, QH],
        [[ 40.2           , -8.47             ,   0], [ 40.2           , -8.47             , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q3'
        [[-40.2           , -14               , -PL], [-40.2           , -14               ,   0], 1.0, QH], # leg
        [[-QL-2.03-1.8-6.6, -3.05-1.78-2.6-6.6,   0], [-QL-2.03-1.8    , -3.05-1.78-2.6    ,   0], 2.0, QH],
        [[-QL-2.03-1.8    , -3.05-1.78-2.6    ,   0], [-QL-2.03-1.8    , -3.05-1.78        ,   0], 2.0, QH],
        [[-QL-2.03-1.8    , -3.05-1.78        ,   0], [-QL-2.03        , -3.05             ,   0], 2.0, QH],
        [[-QL-2.03        , -3.05             ,   0], [ QL+2.03        , -3.05             ,   0], 2.0, QH], # bar
        [[ QL+2.03        , -3.05             ,   0], [ QL+2.03+1.8    , -3.05-1.78        ,   0], 2.0, QH],
        [[ QL+2.03+1.8    , -3.05-1.78        ,   0], [ QL+2.03+1.8    , -3.05-1.78-2.6    ,   0], 2.0, QH],
        [[ QL+2.03+1.8    , -3.05-1.78-2.6    ,   0], [ QL+2.03+1.8+6.6, -3.05-1.78-2.6-6.6,   0], 2.0, QH],
        [[ 40.2           , -14               ,   0], [ 40.2           , -14               , -PL], 1.0, QH], # leg
    ], [
        # WIRE Q4'
        [[-QL             , -4.9              ,   0], [ QL             , -4.9              ,   0], 1.5, QH], # bar
    ]]

    for wires in quadrupole_wires:
        for wire in wires:
            # start
            wire[0][0] += x_offset
            wire[0][1] += y_offset
            wire[0][2] += bottom_offset
            # end
            wire[1][0] += x_offset
            wire[1][1] += y_offset
            wire[1][2] += bottom_offset

    # Define the PCB bottom layer quadrupole wires
    pcblayer_bottom = []
    for i, wires in enumerate(quadrupole_wires):
        current = quadrupole_wire_currents[i]
        pcblayer_bottom.append(
            ac.components.RectangularConductor.create(
                material = "copper",
                current  = current,  # Current (A)
                segments = wires,
            )
        )

    # Bias field properties
    bias_fields = ac.field.BiasFields(
        currents     = coil_currents,  # Currents applied to external coids [A]
        coil_factors = coil_factors,   # Current to Field Conversion [G/A]
        stray_fields = stray_fields,   # Stray field offsets [G]
    )

    atom_chip = ac.AtomChip(
        name        = "PCB Transport",
        atom        = ac.rb87,
        components  = pcblayer_top + pcblayer_bottom,
        bias_fields = bias_fields,
    )
    return atom_chip
# fmt: on


def trap_potential_analysis():
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
            method = "jax",
            # method = "finite-difference",
            # hessian_step = 1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
    )
    # fmt: on

    # logging level
    logging.basicConfig(level=logging.INFO)

    # Build the atom chip
    atom_chip = build_atom_chip()

    # Perform the analysis
    analysis = atom_chip.analyze(options)

    # Save the atom chip layout to a JSON file
    directory = os.path.dirname(__file__)
    atom_chip.save(os.path.join(directory, "transport.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, analysis, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    trap_potential_analysis()

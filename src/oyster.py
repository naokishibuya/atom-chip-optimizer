import os
from typing import List, Tuple
import atom_chip as ac


# fmt: off
#-------------------------------------------------------------
# PCB
#-------------------------------------------------------------

GlassThickness        : float = 0.0   # 0.045; 0.1094; Half-wave plate thickness
height_PCBLAYERTOP    : float = 0.07  # PCB Top Layer tickness (2 oz copper)
PCBCoreThickness      : float = 0.38  # 0.42; %0.38;
height_PCBLAYERBOTTOM : float = 0.07  # PCB Bottom Layer tickness (2 oz copper)
PCB_Yoffset           : float = 0.0
PCB_Xoffset           : float = 0.0
PCBtop_offset         : float = 0.0   # PCB top layer vertical offset from reflecting surface (after QWP fell)

#-------------------------------------------------------------
# Transport wires
#-------------------------------------------------------------
PCB_TrWireLengths     : float = 16.0  # Transport wire length
PCB_TrWireWidth       : float = 0.3   # Transport wire width
PCB_TrWireCurrents    : List[float] = [0.53, 0.89, -0.97, 0.89, 0.53, 0]  # transport wire current (A)

#-------------------------------------------------------------
# Quadrupole wires
#-------------------------------------------------------------
PCB_QuadWireLengths   : float = 62.0  # Quadrupole wire half length
PCBpinL               : float = 50.0  # PCB pin length (leg length)
Q0                    : float = -3.7  # central quadrupole wire current (A)
Q1                    : float = 13.8  # outer quadrupole wire current (A)
PCB_QuadWireCurrents  : List[float] = [
     0, # PCB Wire Q4
    Q1, # PCB Wire Q3
    Q1, # PCB Wire Q2
    Q0, # PCB Wire Q1
    Q0, # PCB Wire Q0
    Q0, # PCB Wire Q1'
    Q1, # PCB Wire Q2'
    Q1, # PCB Wire Q3'
     0, # PCB Wire Q4'
]

#-------------------------------------------------------------
# Bias fields
#-------------------------------------------------------------
# Coil Current [A] to Field [G] Conversion:
Bias_X_CoilFactor  : float = -1.068  # [G/A]
Bias_Y_CoilFactor  : float =  1.8    # [G/A]
Bias_Z_CoilFactor  : float =  3.0    # [G/A]

# Coil currents [A] to be applied to the external coils
Bias_X_CoilCurrent : float = 0.0
Bias_Y_CoilCurrent : float = 0.0
Bias_Z_CoilCurrent : float = 0.0

# PCB stray fields (G)
Bias_X_StrayField  : float = 1.0
Bias_Y_StrayField  : float = 1.0
Bias_Z_StrayField  : float = 0.0


# Define the PCB atom chip
def build_atom_chip(
    # PCB properties
    top_height              : float = height_PCBLAYERTOP,
    bottom_height           : float = height_PCBLAYERBOTTOM,
    x_offset                : float = PCB_Xoffset,
    y_offset                : float = PCB_Yoffset,
    top_offset              : float = -(GlassThickness + PCBtop_offset + height_PCBLAYERTOP / 2),
    bottom_offset           : float = -(GlassThickness + PCBtop_offset + height_PCBLAYERTOP +
                                      PCBCoreThickness + height_PCBLAYERBOTTOM / 2),
    pcb_pin_length          : float = PCBpinL,
    # Transport wire properties
    transport_wire_length   : float = PCB_TrWireLengths,
    transport_wire_width    : float = PCB_TrWireWidth,
    transport_wire_currents : List[float] = PCB_TrWireCurrents,
    # Quadrupole wire properties
    quadrupole_wire_length  : float = PCB_QuadWireLengths,
    quadrupole_wire_currents: List[float] = PCB_QuadWireCurrents,
    # Bias field properties
    coil_factors            : Tuple[float, float, float] = (Bias_X_CoilFactor , Bias_Y_CoilFactor , Bias_Z_CoilFactor),
    coil_currents           : Tuple[float, float, float] = (Bias_X_CoilCurrent, Bias_Y_CoilCurrent, Bias_Z_CoilCurrent),
    stray_fields            : Tuple[float, float, float] = (Bias_X_StrayField , Bias_Y_StrayField , Bias_Z_StrayField),
) -> ac.AtomChip:
    # Transport wire length, width, and height
    TL = transport_wire_length / 2  # half length
    TW = transport_wire_width       # width
    TH = top_height                 # height

    wireCoords_PCBLAYERTOP = [
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

    for wire in wireCoords_PCBLAYERTOP:
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
    I_PCBLAYERTOP = (
        transport_wire_currents +
        transport_wire_currents_reversed +
        transport_wire_currents +
        transport_wire_currents_reversed +
        transport_wire_currents 
    )

    # Define the PCB top layer transport wires
    pcblayer_top = []
    for wire, current in zip(wireCoords_PCBLAYERTOP, I_PCBLAYERTOP):
        pcblayer_top.append(
            ac.components.RectangularConductor(
                material = "gold",
                current  = current,  # Current (A)
                segments = [wire,],
            )
        )

    #----------------------------------------------------------
    # PBC Bottom Layer
    #----------------------------------------------------------

    # Quadrupole wire length and height
    QL = quadrupole_wire_length / 2  # half length
    QH = bottom_height

    # PCB pin length (leg length)
    PL = pcb_pin_length

    # [[start point], [end point], width, height]
    wireCoords_PCBLAYERBOTTOM = [[
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

    for wires in wireCoords_PCBLAYERBOTTOM:
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
    for i, wires in enumerate(wireCoords_PCBLAYERBOTTOM):
        current = quadrupole_wire_currents[i]
        pcblayer_bottom.append(
            ac.components.RectangularConductor(
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
        name        = "PCB Oyster",
        atom        = ac.rb87,
        components  = pcblayer_top + pcblayer_bottom,
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

    # Build the atom chip
    atom_chip = build_atom_chip()

    # Perform the analysis
    atom_chip.analyze(options)

    # Save the atom chip layout to a JSON file
    directory = os.path.dirname(__file__)
    atom_chip.to_json(os.path.join(directory, "oyster.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    main()

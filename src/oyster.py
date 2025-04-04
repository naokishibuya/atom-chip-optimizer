import os
import atom_chip as ac
import jax
from offsets import Offsets


jax.config.update("jax_enable_x64", True)


# fmt: off

offsets = Offsets()
PCBpinL = 50

TL = offsets.transport_wire_length / 2  # Transport wire half length
TW = 0.3  # Transport wire width
QL = offsets.quadrupole_wire_length / 2  # Quadrupole wire half length

wireCoords_PCBLAYERTOP = [
    # [[start point], [end point], width, height]
    [[-0.8,  -1.2,  0], [-0.8, -1.2,  0]],   # WIRE T1
    [[-0.4,  -0.7,  0], [-0.4, -0.7,  0]],   # WIRE T2
    [[-0.0,  -0.2,  0], [-0.0, -0.2,  0]],   # WIRE T3
    [[ 0.4,   0.3,  0], [ 0.4,  0.3,  0]],   # WIRE T4
    [[ 0.8,   0.8,  0], [ 0.8,  0.8,  0]],   # WIRE T5
    [[ 1.2,   1.3,  0], [ 1.2,  1.3,  0]],   # WIRE T6
    # right hand side period:
    [[ 1.6,  -1.2,  0], [ 1.6, -1.2,  0]],   # WIRE T1 opposite sign
    [[ 2.0,  -0.7,  0], [ 2.0, -0.7,  0]],   # WIRE T2 opposite sign
    [[ 2.4,  -0.2,  0], [ 2.4, -0.2,  0]],   # WIRE T3 opposite sign
    [[ 2.8,   0.3,  0], [ 2.8,  0.3,  0]],   # WIRE T4 opposite sign
    [[ 3.2,   0.8,  0], [ 3.2,  0.8,  0]],   # WIRE T5 opposite sign
    [[ 3.6,   1.3,  0], [ 3.6,  1.3,  0]],   # WIRE T6 opposite sign
    # left hand side period:
    [[-3.2,  -1.2,  0], [-3.2, -1.2,  0]],   # WIRE T1 opposite sign
    [[-2.8,  -0.7,  0], [-2.8, -0.7,  0]],   # WIRE T2 opposite sign
    [[-2.4,  -0.2,  0], [-2.4, -0.2,  0]],   # WIRE T3 opposite sign
    [[-2.0,   0.3,  0], [-2.0,  0.3,  0]],   # WIRE T4 opposite sign
    [[-1.6,   0.8,  0], [-1.6,  0.8,  0]],   # WIRE T5 opposite sign
    [[-1.2,   1.3,  0], [-1.2,  1.3,  0]],   # WIRE T6 opposite sign
    # right hand side 2nd period:
    [[ 4.0,  -1.2,  0], [ 4.0, -1.2,  0]],   # WIRE T1 opposite sign
    [[ 4.4,  -0.7,  0], [ 4.4, -0.7,  0]],   # WIRE T2 opposite sign
    [[ 4.8,  -0.2,  0], [ 4.8, -0.2,  0]],   # WIRE T3 opposite sign
    [[ 5.2,   0.3,  0], [ 5.2,  0.3,  0]],   # WIRE T4 opposite sign
    [[ 5.6,   0.8,  0], [ 5.6,  0.8,  0]],   # WIRE T5 opposite sign
    [[ 6.0,   1.3,  0], [ 6.0,  1.3,  0]],   # WIRE T6 opposite sign
    # left hand side 2nd period:
    [[-5.6,  -1.2,  0], [-5.6, -1.2,  0]],   # WIRE T1 opposite sign
    [[-5.2,  -0.7,  0], [-5.2, -0.7,  0]],   # WIRE T2 opposite sign
    [[-4.8,  -0.2,  0], [-4.8, -0.2,  0]],   # WIRE T3 opposite sign
    [[-4.4,   0.3,  0], [-4.4,  0.3,  0]],   # WIRE T4 opposite sign
    [[-4.0,   0.8,  0], [-4.0,  0.8,  0]],   # WIRE T5 opposite sign
    [[-3.6,   1.3,  0], [-3.6,  1.3,  0]],   # WIRE T6 opposite sign
]

for i in range(len(wireCoords_PCBLAYERTOP)):
    wireCoords_PCBLAYERTOP[i][0][0] += offsets.x_offset
    wireCoords_PCBLAYERTOP[i][0][1] += offsets.y_offset - TL
    wireCoords_PCBLAYERTOP[i][0][2] += offsets.top_offset
    wireCoords_PCBLAYERTOP[i][1][0] += offsets.x_offset
    wireCoords_PCBLAYERTOP[i][1][1] += offsets.y_offset + TL
    wireCoords_PCBLAYERTOP[i][1][2] += offsets.top_offset
    wireCoords_PCBLAYERTOP[i].append(TW)
    wireCoords_PCBLAYERTOP[i].append(offsets.top_height)

Ivec=[0.53, 0.89, -0.97, 0.89, 0.53, 0]  # transport wire current (A)
minus_Ivec = [-i for i in Ivec]
I_PCBLAYERTOP = Ivec + minus_Ivec + minus_Ivec + Ivec + Ivec

# Define the PCB top layer transport wires
pcblayer_top = []
for i in range(len(wireCoords_PCBLAYERTOP)):
    current = I_PCBLAYERTOP[i]
    pcblayer_top.append(
        ac.components.RectangularConductor(
            material = "gold",
            current  = current,  # Current (A)
            segments = [wireCoords_PCBLAYERTOP[i],],
        )
    )

#----------------------------------------------------------
# PBC Bottom Layer
#----------------------------------------------------------

wireCoords_PCBLAYERBOTTOM = [
    # [[start point], [end point], width, height]
    # Center bars
    [[-QL             ,  4.9,  0], [QL      ,  4.9,  0]],  # WIRE Q4
    [[-QL-2.03        ,  3.05, 0], [QL+2.03 ,  3.05, 0]],  # WIRE Q3
    [[-QL-3.2         ,  1.45, 0], [QL+3.2  ,  1.45, 0]],  # WIRE Q2
    [[-QL-3.875       ,  0.6,  0], [QL+3.875,  0.6,  0]],  # WIRE Q1
    [[-72/2-2.9       ,  0.0,  0], [72/2+2.9,  0.0,  0]],  # WIRE Q0
    [[-QL-3.875       , -0.6,  0], [QL+3.875, -0.6,  0]],  # WIRE Q1'
    [[-QL-3.2         , -1.45, 0], [QL+3.2  , -1.45, 0]],  # WIRE Q2'
    [[-QL-2.03        , -3.05, 0], [QL+2.03 , -3.05, 0]],  # WIRE Q3'
    [[-QL             , -4.9,  0], [QL      , -4.9,  0]],  # WIRE Q4'

    [[-QL-3.875-5     ,  0.6+1.8,   0], [-QL-3.875  ,  0.6      , 0]], # WIRE Q1
    [[ QL+3.875       ,  0.6,       0], [ QL+3.875+5,  0.6+1.8  , 0]], # WIRE Q1
    [[-QL-3.2-6.4     ,  1.45+6.65, 0], [-QL-3.2    ,  1.45     , 0]], # WIRE Q2
    [[ QL+3.2         ,  1.45,      0], [ QL+3.2+6.4,  1.45+6.65, 0]], # WIRE Q2
    [[-QL-3.875-5     , -0.6-1.8,   0], [-QL-3.875  , -0.6      , 0]], # WIRE Q1'
    [[ QL+3.875       , -0.6,       0], [ QL+3.875+5, -0.6-1.8  , 0]], # WIRE Q1'
    [[-QL-3.2-6.4     , -1.45-6.65, 0], [-QL-3.2    , -1.45     , 0]], # WIRE Q2'
    [[ QL+3.2         , -1.45,      0], [ QL+3.2+6.4, -1.45-6.65, 0]], # WIRE Q2'

    [[-QL-2.03-1.8-6.6,  3.05+1.78+2.6+6.6, 0], [-QL-2.03-1.8    ,  3.05+1.78+2.6    , 0]], # WIRE Q3
    [[-QL-2.03-1.8    ,  3.05+1.78+2.6,     0], [-QL-2.03-1.8    ,  3.05+1.78        , 0]], # WIRE Q3
    [[-QL-2.03-1.8    ,  3.05+1.78,         0], [-QL-2.03        ,  3.05             , 0]], # WIRE Q3
    [[ QL+2.03        ,  3.05,              0], [ QL+2.03+1.8    ,  3.05+1.78        , 0]], # WIRE Q3
    [[ QL+2.03+1.8    ,  3.05+1.78,         0], [ QL+2.03+1.8    ,  3.05+1.78+2.6    , 0]], # WIRE Q3
    [[ QL+2.03+1.8    ,  3.05+1.78+2.6,     0], [ QL+2.03+1.8+6.6,  3.05+1.78+2.6+6.6, 0]], # WIRE Q3

    [[-QL-2.03-1.8-6.6, -3.05-1.78-2.6-6.6, 0], [-QL-2.03-1.8    , -3.05-1.78-2.6    , 0]], # WIRE Q3'
    [[-QL-2.03-1.8    , -3.05-1.78-2.6,     0], [-QL-2.03-1.8    , -3.05-1.78        , 0]], # WIRE Q3'
    [[-QL-2.03-1.8    , -3.05-1.78,         0], [-QL-2.03        , -3.05             , 0]], # WIRE Q3'
    [[ QL+2.03        , -3.05,              0], [ QL+2.03+1.8    , -3.05-1.78        , 0]], # WIRE Q3'
    [[ QL+2.03+1.8    , -3.05-1.78,         0], [ QL+2.03+1.8    , -3.05-1.78-2.6    , 0]], # WIRE Q3'
    [[ QL+2.03+1.8    , -3.05-1.78-2.6,     0], [ QL+2.03+1.8+6.6, -3.05-1.78-2.6-6.6, 0]], # WIRE Q3'

    # legs at both ends
    [[-40.2, 14   , -PCBpinL], [-40.2,  14   ,  0      ]],  # WIRE Q3
    [[ 40.2, 14   ,  0      ], [ 40.2,  14   , -PCBpinL]],  # WIRE Q3
    [[-40.2,  8.47, -PCBpinL], [-40.2,   8.47,  0      ]],  # WIRE Q2
    [[ 40.2,  8.47,  0      ], [ 40.2,   8.47, -PCBpinL]],  # WIRE Q2
    [[-40.2,  3.7 , -PCBpinL], [-40.2,   3.7 ,  0      ]],  # WIRE Q1
    [[ 40.2,  3.7 ,  0      ], [ 40.2,   3.7 , -PCBpinL]],  # WIRE Q1
    [[-40.2,  0.0 , -PCBpinL], [-40.2,   0.0 ,  0      ]],  # WIRE Q0
    [[ 40.2,  0.0 ,  0      ], [ 40.2,   0.0 , -PCBpinL]],  # WIRE Q0
    [[-40.2, -3.7 , -PCBpinL], [-40.2,  -3.7 ,  0      ]],  # WIRE Q1'
    [[ 40.2, -3.7 ,  0      ], [ 40.2,  -3.7 , -PCBpinL]],  # WIRE Q1'
    [[-40.2, -8.47, -PCBpinL], [-40.2,  -8.47,  0      ]],  # WIRE Q2'
    [[ 40.2, -8.47,  0      ], [ 40.2,  -8.47, -PCBpinL]],  # WIRE Q2'
    [[-40.2, -14  , -PCBpinL], [-40.2, -14   ,  0      ]],  # WIRE Q3'
    [[ 40.2, -14  ,  0      ], [ 40.2, -14   , -PCBpinL]],  # WIRE Q3'
]

width_PCBLAYERBOTTOM = [
    1.5,
    2.0,
    1.0,
    0.5,
    0.5,
    0.5,
    1.0,
    2.0,
    1.5,

    0.5,
    0.5,
    1.0,
    1.0,
    0.5,
    0.5,
    1.0,
    1.0,

    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,

    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,

    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

for i in range(len(wireCoords_PCBLAYERBOTTOM)):
    # start
    wireCoords_PCBLAYERBOTTOM[i][0][0] += offsets.x_offset
    wireCoords_PCBLAYERBOTTOM[i][0][1] += offsets.y_offset
    wireCoords_PCBLAYERBOTTOM[i][0][2] += offsets.bottom_offset
    # end
    wireCoords_PCBLAYERBOTTOM[i][1][0] += offsets.x_offset
    wireCoords_PCBLAYERBOTTOM[i][1][1] += offsets.y_offset
    wireCoords_PCBLAYERBOTTOM[i][1][2] += offsets.bottom_offset
    # width, height
    wireCoords_PCBLAYERBOTTOM[i].append(width_PCBLAYERBOTTOM[i])
    wireCoords_PCBLAYERBOTTOM[i].append(offsets.bottom_height)

Q0= -3.7  # central quadrupole wire current (A)
Q1= 13.8  # outer quadrupole wire current (A)

PCB_WireQ4  = 0
PCB_WireQ3  = Q1
PCB_WireQ2  = Q1
PCB_WireQ1  = Q0
PCB_WireQ0  = Q0
PCB_WireQ1p = Q0
PCB_WireQ2p = Q1
PCB_WireQ3p = Q1
PCB_WireQ4p = 0

I_PCBLAYERBOTTOM = [
    PCB_WireQ4,
    PCB_WireQ3,
    PCB_WireQ2,
    PCB_WireQ1,
    PCB_WireQ0,
    PCB_WireQ1p,
    PCB_WireQ2p,
    PCB_WireQ3p,
    PCB_WireQ4p,

    PCB_WireQ1,
    PCB_WireQ1,
    PCB_WireQ2,
    PCB_WireQ2,
    PCB_WireQ1p,
    PCB_WireQ1p,
    PCB_WireQ2p,
    PCB_WireQ2p,

    PCB_WireQ3,
    PCB_WireQ3,
    PCB_WireQ3,
    PCB_WireQ3,
    PCB_WireQ3,
    PCB_WireQ3,

    PCB_WireQ3p,
    PCB_WireQ3p,
    PCB_WireQ3p,
    PCB_WireQ3p,
    PCB_WireQ3p,
    PCB_WireQ3p,

    PCB_WireQ3,
    PCB_WireQ3,
    PCB_WireQ2,
    PCB_WireQ2,
    PCB_WireQ1,
    PCB_WireQ1,
    PCB_WireQ0,
    PCB_WireQ0,
    PCB_WireQ1p,
    PCB_WireQ1p,
    PCB_WireQ2p,
    PCB_WireQ2p,
    PCB_WireQ3p,
    PCB_WireQ3p,
]

# Define the PCB bottom layer quadrupole wires
pcblayer_bottom = []
for i in range(len(wireCoords_PCBLAYERBOTTOM)):
    current = I_PCBLAYERBOTTOM[i]
    pcblayer_bottom.append(
        ac.components.RectangularConductor(
            material = "copper",
            current  = current,  # Current (A)
            segments = [wireCoords_PCBLAYERBOTTOM[i],],
        )
    )


# Bias field properties
bias_fields = ac.field.BiasFields(
    currents     = [0.0, 0.0, 0.0],  # Currents applied to external coids [A]
    coil_factors = [-1.068, 1.8, 3.0], # Current to Field Conversion [G/A]
    stray_fields = [1.0, 1.0, 0.0],   # Stray field offsets [G]
)


# fmt: on


# Define the PCB atom chip
def main():
    atom_chip = ac.AtomChip(
        name="PCB",
        atom=ac.rb87,
        components=pcblayer_top + pcblayer_bottom,
        bias_fields=bias_fields,
    )

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
        total_atoms=1e6,
        condensed_atoms=1e4,
        verbose = True,
    )
    # fmt: on

    atom_chip.analyze(options)
    directory = os.path.dirname(__file__)
    atom_chip.to_json(os.path.join(directory, "oyster.json"))
    ac.visualization.show(atom_chip, os.path.join(directory, "oyster.yaml"))


if __name__ == "__main__":
    main()

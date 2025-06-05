import os
import logging
import atom_chip as ac
from atom_chip.transport import builder


def main():
    """
    Initial Static Transport Trap Analysis
    """
    # Define the wire layout
    shifting_wires, guiding_wires = builder.setup_wire_layout()
    shifting_wire_currents, guiding_wire_currents = builder.setup_wire_currents()

    bias_fields = builder.make_bias_fields()

    atom_chip = builder.build_atom_chip(
        shifting_wires=shifting_wires,
        shifting_wire_currents=shifting_wire_currents,
        guiding_wires=guiding_wires,
        guiding_wire_currents=guiding_wire_currents,
        bias_fields=bias_fields,
    )

    # Define the analysis options
    # fmt: on
    options = ac.potential.AnalysisOptions(
        search=dict(
            x0=[0.0, 0.0, 0.5],  # Initial guess
            bounds=[(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)],
            method="Nelder-Mead",
            options=dict(
                xatol=1e-10,
                fatol=1e-10,
                maxiter=int(1e5),
                maxfev=int(1e5),
                disp=True,
            ),
        ),
        hessian=dict(
            # method = "jax",
            method="finite-difference",
            hessian_step=1e-5,  # Step size for Hessian calculation
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
    main()

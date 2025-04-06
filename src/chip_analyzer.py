import os
import atom_chip as ac


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
        condensed_atoms=5e4,
        verbose = True,
    )
    # fmt: on

    # Build the atom chip
    # fmt: off
    atom_chip = ac.AtomChip(
        name        = "Atom Chip Analyzer",
        atom        = ac.rb87,
        components  = [],
        bias_fields = ac.field.ZERO_BIAS_FIELD,
    )
    # fmt: on

    # Parse command line arguments
    directory = os.path.dirname(__file__)
    atom_chip.from_json(os.path.join(directory, "chip_tester.json"))

    # Analyze the atom chip
    atom_chip.analyze(options)

    # Perform the visualization
    ac.visualization.show(atom_chip, os.path.join(directory, "chip_analyzer.yaml"))


if __name__ == "__main__":
    main()

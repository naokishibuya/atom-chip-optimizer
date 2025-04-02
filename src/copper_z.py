import atom_chip as ac
from offsets import Offsets


def main():
    offsets = Offsets()
    atom_chip = ac.AtomChip(
        name="example_chip",
        atom=ac.rb87,
        components=[
            ac.components.RectangularConductor(
                material="copper",
                current=85.0,
                segments=[
                    # [[start point], [end point], width, height]
                    [[0, -40.2, -127.5], [0, -40.2, -34.5], 3, 3],
                    [[0, -37.2, -31.0], [0, -25.0, -31.0], 5, 7],
                    [[0, -22.5, -27.5], [0, -22.5, -2.5], 3, 3],
                    [[-3.25, -17.75, -1.25], [-3.25, -6.5, -1.25], 1.5, 2.5],
                    [[-3.25, -6.5, -0.5], [-3.25, -0.15, -0.5], 1.5, 1],
                    [[-3.0, 0.0, -0.5], [3.0, 0.0, -0.5], 1.5, 1],
                    [[3.25, 0.15, -0.5], [3.25, 6.5, -0.5], 1.5, 1],
                    [[3.25, 6.5, -1.25], [3.25, 17.75, -1.25], 1.5, 2.5],
                    [[0, 22.5, -2.5], [0, 22.5, -27.5], 3, 3],
                    [[0, 25.0, -31.0], [0, 37.2, -31.0], 5, 7],
                    [[0, 40.2, -34.5], [0, 40.2, -127.5], 3, 3],
                ],
                z_offset=offsets.copper_z_offset,
            ),
            ac.components.RectangularConductor(
                material="copper",
                current=0.0,
                segments=[
                    # [[start point], [end point], width, height]
                    [[-10, -17, 0], [-10, 17, 0], 2, 1],
                    [[10, -17, 0], [10, 17, 0], 2, 1],
                ],
                z_offset=offsets.copper_sidebars_offset,
            ),
        ],
        bias_fields=ac.field.BiasFields(
            currents=[17.4, 44.3, 0.0],  # Currents applied to external coids [A]
            coil_factors=[-1.068, 1.8, 3.0],  # Current to Field Conversion [G/A]
            stray_fields=[3.5, -0.1, 0.0],  # Stray field offsets [G]
        ),
    )

    # fmt: off
    search_options = dict(
        x0      = [0.0, 0.0, 0.5],  # Initial guess
        bounds  = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)],
        method  = "Nelder-Mead",
        options = {
            "xatol"  : 1e-10,
            "fatol"  : 1e-10,
            "maxiter": int(1e5),
            "maxfev" : int(1e5),
            "disp"   : True,
        },
        hessian_step = 1e-5,  # Step size for Hessian calculation
        verbose = True,
    )
    # fmt: on

    atom_chip.search_field_minimum(search_options)
    atom_chip.search_potential_minimum(search_options)
    ac.visualization.show(atom_chip, "src/copper_z.yaml")


if __name__ == "__main__":
    main()

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

    import numpy as np

    points = np.array([[0.0, 0.0, 0.5]])
    # TODO
    # B_mag, B = atom_chip.get_fields(points)
    # print(f"Magnetic field at {points}: {B_mag} {B}")
    E_min = ac.search.search_minimum_potential(
        atom_chip.get_potentials,
        initial_guess=points,
        max_iterations=int(1e10),
        learning_rate=1e-2,
        tolerance=1e-10,
    )
    print("min_value", E_min.value)
    print("point", E_min.point)
    print("grads", E_min.grads)

    ac.visualization.show(atom_chip, E_min, "src/copper_z.yaml")


if __name__ == "__main__":
    main()

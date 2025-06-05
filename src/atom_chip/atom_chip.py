from typing import List, NamedTuple, Tuple
import logging
import jax
import pandas as pd
import json
import time
import jax.numpy as jnp
from .components import RectangularConductor, RectangularSegment
from .field import BiasFields, BiasFieldParams, get_bias_fields, biot_savart_rectangular, ZERO_BIAS_FIELD
from .potential import Atom, AnalysisOptions, FieldAnalysis, PotentialAnalysis
from .potential import trap_potential_energy, analyze_field, analyze_trap


class AtomChipWires(NamedTuple):
    starts: jnp.ndarray  # shape (n_wires, 3)
    ends: jnp.ndarray  # shape (n_wires, 3)
    widths: jnp.ndarray  # shape (n_wires,)
    heights: jnp.ndarray  # shape (n_wires,)


@jax.jit
def trap_magnetic_fields(
    points: jnp.ndarray, wires: AtomChipWires, currents: jnp.ndarray, bias: BiasFieldParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the magnetic field at given points in space.

    Returns:
        B_mag (jnp.ndarray): Magnitude of the magnetic field at the points.
        B (jnp.ndarray): Magnetic field vector at the points.
    """
    # Get the bias fields
    B = get_bias_fields(points, bias)
    B = B + biot_savart_rectangular(points, wires.starts, wires.ends, wires.widths, wires.heights, currents)
    B_mag = jnp.linalg.norm(B, axis=1)
    return B_mag, B


@jax.jit
def trap_potential_energies(
    points: jnp.ndarray, atom: Atom, wires: AtomChipWires, currents: jnp.ndarray, bias: BiasFieldParams
):
    """
    Compute the potential energy at given points in space.
    """
    B_mag, B = trap_magnetic_fields(points, wires, currents, bias)
    z = points[:, 2]
    return trap_potential_energy(atom, B_mag, z), B_mag, B


class AtomChipAnalysis(NamedTuple):
    field: FieldAnalysis
    potential: PotentialAnalysis


class AtomChip:
    """
    Represents an Atom Chip with atom, wires, and bias fields.
    """

    def __init__(
        self,
        name: str,
        atom: Atom,
        components: List[RectangularConductor],
        bias_fields: BiasFields,
    ):
        self.name = name
        self.atom = atom
        self.components = components
        self.bias_fields = bias_fields

    def get_fields(self, points: jnp.array) -> jnp.ndarray:
        points = jnp.atleast_2d(points).astype(jnp.float64)
        wires = atom_chip_components_to_wires(self.components)
        currents = atom_chip_components_to_currents(self.components)
        return trap_magnetic_fields(points, wires, currents, self.bias_fields.params)

    def get_potentials(self, points: jnp.array) -> jnp.ndarray:
        points = jnp.atleast_2d(points).astype(jnp.float64)
        wires = atom_chip_components_to_wires(self.components)
        currents = atom_chip_components_to_currents(self.components)
        return trap_potential_energies(points, self.atom, wires, currents, self.bias_fields.params)

    def analyze(self, options: AnalysisOptions) -> AtomChipAnalysis:
        logging.info(f"Bias fields: {self.bias_fields.to_dict()}")
        start_time = time.time()
        field_analysis = analyze_field(self.atom, self.get_fields, options)
        potential_analysis = analyze_trap(self.atom, self.get_potentials, options)
        end_time = time.time()
        logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
        return AtomChipAnalysis(
            field=field_analysis,
            potential=potential_analysis,
        )

    def save(self, path: str) -> str:
        """
        Save the AtomChip to a JSON file.
         - If a path is provided, the JSON is saved to that file.
         - Otherwise, the JSON string is returned.
        """
        data = self.to_json()
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        return json.dumps(data, indent=2)

    def load(self, path: str):
        """
        Load the AtomChip from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return self.from_json(data)

    def to_json(self) -> List[dict]:
        """
        Serialize the AtomChip to JSON format.
        """
        wires = []
        for component_id, component in enumerate(self.components):
            segment_id = 0
            for start, end, width, height in zip(
                component.starts.tolist(),
                component.ends.tolist(),
                component.widths.tolist(),
                component.heights.tolist(),
            ):
                wires.append(
                    {
                        "component_id": component_id,
                        "segment_id": segment_id,
                        "material": component.material,
                        "current": float(component.current),
                        "start": start,
                        "end": end,
                        "width": width,
                        "height": height,
                    }
                )
                segment_id += 1
        data = {
            "wires": wires,
            "bias_fields": self.bias_fields.to_dict(),
        }
        return data

    def from_json(self, data: List[dict]):
        """
        Load the AtomChip from a JSON data.

        The atom and bias fields are not loaded (they remain unchanged).
        Any analysis results are cleared.
        """

        # load and sort the data by component_id and segment_id
        wires = data["wires"]
        wires_df = pd.DataFrame(wires)
        wires_df = wires_df.sort_values(by=["component_id", "segment_id"]).reset_index(drop=True)

        # iterate over the components and create a list of RectangularConductor objects
        components = []
        for component_id, group in wires_df.groupby("component_id"):
            segments = []
            for _, row in group.iterrows():
                segments.append(
                    RectangularSegment(
                        start=row.start,
                        end=row.end,
                        width=row.width,
                        height=row.height,
                    )
                )
            # create a new component and add it to the list
            # fmt: off
            components.append(RectangularConductor.create(
                material = group.iloc[0]["material"],
                current  = group.iloc[0]["current"].astype(float),
                segments = segments,
            ))
            # fmt: on

        # load the bias fields
        if "bias_fields" in data:
            bias_fields = data["bias_fields"]
            # fmt: off
            bias_fields = BiasFields(
                coil_factors = jnp.array(bias_fields["coil_factors"], dtype=jnp.float64),
                currents     = jnp.array(bias_fields["currents"]    , dtype=jnp.float64),
                stray_fields = jnp.array(bias_fields["stray_fields"], dtype=jnp.float64),
            )
            # fmt: on
        else:
            bias_fields = ZERO_BIAS_FIELD

        return AtomChip(
            name=self.name,
            atom=self.atom,
            components=components,
            bias_fields=bias_fields,
        )


def atom_chip_components_to_wires(components: List[RectangularConductor]) -> AtomChipWires:
    # consolidate all the components into a single call to biot_savart_rectangular
    # to avoid multiple calls to JAX
    # fmt: off
    starts   = jnp.concatenate([component.starts   for component in components])
    ends     = jnp.concatenate([component.ends     for component in components])
    widths   = jnp.concatenate([component.widths   for component in components])
    heights  = jnp.concatenate([component.heights  for component in components])
    # fmt: on
    return AtomChipWires(starts, ends, widths, heights)


def atom_chip_components_to_currents(components: List[RectangularConductor]) -> jnp.ndarray:
    return jnp.concatenate([component.currents for component in components])

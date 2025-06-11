from typing import List, NamedTuple, Tuple
import logging
import jax
import pandas as pd
import json
import time
import jax.numpy as jnp
from .components import RectangularConductor, RectangularSegment
from .field import BiasConfig
from .potential import Atom, AnalysisOptions, FieldAnalysis, PotentialAnalysis
from . import field, potential


# fmt: off
class WireConfig(NamedTuple):
    starts : jnp.ndarray  # shape (n_wires, 3)
    ends   : jnp.ndarray  # shape (n_wires, 3)
    widths : jnp.ndarray  # shape (n_wires,)
    heights: jnp.ndarray  # shape (n_wires,)
# fmt: on


@jax.jit
def trap_magnetic_fields(
    points: jnp.ndarray, wires: WireConfig, currents: jnp.ndarray, bias: BiasConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the magnetic field at given points in space.

    Returns:
        B_mag (jnp.ndarray): Magnitude of the magnetic field at the points.
        B (jnp.ndarray): Magnetic field vector at the points.
    """
    # Get the bias fields
    B = field.get_bias_fields(points, bias)
    B = B + field.biot_savart_rectangular(points, wires.starts, wires.ends, wires.widths, wires.heights, currents)
    B_mag = jnp.linalg.norm(B, axis=1)
    return B_mag, B


@jax.jit
def trap_potential_energies(
    points: jnp.ndarray, atom: Atom, wires: WireConfig, currents: jnp.ndarray, bias: BiasConfig
):
    """
    Compute the potential energy at given points in space.
    """
    B_mag, B = trap_magnetic_fields(points, wires, currents, bias)
    z = points[:, 2]
    return potential.trap_potential_energy(atom, B_mag, z), B_mag, B


# This is not a JIT function
def analyze_field(
    atom: Atom,
    wires: WireConfig,
    currents: jnp.ndarray,
    bias: BiasConfig,
    options: AnalysisOptions,
) -> FieldAnalysis:
    """
    Analyze the trap magnetic field at give points in space.
    """
    return potential.analyze_field(
        atom,
        lambda p: trap_magnetic_fields(jnp.atleast_2d(p), wires, currents, bias),
        options=options,
    )


# This is not a JIT function
def analyze_trap(
    atom: Atom,
    wires: WireConfig,
    currents: jnp.ndarray,
    bias: BiasConfig,
    options: AnalysisOptions,
) -> PotentialAnalysis:
    """
    Analyze the trap potential energy at given points in space.
    """
    return potential.analyze_trap(
        atom,
        lambda p: trap_potential_energies(jnp.atleast_2d(p), atom, wires, currents, bias),
        options=options,
    )


class AtomChipAnalysis(NamedTuple):
    field: FieldAnalysis
    potential: PotentialAnalysis


def analyze_all(
    atom: Atom,
    wires: WireConfig,
    currents: jnp.ndarray,
    bias_config: BiasConfig,
    options: AnalysisOptions,
) -> AtomChipAnalysis:
    """
    Analyze the trap magnetic field and potential energy at given points in space.
    """
    logging.info(f"Bias fields: {field.bias_config_to_dict(bias_config)}")
    start_time = time.time()
    field_analysis = analyze_field(atom, wires, currents, bias_config, options)
    potential_analysis = analyze_trap(atom, wires, currents, bias_config, options)
    end_time = time.time()
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    return AtomChipAnalysis(field=field_analysis, potential=potential_analysis)


class AtomChip:
    """
    Represents an Atom Chip with atom, wires, and bias fields.
    """

    def __init__(
        self,
        name: str,
        atom: Atom,
        components: List[RectangularConductor],
        bias_config: BiasConfig,
    ):
        self.name = name
        self.atom = atom
        self.components = components
        self.bias_config = bias_config
        self.wires = atom_chip_components_to_wires(self.components)
        self.currents = atom_chip_components_to_currents(self.components)

    def get_fields(self, points: jnp.array) -> jnp.ndarray:
        points = jnp.atleast_2d(points).astype(jnp.float64)
        return trap_magnetic_fields(points, self.wires, self.currents, self.bias_config)

    def get_potentials(self, points: jnp.array) -> jnp.ndarray:
        points = jnp.atleast_2d(points).astype(jnp.float64)
        return trap_potential_energies(points, self.atom, self.wires, self.currents, self.bias_config)

    def analyze(self, options: AnalysisOptions) -> AtomChipAnalysis:
        return analyze_all(self.atom, self.wires, self.currents, self.bias_config, options)

    def save(self, path: str):
        """
        Save the AtomChip to a JSON file.
         - If a path is provided, the JSON is saved to that file.
         - Otherwise, the JSON string is returned.
        """
        save_atom_chip(self, path)


def atom_chip_components_to_wires(components: List[RectangularConductor]) -> WireConfig:
    # consolidate all the components into a single call to biot_savart_rectangular
    # to avoid multiple calls to JAX
    # fmt: off
    starts   = jnp.concatenate([component.starts   for component in components])
    ends     = jnp.concatenate([component.ends     for component in components])
    widths   = jnp.concatenate([component.widths   for component in components])
    heights  = jnp.concatenate([component.heights  for component in components])
    # fmt: on
    return WireConfig(starts, ends, widths, heights)


def atom_chip_components_to_currents(components: List[RectangularConductor]) -> jnp.ndarray:
    return jnp.concatenate([component.currents for component in components])


def save_atom_chip(atom_chip: AtomChip, path: str):
    """
    Save the AtomChip to a JSON file.
        - If a path is provided, the JSON is saved to that file.
        - Otherwise, the JSON string is returned.
    """
    data = atom_chip_to_json(atom_chip)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def atom_chip_to_json(atom_chip: AtomChip) -> List[dict]:
    """
    Serialize the AtomChip to JSON format.
    """
    wires = []
    for component_id, component in enumerate(atom_chip.components):
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
        "bias_fields": field.bias_config_to_dict(atom_chip.bias_config),
    }
    return data


def load_atom_chip(name: str, atom: Atom, path: str):
    """
    Load the AtomChip from a JSON file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return json_to_atom_chip(name, atom, data)


def json_to_atom_chip(name: str, atom: Atom, data: List[dict]):
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
        bias_config = field.dict_to_bias_config(data["bias_fields"])
    else:
        bias_config = field.ZERO_BIAS_CONFIG

    return AtomChip(
        name=name,
        atom=atom,
        components=components,
        bias_config=bias_config,
    )

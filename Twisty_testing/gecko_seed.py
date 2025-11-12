"""Build deterministic gecko probability tensors and preview them in MuJoCo."""

from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np
import mujoco
from mujoco import viewer
from rich.console import Console

from robot_body.body_config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleRotationsTheta,
    ModuleType,
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from robot_body.hi_prob_decoding import HighProbabilityDecoder
from robot_body.prebuilt.gecko import gecko as prebuilt_gecko
from robot_body.modules.brick import BrickModule
from robot_body.modules.core import CoreModule as LocalCoreModule
from robot_body.modules.hinge import HingeModule
from simulation.environments._simple_flat import SimpleFlatWorld

console = Console()

# Keep tensors compatible with the rest of the pipeline.
DEFAULT_NUM_SLOTS = 20

# Module definitions lifted from robot_body/prebuilt/gecko.py
GECKO_TYPES: dict[int, ModuleType] = {
    0: ModuleType.CORE,
    1: ModuleType.HINGE,
    2: ModuleType.BRICK,
    3: ModuleType.HINGE,
    4: ModuleType.BRICK,
    5: ModuleType.HINGE,
    6: ModuleType.BRICK,
    7: ModuleType.HINGE,
    8: ModuleType.BRICK,
    9: ModuleType.HINGE,
    10: ModuleType.BRICK,
    11: ModuleType.HINGE,
    12: ModuleType.BRICK,
    15: ModuleType.HINGE,
    17: ModuleType.HINGE,
}

GECKO_ROTATIONS: dict[int, ModuleRotationsIdx] = {
    5: ModuleRotationsIdx.DEG_90,
    15: ModuleRotationsIdx.DEG_90,
    7: ModuleRotationsIdx.DEG_270,
    17: ModuleRotationsIdx.DEG_90,
    9: ModuleRotationsIdx.DEG_45,
    11: ModuleRotationsIdx.DEG_315,
}

GECKO_EDGES: list[tuple[int, int, ModuleFaces]] = [
    (0, 1, ModuleFaces.FRONT),
    (1, 2, ModuleFaces.FRONT),
    (2, 3, ModuleFaces.FRONT),
    (3, 4, ModuleFaces.FRONT),
    (0, 5, ModuleFaces.LEFT),
    (5, 15, ModuleFaces.FRONT),
    (15, 6, ModuleFaces.FRONT),
    (0, 7, ModuleFaces.RIGHT),
    (7, 17, ModuleFaces.FRONT),
    (17, 8, ModuleFaces.FRONT),
    (4, 9, ModuleFaces.LEFT),
    (9, 10, ModuleFaces.FRONT),
    (4, 11, ModuleFaces.RIGHT),
    (11, 12, ModuleFaces.FRONT),
]


class GeckoProbabilities(NamedTuple):
    module: np.ndarray
    connection: np.ndarray
    rotation: np.ndarray


def gecko_graph() -> nx.DiGraph:
    """Return the NetworkX graph describing the reference gecko."""
    graph = nx.DiGraph()
    for node, module_type in GECKO_TYPES.items():
        rotation = GECKO_ROTATIONS.get(node, ModuleRotationsIdx.DEG_0)
        graph.add_node(
            node,
            type=module_type.name,
            rotation=rotation.name,
        )
    for parent, child, face in GECKO_EDGES:
        graph.add_edge(parent, child, face=face.name)
    return graph


def gecko_probability_spaces(
    slots: int = DEFAULT_NUM_SLOTS,
) -> GeckoProbabilities:
    """Return one-hot probability tensors that decode back into the gecko."""
    graph = gecko_graph()
    highest_index = max(graph.nodes)
    if slots <= highest_index:
        msg = f"Need at least {highest_index + 1} slots, got {slots}."
        raise ValueError(msg)

    type_probs = np.zeros(
        (slots, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )
    type_probs[:, ModuleType.NONE.value] = 1.0

    conn_probs = np.zeros(
        (slots, slots, NUM_OF_FACES),
        dtype=np.float32,
    )

    rot_probs = np.zeros(
        (slots, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )
    rot_probs[:, ModuleRotationsIdx.DEG_0.value] = 1.0

    for node, data in graph.nodes(data=True):
        module_type = ModuleType[data["type"]]
        rotation = ModuleRotationsIdx[data["rotation"]]

        type_probs[node, :] = 0.0
        type_probs[node, module_type.value] = 1.0

        rot_probs[node, :] = 0.0
        rot_probs[node, rotation.value] = 1.0

    for parent, child, data in graph.edges(data=True):
        face = ModuleFaces[data["face"]]
        conn_probs[parent, child, face.value] = 1.0

    return GeckoProbabilities(type_probs, conn_probs, rot_probs)


def decode_seed_graph(
    slots: int = DEFAULT_NUM_SLOTS,
) -> nx.DiGraph:
    """Use the HighProbabilityDecoder to rebuild the gecko graph."""
    type_probs, conn_probs, rot_probs = gecko_probability_spaces(slots)
    hpd = HighProbabilityDecoder(slots)
    return hpd.probability_matrices_to_graph(
        type_probs.copy(),
        conn_probs.copy(),
        rot_probs.copy(),
    )


def _graph_to_local_core_module(graph: nx.DiGraph) -> LocalCoreModule:
    """Convert a graph into a core module using the same modules as the prebuilt gecko."""
    modules: dict[int, LocalCoreModule | BrickModule | HingeModule | None] = {}

    for node, data in graph.nodes(data=True):
        module_type = ModuleType[data["type"]]
        rotation_idx = ModuleRotationsIdx[data["rotation"]]

        module = None
        match module_type:
            case ModuleType.CORE:
                module = LocalCoreModule(index=node)
            case ModuleType.BRICK:
                module = BrickModule(index=node)
            case ModuleType.HINGE:
                module = HingeModule(index=node)
            case ModuleType.NONE:
                module = None

        modules[node] = module
        if module:
            angle = ModuleRotationsTheta[rotation_idx.name].value
            module.rotate(angle)

    for parent, child, data in graph.edges(data=True):
        face_enum = ModuleFaces[data["face"]]
        site_key = str(face_enum)
        parent_module = modules[parent]
        child_module = modules[child]
        if parent_module is None or child_module is None:
            continue
        parent_module.sites[site_key].attach_body(
            body=child_module.body,
            prefix=f"{parent_module.index}-{child_module.index}-{face_enum.value}-",
        )

    core_module = modules.get(0)
    if not isinstance(core_module, LocalCoreModule):
        msg = "Decoded graph does not contain a valid core module."
        raise ValueError(msg)
    return core_module


def _graphs_match(reference: nx.DiGraph, decoded: nx.DiGraph) -> bool:
    """Check whether two graphs share identical nodes, edges, and attributes."""

    def node_key(graph: nx.DiGraph) -> set[tuple[int, str, str]]:
        return {
            (node, attrs["type"], attrs["rotation"])
            for node, attrs in graph.nodes(data=True)
        }

    def edge_key(graph: nx.DiGraph) -> set[tuple[int, int, str]]:
        return {
            (parent, child, attrs["face"])
            for parent, child, attrs in graph.edges(data=True)
        }

    ref_nodes = node_key(reference)
    dec_nodes = node_key(decoded)
    ref_edges = edge_key(reference)
    dec_edges = edge_key(decoded)

    missing_nodes = ref_nodes - dec_nodes
    extra_nodes = dec_nodes - ref_nodes
    missing_edges = ref_edges - dec_edges
    extra_edges = dec_edges - ref_edges

    if missing_nodes or extra_nodes or missing_edges or extra_edges:
        if missing_nodes:
            console.log("[red]Missing nodes:", missing_nodes)
        if extra_nodes:
            console.log("[yellow]Unexpected nodes:", extra_nodes)
        if missing_edges:
            console.log("[red]Missing edges:", missing_edges)
        if extra_edges:
            console.log("[yellow]Unexpected edges:", extra_edges)
        return False

    return True


def _build_world_and_launch(robot_title: str, core_module) -> None:
    """Attach a robot to the flat world and launch the MuJoCo viewer."""
    console.log(f"Launching `{robot_title}` — close the window to continue.")
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(
        core_module.spec,
        spawn_xy=[0.0, 0.0],
        spawn_z=0.3,
        correct_for_bounding_box=True,
    )
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    viewer.launch(model, data)


def demo_viewer() -> None:
    """Show the original and decoded geckos back-to-back in the viewer."""
    original_robot = prebuilt_gecko()
    _build_world_and_launch("Prebuilt Gecko", original_robot)

    decoded_graph = decode_seed_graph()
    reference_graph = gecko_graph()

    if _graphs_match(reference_graph, decoded_graph):
        console.log("[green]Decoded graph matches the reference gecko graph.")
    else:
        console.log("[red]Decoded graph does not match reference graph!")

    decoded_robot = _graph_to_local_core_module(decoded_graph)
    _build_world_and_launch("Decoded Gecko From Probabilities", decoded_robot)


if __name__ == "__main__":
    demo_viewer()

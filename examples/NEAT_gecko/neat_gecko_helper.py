from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any, Callable

import neat
import numpy as np
import numpy.typing as npt


def fitness_function_basic(history: list[npt.NDArray[np.float64]]) -> float:
    """Compute forward progress along the Y axis.

    Returns positive distance traveled forward (end_y - start_y).
    """
    if not history:
        return 0.0
    ys = float(history[0][1])
    ye = float(history[-1][1])
    return ys - ye


def create_file_name(
    experiment_name: str,
    robot: Callable,
    optimizer_name: str,
    generations: int,
    pop_size: int,
) -> str:
    """Construct a descriptive file name for artifacts.

    Components: experiment, robot, optimizer, generations, pop size, timestamp.
    """
    name_components: list[str] = []
    name_components.append(experiment_name)

    robot_name = robot.__name__
    name_components.append(robot_name)

    name_components.append(optimizer_name.lower())

    name_components.append(f"g{generations}")
    name_components.append(f"pop{pop_size}")

    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    name_components.append(timestamp)

    return "_".join(name_components)


def save_network(genome: neat.DefaultGenome, config: neat.Config) -> dict[str, Any]:
    """Save a NEAT genome's architecture and weights for inheritance.

    Extracts the complete network structure including:
    - Node genes (hidden and output nodes with their properties)
    - Connection genes (weights, enabled status)
    - Metadata (original input/output sizes for reference)

    Args:
        genome: NEAT genome to save
        config: NEAT configuration object

    Returns:
        Dictionary containing network structure that can be serialized
    """
    genome_config = config.genome_config

    # Save metadata
    saved_data = {
        "original_input_size": genome_config.num_inputs,
        "original_output_size": genome_config.num_outputs,
        "nodes": [],
        "connections": [],
    }

    # Save node genes (excluding input nodes, which are defined by structure)
    for node_id, node_gene in genome.nodes.items():
        node_data = {
            "id": node_id,
            "activation": node_gene.activation,
            "aggregation": node_gene.aggregation,
            "bias": node_gene.bias,
            "response": node_gene.response,
        }
        saved_data["nodes"].append(node_data)

    # Save connection genes
    for conn_key, conn_gene in genome.connections.items():
        conn_data = {
            "in_node": conn_key[0],
            "out_node": conn_key[1],
            "weight": conn_gene.weight,
            "enabled": conn_gene.enabled,
        }
        saved_data["connections"].append(conn_data)

    return saved_data


def load_network(
    saved_data: dict[str, Any],
    new_input_size: int,
    new_output_size: int,
    config: neat.Config,
    rng: np.random.Generator | None = None,
) -> neat.DefaultGenome:
    """Load a saved NEAT genome and adapt it to new input/output sizes.

    Creates a new genome with the specified input/output dimensions while
    preserving as much of the parent network structure as possible:
    - Hidden nodes are transferred directly with their weights
    - Connections between hidden nodes are preserved
    - Connections involving inputs/outputs are mapped when possible:
      * Input connections: map to corresponding new inputs (by index)
      * Output connections: map to corresponding new outputs (by index)
      * Unmappable connections are dropped
    - New inputs/outputs without connections get random initialization

    Args:
        saved_data: Dictionary from save_network()
        new_input_size: Number of inputs for new morphology
        new_output_size: Number of outputs for new morphology
        config: NEAT configuration object (with updated input/output sizes)
        rng: Random number generator for new connections (optional)

    Returns:
        New genome adapted to the new input/output sizes
    """
    if rng is None:
        rng = np.random.default_rng()

    genome_config = config.genome_config
    old_input_size = saved_data["original_input_size"]
    old_output_size = saved_data["original_output_size"]

    # Create input/output key mappings
    # NEAT uses negative keys for inputs: [-1, -2, ..., -num_inputs]
    # and positive keys for outputs: [0, 1, ..., num_outputs-1]
    old_input_keys = [-i for i in range(1, old_input_size + 1)]
    new_input_keys = [-i for i in range(1, new_input_size + 1)]
    old_output_keys = list(range(old_output_size))
    new_output_keys = list(range(new_output_size))

    # Create mapping: old key -> new key (where applicable)
    input_mapping = {}
    for i, old_key in enumerate(old_input_keys):
        if i < len(new_input_keys):  # Map first N inputs
            input_mapping[old_key] = new_input_keys[i]

    output_mapping = {}
    for i, old_key in enumerate(old_output_keys):
        if i < len(new_output_keys):  # Map first N outputs
            output_mapping[old_key] = new_output_keys[i]

    # Create new genome
    genome_id = 0  # Will be set properly by the population
    new_genome = neat.DefaultGenome(genome_id)
    new_genome.configure_new(config.genome_config)

    # Transfer node genes (hidden and output nodes)
    # First, clear default nodes and rebuild with parent structure
    new_genome.nodes.clear()

    # Add output nodes with new keys
    for new_out_key in new_output_keys:
        # Find corresponding old output node if it exists
        old_index = new_output_keys.index(new_out_key)
        matching_old_node = None

        if old_index < len(old_output_keys):
            old_out_key = old_output_keys[old_index]
            # Find this node in saved data
            for node_data in saved_data["nodes"]:
                if node_data["id"] == old_out_key:
                    matching_old_node = node_data
                    break

        # Create node gene
        node_gene = genome_config.node_gene_type(new_out_key)
        if matching_old_node:
            # Copy properties from parent
            node_gene.activation = matching_old_node["activation"]
            node_gene.aggregation = matching_old_node["aggregation"]
            node_gene.bias = matching_old_node["bias"]
            node_gene.response = matching_old_node["response"]
        else:
            # New output node - initialize randomly
            node_gene.init_attributes(config.genome_config)

        new_genome.nodes[new_out_key] = node_gene

    # Add hidden nodes (IDs typically > max(output_keys))
    for node_data in saved_data["nodes"]:
        node_id = node_data["id"]
        # Skip output nodes (already handled above)
        if node_id in old_output_keys:
            continue

        # This is a hidden node - transfer it directly
        node_gene = genome_config.node_gene_type(node_id)
        node_gene.activation = node_data["activation"]
        node_gene.aggregation = node_data["aggregation"]
        node_gene.bias = node_data["bias"]
        node_gene.response = node_data["response"]
        new_genome.nodes[node_id] = node_gene

    # Transfer connection genes with input/output mapping
    for conn_data in saved_data["connections"]:
        old_in = conn_data["in_node"]
        old_out = conn_data["out_node"]

        # Map input node if applicable
        if old_in in old_input_keys:
            if old_in not in input_mapping:
                continue  # Skip unmappable input connection
            new_in = input_mapping[old_in]
        else:
            new_in = old_in  # Hidden node - keep same ID

        # Map output node if applicable
        if old_out in old_output_keys:
            if old_out not in output_mapping:
                continue  # Skip unmappable output connection
            new_out = output_mapping[old_out]
        else:
            new_out = old_out  # Hidden node - keep same ID

        # Create connection gene
        conn_key = (new_in, new_out)
        conn_gene = genome_config.conn_gene_type(conn_key)
        conn_gene.weight = conn_data["weight"]
        conn_gene.enabled = conn_data["enabled"]
        new_genome.connections[conn_key] = conn_gene

    return new_genome


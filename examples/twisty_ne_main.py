"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-07-08
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     In progress ⚙️

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ] documentation

"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco
import numpy as np
from rich.console import Console
from mujoco import viewer

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (NUM_OF_FACES,
                                                       NUM_OF_ROTATIONS,
                                                       NUM_OF_TYPES_OF_MODULES)
from ariel.body_phenotypes.robogen_lite.constructor import \
    construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder, save_graph_as_json)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.cpg_with_sensory_feedback import \
    CPGSensoryFeedback
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
# from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec.a001 import Individual
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner
from pydantic_settings import BaseSettings
from twisty_src.neuroevolution import NeuroEvolution

# DEBUG
import copy

if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# twisty indexes (rotations of 45, 135, 225, 315 degrees) 
TWIST_I = [1, 3, 5, 7]

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

# Global variables
SPAWN_POS = [0, 0, 0]
NUM_OF_MODULES = 30
# TARGET_POSITION = [5, 0, 0.5]

class EASettings(BaseSettings):

    # world: SimpleFlatWorld = SimpleFlatWorld()
    starting_pos: list[float] = [0, 0, 0]

    # Robot evolution parameters
    population_size: int = 1
    num_of_generations: int = 1

    # Neuroevolution parameters
    population_size_brains: int = 2
    num_of_generations_brains: int = 3
    mutation_rate_brains: float = 0
    mutation_magnitude_brains: float = 0.5
    mutation_scale_brains: float = 5.0
    
    
    nn_hidden_layers: list[int] = [64, 32]
    
    is_maximisation: bool = True

config = EASettings()

def fitness_function_basic(history: list[float]) -> float:
    xs, _, _ = config.starting_pos
    xe, _, _ = history[-1]

    # maximize the distance
    x_distance = xs - xe
    return x_distance

def create_individual(con_twisty: bool) -> Individual:
    ind = Individual()
    num_modules = 20

    # "Type" probability space
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    if con_twisty is True:   
        rotation_probability_space = RNG.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
    else:
        # if twisty not true, the twisted angles are given the value zero, so they can't be selected.
        rotation_probability_space = RNG.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
        for i in rotation_probability_space:
            i[TWIST_I] = [0]*4

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )
    # Save the graph to a file
    save_graph_as_json(
        graph,
        DATA / "graph.json",
    )
    ind.genotype = graph
    ind.twisty = con_twisty
    return ind

def main() -> None:
    """Entry point for neuroevolution experiment - evolves brains for a population of robots."""
    console.log("[bold cyan]Starting Neuroevolution Experiment")
    
    # Configuration
    population_size = config.population_size
    
    # Initialize population of robot morphologies
    console.log(f"[yellow]Creating population of {population_size} robot morphologies...")
    initial_population = []
    for _ in range(population_size):
        # twisty is false, no 45 degrees angles
        initial_population.append(create_individual(False))
    for _ in range(population_size):
        # twisty is True
        initial_population.append(create_individual(True))
    # Print all nodes
    for ind in initial_population:
        # core = construct_mjspec_from_graph(ind.genotype)
        ne = NeuroEvolution(
            fitness_function=fitness_function_basic,
            nn_hidden_layers=config.nn_hidden_layers,
            population_size=config.population_size_brains,
            num_of_generations=config.num_of_generations_brains,
            mutation_rate=config.mutation_rate_brains,
            mutation_magnitude=config.mutation_magnitude_brains,
            mutation_scale=config.mutation_scale_brains,
            starting_pos=config.starting_pos,
            is_maximisation=config.is_maximisation,
        )
        brain = ne.evolve(ind)
        ind.brain_genotype = brain
        # Simulate the robot
        # run(core, ind)

if __name__ == "__main__":
    # Test several times
    main()
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
# from typing import TYPE_CHECKING, Any

# Third-party libraries
import numpy as np
from rich.console import Console

# Local libraries
# from ariel.body_phenotypes.robogen_lite.config import (NUM_OF_FACES,
#                                                        NUM_OF_ROTATIONS,
#                                                        NUM_OF_TYPES_OF_MODULES)
# from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
#     HighProbabilityDecoder, save_graph_as_json)
from ariel.ec.a001 import Individual
from pydantic_settings import BaseSettings
from twisty_src.neuroevolution import NeuroEvolution

# if TYPE_CHECKING:
#     from networkx import DiGraph

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
# NUM_OF_MODULES = 30
# TARGET_POSITION = [5, 0, 0.5]

class EASettings(BaseSettings):

    # world: SimpleFlatWorld = SimpleFlatWorld()
    starting_pos: list[float] = [0, 0, 0]

    # Robot evolution parameters
    population_size: int = 1
    max_evals: int = 100
    num_of_generations: int = 1

    # Neuroevolution parameters
    population_size_brains: int = 50
    tournament_size_brains: int = 5
    # num_of_generations_brains: int = 10
    max_evals_brains: int = 1000
    mutation_rate_brains: float = 0.05
    mutation_magnitude_brains: float = 0.1
    mutation_scale_brains: float = 1
    
    
    nn_hidden_layers: list[int] = [64, 32]
    
    is_maximisation: bool = True

config = EASettings()

def fitness_function_basic(starting_pos, history: list[float]) -> float:
    # move forward, direction got trail and errored
    
    _, ys, _ = starting_pos
    _, ye, _ = history[-1]

    # maximize the distance
    y_distance = ys - ye # negative y distance = forward movement
    return y_distance

def create_gecko_individual() -> Individual:
    ind = Individual()
    ind.genotype = "gecko or smth, idk"
    ind.twisty = True
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
        initial_population.append(create_gecko_individual())

    for ind in initial_population:
        # core = construct_mjspec_from_graph(ind.genotype)
        ne = NeuroEvolution(
            fitness_function=fitness_function_basic,
            nn_hidden_layers=config.nn_hidden_layers,
            population_size=config.population_size_brains,
            # num_of_generations=config.num_of_generations_brains,
            max_evals=config.max_evals_brains,
            mutation_rate=config.mutation_rate_brains,
            mutation_magnitude=config.mutation_magnitude_brains,
            mutation_scale=config.mutation_scale_brains,
            starting_pos=config.starting_pos,
            is_maximisation=config.is_maximisation,
            tournament_size=config.tournament_size_brains,
        )
        brain = ne.evolve(None, use_gecko=True, verbose=True)
        ind.brain_genotype = brain

if __name__ == "__main__":
    # Test several times
    main()
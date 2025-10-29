from ea_components.individual import Individual
import numpy as np
from typing import Any

SEED = 40
RNG = np.random.default_rng(SEED)

def tournament_selection(population: list[Any], tournament_size: int = 3) -> Individual:
    """Tournament selection."""
    tournament = RNG.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda ind: ind.fitness)
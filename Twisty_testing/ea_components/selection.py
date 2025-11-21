from ea_components.individual import Individual
import numpy as np
from typing import Any

SEED = 40
RNG = np.random.default_rng(SEED)

def tournament_selection(population: list[Any], tournament_size: int = 3) -> Individual:
    """Tournament selection."""
    tournament = RNG.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda ind: ind.fitness)

def exponantial_rank_selection(population: list[Any]) -> Individual:
    ranked = sorted(population, key=lambda ind: ind.fitness)
    selection_probs = []
    for i in range(len(ranked)):
        prob = (1 - np.exp(-i))/ len(ranked)
        selection_probs.append(prob)
    rng = RNG.uniform(0, sum(selection_probs))
    total = 0
    i = 0
    for prob in selection_probs:
        total += prob
        if rng <= total:
            break
        i += 1
    return population[-(i+1)]

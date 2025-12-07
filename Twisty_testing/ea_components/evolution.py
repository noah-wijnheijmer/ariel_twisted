from ea_components.individual import Individual
from ea_components.selection import exponantial_rank_selection
from ea_components.reproduction import crossover_individuals, mutate_individual
import numpy as np


def evolve_generation(population: list[Individual],
                                    rng, 
                                    id: int = -1, 
                                    mutation_rate: float = 0.1,
                                    crossover_rate: float = 0.7,
                                    elitism: int = 1) -> list[Individual]:
    """Evolve one generation with tournament selection and elitism.
    
    Args:
        population: Current population to evolve
        mutation_rate: Probability of mutation for each matrix type
        crossover_rate: Probability of using crossover vs mutation-only
        elitism: Number of best individuals to keep unchanged
        
    Returns:
        New evolved population
    """
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    # Next generation
    offsprings = []
    # Elitism - keep best individual(s)
    offsprings.extend(population[:elitism])
    ids = []
    for individual in population:
        ids.append(individual.id)
    i = max(ids)+1
    if id != -1:
        i = id
    # Generate offspring through crossover and mutation
    while len(offsprings) < 3 * len(population):
        if rng.random() < crossover_rate:
            # Crossover: select two parents and create offspring
            parent1 = exponantial_rank_selection(population, rng)
            parent2 = exponantial_rank_selection(population, rng)
            child = crossover_individuals(parent1, parent2, rng, id=i)
            
            # Apply mutation to crossover offspring (optional but recommended)
            if rng.random() < mutation_rate:
                child = mutate_individual(individual=child, rng=rng, id=i, mutation_rate=mutation_rate)  # Lower mutation rate for crossover offspring
        else:
            # Mutation only: select one parent and mutate
            parent = exponantial_rank_selection(population, rng)
            child = mutate_individual(individual=parent, rng=rng, id=i, mutation_rate=mutation_rate )
        child.id = i
        offsprings.append(child)
        i += 1
    return offsprings
from ea_components.individual import Individual
from ea_components.selection import tournament_selection
from ea_components.reproduction import crossover_individuals, mutate_individual
import numpy as np

SEED = 40
RNG = np.random.default_rng(SEED)

def evolve_generation(population: list[Individual],
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
    # Sort by fitness for elitism
    fitnesses = []
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    for ind in population:
        fitnesses.append(ind.fitness)
    print(max(fitnesses))
    # Next generation
    new_population = []
    # Elitism - keep best individual(s)
    new_population.extend(population[:elitism])
    ids = []
    for individual in population:
        ids.append(individual.id)
    i = max(ids)+1
    if id != -1:
        i = id
    # Generate offspring through crossover and mutation
    while len(new_population) < len(population):
        if RNG.random() < crossover_rate:
            # Crossover: select two parents and create offspring
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover_individuals(parent1, parent2, id=i)
            
            # Apply mutation to crossover offspring (optional but recommended)
            if RNG.random() < mutation_rate:
                child = mutate_individual(individual=child, id=i, mutation_rate=mutation_rate)  # Lower mutation rate for crossover offspring
        else:
            # Mutation only: select one parent and mutate
            parent = tournament_selection(population)
            child = mutate_individual(individual=parent, id=i, mutation_rate=mutation_rate )
        child.id = i
        new_population.append(child)
        i += 1
    fitnesses = []
    return new_population
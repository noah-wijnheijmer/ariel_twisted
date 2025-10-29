from ea_components.individual import Individual
from ea_components.selection import tournament_selection
from ea_components.reproduction import crossover_individuals, mutate_individual
import numpy as np

SEED = 40
RNG = np.random.default_rng(SEED)

def evolve_generation(population: list[Individual], 
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
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Next generation
    new_population = []
    
    # Elitism - keep best individual(s)
    new_population.extend(population[:elitism])
    
    # Generate offspring through crossover and mutation
    while len(new_population) < len(population):
        if RNG.random() < crossover_rate:
            # Crossover: select two parents and create offspring
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover_individuals(parent1, parent2)
            
            # Apply mutation to crossover offspring (optional but recommended)
            if RNG.random() < mutation_rate:
                child = mutate_individual(child, mutation_rate)  # Lower mutation rate for crossover offspring
        else:
            # Mutation only: select one parent and mutate
            parent = tournament_selection(population)
            child = mutate_individual(parent, mutation_rate)
        
        new_population.append(child)
    
    return new_population
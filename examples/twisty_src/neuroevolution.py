"""TODO(jmdm): description of script.

Author:     Fullitu
Date:       2025-10-04
Py Ver:     3.12
Status:     In progress ⚙️

Notes
-----
    For now everything is copied from a004, exact implementation needed. 
    Look into CGA approach (papers on the WA group)

References
----------
    

Todo
----
    [ ] documentation
    [ ] implement crossover
    [ ] implement mutation
    [ ] implement evaluation (import from main twisty script?)
    [ ] implement survivor selection
    [ ] implement main evolve() method

"""

import random
from rich.console import Console

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EAStep, EASettings, Population
from twisty_brain import RobotBrain

console = Console()

class NeuroEvolution:
    def __init__(
            self,
            population_size: int = 10,
            num_generations: int = 100,
            nn_input_size: int = 10,
            nn_output_size: int = 4,
            nn_hidden_layers: list[int] | None = 3,
            ):
        self.config = EASettings()

    def create_individual(self) -> RobotBrain:
        return RobotBrain(
            input_size=self.config.nn_input_size,
            output_size=self.config.nn_output_size,
            hidden_layers=self.config.nn_hidden_layers
        )

    def parent_selection(self, population: Population) -> Population:
        random.shuffle(population)
        for idx in range(0, len(population) - 1, 2):
            ind_i = population[idx]
            ind_j = population[idx + 1]

            # Compare fitness values
            if ind_i.fitness > ind_j.fitness and self.config.is_maximisation:
                ind_i.tags = {"ps": True}
                ind_j.tags = {"ps": False}
            else:
                ind_i.tags = {"ps": False}
                ind_j.tags = {"ps": True}
        return population


    def crossover(self, population: Population) -> Population:
        # parents = [ind for ind in population if ind.tags.get("ps", False)]
        # for idx in range(0, len(parents), 2):
        #     parent_i = parents[idx]
        #     parent_j = parents[idx]
        #     genotype_i, genotype_j = Crossover.one_point(
        #         cast("list[int]", parent_i.genotype),
        #         cast("list[int]", parent_j.genotype),
        #     )

        #     # First child
        #     child_i = Individual()
        #     child_i.genotype = genotype_i
        #     child_i.tags = {"mut": True}
        #     child_i.requires_eval = True

        #     # Second child
        #     child_j = Individual()
        #     child_j.genotype = genotype_j
        #     child_j.tags = {"mut": True}
        #     child_j.requires_eval = True

        #     population.extend([child_i, child_j])
        # return population
        pass

    def mutation(self, population: Population) -> Population:
        # for ind in population:
        #     if ind.tags.get("mut", False):
        #         genes = cast("list[int]", ind.genotype)
        #         mutated = IntegerMutator.integer_creep(
        #             individual=genes,
        #             span=1,
        #             mutation_probability=0.5,
        #         )
        #         ind.genotype = mutated
        #         ind.requires_eval = True
        # return population
        pass

    def evaluate(self, population: Population) -> Population:
        # for ind in population:
        #     if ind.requires_eval:
        #         # Count ones in genotype as fitness
        #         ind.fitness = sum(1 for gene in ind.genotype if gene == 1)
        # return population
        pass

    def survivor_selection(self, population: Population) -> Population:
        # random.shuffle(population)
        # current_pop_size = len(population)
        # for idx in range(len(population)):
        #     ind_i = population[idx]
        #     ind_j = population[idx + 1]

        #     # Kill worse individual
        #     if ind_i.fitness > ind_j.fitness and self.config.is_maximisation:
        #         ind_j.alive = False
        #     else:
        #         ind_i.alive = False

        #     # Termination condition
        #     current_pop_size -= 1
        #     if current_pop_size <= self.config.target_population_size:
        #         break
        # return population
        pass

    def evolve(self) -> None:
        """Entry point."""
        # # Create initial population
        # population_list = [self.create_individual() for _ in range(10)]
        # population_list = self.evaluate(population_list)

        # # Create EA steps
        # ops = [
        #     EAStep("parent_selection", self.parent_selection),
        #     EAStep("crossover", self.crossover),
        #     EAStep("mutation", self.mutation),
        #     EAStep("evaluation", self.evaluate),
        #     EAStep("survivor_selection", self.survivor_selection),
        # ]

        # # Initialize EA
        # ea = EA(
        #     population_list,
        #     operations=ops,
        #     num_of_generations=100,
        # )

        # ea.run()

        # best = ea.get_solution(only_alive=False)
        # console.log(best)

        # median = ea.get_solution("median", only_alive=False)
        # console.log(median)

        # worst = ea.get_solution("worst", only_alive=False)
        # console.log(worst)
        pass
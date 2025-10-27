"""TODO(jmdm): description of script.

Author:     Fullitu
Date:       2025-10-04
Py Ver:     3.12
Status:     In progress ⚙️

Notes
-----
    For now everything is copied from a004, exact implementation needed. 
    Look into CGA approach (papers on the WA group)

TODO:

[x] normalize weights
[x] add parameters as input to evolve instead of getting them from EASettings
    NOTE: this is passed as a constructor to NE class, not evlove itself
[LATER] increasing num of gens and stuff as it increases

"""
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np
from collections.abc import Callable
import mujoco
from mujoco import viewer
from rich.console import Console
# import copy

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.a001 import Individual, JSONIterable
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from twisty_src.twisty_brain import RobotBrain
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner
from ariel.ec.a005 import Crossover
from typing import cast
from typing import Any
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

console = Console()

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
# CWD = Path.cwd()
# DATA = Path(CWD / "__data__" / SCRIPT_NAME)
# DATA.mkdir(exist_ok=True)
VIDEO_PATH = Path("examples/twisty_videos")
SEED = 42

class NeuroEvolution:
    type Population = list[Individual]
    
    @dataclass
    class NESettings:
        fitness_function: Callable[[list[float]], float]
        nn_hidden_layers: list[int]
        starting_pos: list[float]
        is_maximisation: bool
        # num_of_generations: int
        max_evals: int
        population_size: int
        mutation_rate: float
        mutation_scale: float
        mutation_magnitude: float
        random_seed: int = 42
        tournament_size: int = 5
    
    def __init__(
            self,
            fitness_function: Any,
            nn_hidden_layers: list[int],
            starting_pos: list[float],
            is_maximisation: bool,
            # num_of_generations: int,
            max_evals: int,
            population_size: int,
            mutation_rate: float,
            mutation_scale: float,
            mutation_magnitude: float,
            random_seed: int = 42,
            tournament_size: int = 5,
            ):
        
        self.fitness_function = fitness_function
        self.config = self.NESettings(
            random_seed=random_seed,
            fitness_function=fitness_function,
            nn_hidden_layers=nn_hidden_layers,
            starting_pos=starting_pos,
            is_maximisation=is_maximisation,
            # num_of_generations=num_of_generations,
            max_evals=max_evals,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            mutation_magnitude=mutation_magnitude,
            tournament_size=tournament_size,
        )
        self.rng = np.random.default_rng(self.config.random_seed)
        self.id_counter = 0

    def create_individual(self,
                          input_size: int,
                          output_size: int,
                          hidden_layers: list[int]) -> Individual:
        new_brain = RobotBrain(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers
        )

        ind = Individual()
        ind.id = self.id_counter
        self.id_counter += 1
        ind.genotype = new_brain.get_weights_as_vector().tolist()
        ind.tags = {"ps": False, "mut": False}
        ind.requires_eval = True
        return ind

    def tournament_parent_selection(self, population: Population) -> Population:
        selected_indices = random.sample(range(len(population)), min(self.config.tournament_size, len(population)))
        tournament = [{"index": i, "individual": population[i]} for i in selected_indices]
        tournament = sorted(tournament, key=lambda ind: ind["individual"].fitness, reverse=self.config.is_maximisation)
        winning_index = [winner["index"] for winner in tournament[:2]]
        for idx in winning_index:
            population[idx].tags["ps"] = True
        return population

    def crossover(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", True)]
        for idx in range(0, len(parents) - 1, 2):
            parent_i = parents[idx]
            parent_j = parents[idx + 1]
            genotype_i, genotype_j = Crossover.one_point(
                cast("list[int]", parent_i.genotype),
                cast("list[int]", parent_j.genotype),
            )

            # First child
            child_i = Individual()
            child_i.id = self.id_counter
            self.id_counter += 1
            child_i.genotype = genotype_i
            child_i.tags = {"ps": False, "mut": True}
            child_i.requires_eval = True

            # Second child
            child_j = Individual()
            child_j.id = self.id_counter
            self.id_counter += 1
            child_j.genotype = genotype_j
            child_j.tags = {"ps": False, "mut": True}
            child_j.requires_eval = True

            population.extend([child_i, child_j])
        return population

    def mutation(self, population: Population) -> Population:
        """
            Normal creep mutation -> add small gaussian noise to some genes.
            Mark requires_eval True only if at least one gene was changed.
        """
        for ind in population:
            if not ind.tags["mut"] and random.random() < self.config.mutation_rate:
                mutated = False
                for i in range(len(ind.genotype)):
                    if random.random() < self.config.mutation_magnitude:
                        ind.genotype[i] += float(np.random.normal(loc=0.0, scale=self.config.mutation_scale))
                        mutated = True
                if mutated:
                    ind.requires_eval = True
                    ind.tags["mut"] = True
        
        return population

    def evaluate(self,
                original_ind: Individual,
                brain_pop: Population,
                nn_input_size: int,
                nn_output_size: int,
                use_gecko: bool=False,
                verbose: bool=False,
    ) -> tuple[Population, int]:
        eval_count = 0
        for ind in brain_pop:
            if ind.requires_eval:
                eval_count += 1
                history = self.run(original_ind, ind, nn_input_size, nn_output_size, use_gecko=use_gecko)
                ind.fitness = self.fitness_function(self.config.starting_pos, history)
                ind.requires_eval = False
                if verbose:
                    if use_gecko:
                        console.log(f"Evaluated gecko using brain {ind.id} with fitness: {ind.fitness}")
                    else:
                        console.log(f"Evaluated robot {original_ind.id} using brain {ind.id} with fitness: {ind.fitness}")

        return brain_pop, eval_count

    def survivor_selection(self, population: Population, dead_population: Population) -> tuple[Population, Population]:
        random.shuffle(population)
        current_pop_size = len(population)
        while current_pop_size > self.config.population_size:
            idx = random.randint(0, current_pop_size - 2)

            sorted_inds = sorted([(idx, population[idx]), (idx + 1, population[idx + 1])], key=lambda ind: ind[1].fitness, reverse=self.config.is_maximisation)

            # Kill worse individual
            worse_idx = sorted_inds[1][0]
            population[worse_idx].alive = False

            dead_population.append(population[worse_idx])
            population.pop(worse_idx)
            current_pop_size -= 1

        return population, dead_population

    def reset_population_tags(self, population: Population) -> Population:
        for ind in population:
            ind.tags["ps"] = False
            ind.tags["mut"] = False
        return population

    def run(
        self,
        original_ind: Individual,
        brain_ind: Individual,
        nn_input_size: int,
        nn_output_size: int,
        mode: str = "simple_runner",
        use_gecko: bool = False
    ) -> Population:
        
        # THIS IS CRAZY INEFFICIENT CHANGE IT LATER
        # ------------------------------------------------------------------ #
        brain = RobotBrain(
            input_size=nn_input_size,
            output_size=nn_output_size,
            hidden_layers=self.config.nn_hidden_layers,
        )
        brain.set_weights_from_vector(brain_ind.genotype)
        # ------------------------------------------------------------------ #
        
        if use_gecko:
            robot = gecko()
        else:
            robot = construct_mjspec_from_graph(original_ind.genotype)

        mujoco.set_mjcb_control(None)

        # MuJoCo basics
        world = SimpleFlatWorld()

        # Set random colors for geoms
        for i in range(len(robot.spec.geoms)):
            robot.spec.geoms[i].rgba[-1] = 0.5

        # Spawn the robot at the world
        starting_pos = self.config.starting_pos.copy()
        # console.log(f"[bold yellow]Spawning robot at {starting_pos}")
        world.spawn(robot.spec, spawn_position=starting_pos, correct_for_bounding_box=True)

        # Compile the model
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)

        # Define action specification and set policy
        data.ctrl = self.rng.normal(scale=0.1, size=model.nu)

        # Initialize robot tracker
        mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )
        tracker.setup(world.spec, data)

        # Initialize controller
        ctrl = Controller(
            controller_callback_function=brain.forward_control,
            tracker=tracker,
        )

        mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        match mode:
            # Launches interactive viewer
            case "launcher":
                viewer.launch(
                    model=model,
                    data=data,
                )

            # This disables visualisation (fastest option)
            case "simple_runner":
                simple_runner(
                    model,
                    data,
                    duration=15,
                )

            # Records video of the simulation TODO, alongside saving genotypes
            case "video":
                if use_gecko:
                    twisty_or_not = "gecko"
                else:
                    twisty_or_not = "twisty" if original_ind.twisty else "not_twisty"   
                path_to_video_folder = str(VIDEO_PATH / twisty_or_not)
                    
                vid_path = Path(path_to_video_folder)
                vid_path.mkdir(parents=True, exist_ok=True)
                video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # Render with video recorder
                video_renderer(
                    model,
                    data,
                    duration=15,
                    video_recorder=video_recorder,
                )
                
            case _:
                console.log(f"Mode '{mode}' not recognized. No simulation run.")

        return tracker.history["xpos"][0]
    
    def evolve(self, original_ind: Individual, use_gecko: bool=False, verbose: bool =False) -> JSONIterable:
        
        """

        Runs the full neuroevolution process, with respect to the given body and config.
        Returns best robobrain for given body and score that this brain achived.
        
        """

        console.log("[bold cyan]Starting Neuroevolution Process")

        # Generate the model and data to determine input + output sizes of NN

        if use_gecko:
            robot = gecko()
        else:
            robot = construct_mjspec_from_graph(original_ind.genotype)
        
        mujoco.set_mjcb_control(None)
        world = SimpleFlatWorld()
        world.spawn(robot.spec, spawn_position=self.config.starting_pos, correct_for_bounding_box=True)

        model = world.spec.compile()
        data = mujoco.MjData(model)

        nn_input_size = len(data.qpos)
        nn_output_size = model.nu
        mujoco.mj_resetData(model, data)

        # Create initial population
        brain_population = [self.create_individual(nn_input_size, nn_output_size, self.config.nn_hidden_layers) 
                            for _ in range(self.config.population_size)]
        brain_population, num_evaluations = self.evaluate(original_ind, brain_population, nn_input_size, nn_output_size, use_gecko=use_gecko, verbose=False)
        best_init = sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0]
        self.run(original_ind, best_init, nn_input_size, nn_output_size, mode="video", use_gecko=use_gecko) # save initial best brain video

        console.log(f"Initial best brain fitness: {best_init.fitness}")
        console.log(f"Neuroevolution will run {self.config.max_evals} evaluations.")

        i = 0
        dead_population = []
        while num_evaluations < self.config.max_evals:
            brain_population = self.tournament_parent_selection(brain_population)
            brain_population = self.crossover(brain_population)
            brain_population = self.mutation(brain_population)
            brain_population, current_evals = self.evaluate(original_ind, brain_population, nn_input_size, nn_output_size, use_gecko=use_gecko, verbose=False)
            brain_population, dead_population = self.survivor_selection(brain_population, dead_population)
            brain_population = self.reset_population_tags(brain_population)

            # Save video of the best brain in the generation
            best_brain_in_gen = brain_population[0]
            for ind in brain_population:
                if self.config.is_maximisation:
                    if ind.fitness > best_brain_in_gen.fitness:
                        best_brain_in_gen = ind
                else:
                    if ind.fitness < best_brain_in_gen.fitness:
                        best_brain_in_gen = ind

            if num_evaluations % (self.config.max_evals / 5) > (num_evaluations + current_evals) % (self.config.max_evals / 5): # 5 checkpoints
                self.run(original_ind, best_brain_in_gen, nn_input_size, nn_output_size, mode="video", use_gecko=use_gecko)
            
            if verbose:
                console.log(f"Generation {i+1} completed.\tcurrent evaluations: {num_evaluations} out of {self.config.max_evals}\t Fitness of best brain: {best_brain_in_gen.fitness}")

            num_evaluations += current_evals
            i += 1

        best_brain = sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0]

        # Save best brain video
        self.run(original_ind, best_brain, nn_input_size, nn_output_size, mode="video", use_gecko=use_gecko)

        # Run best brain in launcher
        self.run(original_ind, best_brain, nn_input_size, nn_output_size, mode="launcher", use_gecko=use_gecko)

        return best_brain.genotype
    
    def get_hyperparameters(self) -> dict[Any, Any]:
        """Returns the hyperparameters used in the neuroevolution process."""
        return self.config.__dict__
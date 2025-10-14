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

console = Console()

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
VIDEO_PATH = Path("examples/twisty_videos")
SEED = 40

class NeuroEvolution:
    type Population = list[Individual]
    
    @dataclass
    class NESettings:
        fitness_function: Callable[[list[float]], float]
        nn_hidden_layers: list[int]
        starting_pos: list[float]
        is_maximisation: bool
        num_of_generations: int
        population_size: int
        mutation_rate: float
        mutation_scale: float
        mutation_magnitude: float
        random_seed: int = 42
    
    def __init__(
            self,
            fitness_function: Callable[[list[float]], float],
            nn_hidden_layers: list[int],
            starting_pos: list[float],
            is_maximisation: bool,
            num_of_generations: int,
            population_size: int,
            mutation_rate: float,
            mutation_scale: float,
            mutation_magnitude: float,
            random_seed: int = 42
            ):
        
        self.fitness_function = fitness_function
        self.config = self.NESettings(
            random_seed=random_seed,
            fitness_function=fitness_function,
            nn_hidden_layers=nn_hidden_layers,
            starting_pos=starting_pos,
            is_maximisation=is_maximisation,
            num_of_generations=num_of_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            mutation_magnitude=mutation_magnitude,
        )
        self.rng = np.random.default_rng(self.config.random_seed)

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
        ind.genotype = new_brain.get_weights_as_vector().tolist()
        ind.tags = {"mut": False}
        return ind

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
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        for idx in range(0, len(parents), 2):
            parent_i = parents[idx]
            parent_j = parents[idx]
            genotype_i, genotype_j = Crossover.one_point(
                cast("list[int]", parent_i.genotype),
                cast("list[int]", parent_j.genotype),
            )

            # First child
            child_i = Individual()
            child_i.genotype = genotype_i
            child_i.tags = {"mut": True}
            child_i.requires_eval = True

            # Second child
            child_j = Individual()
            child_j.genotype = genotype_j
            child_j.tags = {"mut": True}
            child_j.requires_eval = True

            population.extend([child_i, child_j])
        return population

    def mutation(self, population: Population) -> Population:
        """
            Normal creep mutation -> add small gaussian noise to some genes.
        """

        for ind in population:
            if ind.tags.get("mut", False):
                if np.random.random() > self.config.mutation_rate:
                    continue
                
                mutated = cast("list[int]", ind.genotype)
                for gene in mutated:
                    if np.random.random() > self.config.mutation_magnitude:
                        continue
                    gene += np.random.normal(loc=0.0, scale=self.config.mutation_scale)

                ind.genotype = mutated
                ind.requires_eval = True
                ind.tags = {"mut": True}
        return population

    def reset_population_tags(self, population: Population) -> Population:
        for ind in population:
            ind.tags = {"mut": False}
        return population

    def evaluate(self,
                 original_ind: Individual,
                 brain_pop: Population,
                 nn_input_size: int,
                 nn_output_size: int
    ) -> Population:
        
        for ind in brain_pop:
            if ind.requires_eval:
                history = self.run(original_ind, ind, nn_input_size, nn_output_size)
                ind.fitness = self.fitness_function(history)
                ind.requires_eval = False
                console.log(f"Evaluated robot {original_ind.id} using brain {ind.id} with fitness: {ind.fitness}")

        return brain_pop

    def run(
        self,
        original_ind: Individual,
        brain_ind: Individual,
        nn_input_size: int,
        nn_output_size: int,
        mode: str = "simple_runner",
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
        robot = construct_mjspec_from_graph(original_ind.genotype)

        mujoco.set_mjcb_control(None)

        # MuJoCo basics
        world = SimpleFlatWorld()

        # Set random colors for geoms
        for i in range(len(robot.spec.geoms)):
            robot.spec.geoms[i].rgba[-1] = 0.5

        # Spawn the robot at the world
        world.spawn(robot.spec, spawn_position=self.config.starting_pos)

        # Compile the model
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)

        # TODO save genotype
        # Save the model to XML
        # xml = world.spec.to_xml()
        # with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        #     f.write(xml)

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
            time_steps_per_ctrl_step=1,
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
                    duration=30,
                )

            # Records video of the simulation TODO, alongside saving genotypes
            case "video":
                # twisty_or_not = "twisty" if original_ind.twisty else "not_twisty"
                # path_to_video_folder = str(VIDEO_PATH / twisty_or_not)
                # vid_path = Path(path_to_video_folder)
                # vid_path.mkdir(parents=True, exist_ok=True)
                # video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # # Render with video recorder
                # video_renderer(
                #     model,
                #     data,
                #     duration=30,
                #     video_recorder=video_recorder,
                # )
                # TODO remove below and uncomment above when done
                simple_runner(
                    model,
                    data,
                    duration=30,
                )
                
            case _:
                console.log(f"Mode '{mode}' not recognized. No simulation run.")

        return tracker.history["xpos"][0]

    def survivor_selection(self, population: Population) -> Population:
        random.shuffle(population)
        current_pop_size = len(population)
        for idx in range(len(population)):
            ind_i = population[idx]
            ind_j = population[idx + 1]

            # Kill worse individual
            if ind_i.fitness > ind_j.fitness and self.config.is_maximisation:
                ind_j.alive = False
            else:
                ind_i.alive = False

            # Termination condition
            current_pop_size -= 1
            if current_pop_size <= self.config.population_size:
                break

        return population
    
    def evolve(self, original_ind: Individual) -> JSONIterable:
        
        """

        Runs the full neuroevolution process, with respect to the given body and config.
        Returns best robobrain for given body and score that this brain achived.
        
        """

        # Generate the model and data to determine input + output sizes of NN

        mujoco.set_mjcb_control(None) # NOTE: IDK WHY THIS NEEDS TO BE HERE BUT IT WORKS SO I'M NOT COMPLAINING
        robot = construct_mjspec_from_graph(original_ind.genotype)

        world = SimpleFlatWorld()
        world.spawn(robot.spec, spawn_position=self.config.starting_pos)

        model = world.spec.compile()
        data = mujoco.MjData(model)

        nn_input_size = len(data.qpos)
        nn_output_size = model.nu
        mujoco.mj_resetData(model, data)

        # Create initial population
        brain_population = [self.create_individual(nn_input_size, nn_output_size, self.config.nn_hidden_layers) 
                            for _ in range(self.config.population_size)]
        brain_population = self.evaluate(original_ind, brain_population, nn_input_size, nn_output_size)
        previous_best_brain = sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0]
        self.run(original_ind, previous_best_brain, nn_input_size, nn_output_size, mode="video")
        
        for _ in range(self.config.num_of_generations):
            brain_population = self.parent_selection(brain_population)
            brain_population = self.crossover(brain_population)
            brain_population = self.mutation(brain_population)
            brain_population = self.evaluate(original_ind, brain_population, nn_input_size, nn_output_size)
            brain_population = self.survivor_selection(brain_population)
            brain_population = self.reset_population_tags(brain_population)

            # Save video of the best brain in the generation
            best_brain_in_gen = sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0]
            if previous_best_brain.genotype != best_brain_in_gen.genotype:
                self.run(original_ind, best_brain_in_gen, nn_input_size, nn_output_size, mode="video")

        best_brain = sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0]

        # Uncomment for debug to see best brain in action at the end of evolution
        self.run(original_ind, best_brain, nn_input_size, nn_output_size, mode="launcher")

        return best_brain.genotype
    
    def get_hyperparameters(self) -> dict[Any, Any]:
        """Returns the hyperparameters used in the neuroevolution process."""
        return self.config.__dict__
        
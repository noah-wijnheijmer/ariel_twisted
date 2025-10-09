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
from pathlib import Path
import numpy as np
from collections.abc import Callable
import mujoco
from mujoco import viewer
from rich.console import Console
import copy

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
# from ariel.ec.a000 import IntegersGenerator
from ariel.ec.a001 import Individual, JSONIterable
# from ariel.ec.a004 import EA, EAStep, EASettings, Population
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from twisty_src.twisty_brain import RobotBrain
# from twisty import EASettings, DATA, SCRIPT_NAME, SEED
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner

console = Console()

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
# CWD = Path.cwd()
# DATA = Path(CWD / "__data__" / SCRIPT_NAME)
# DATA.mkdir(exist_ok=True)
SEED = 40

class NeuroEvolution:
    type Population = list[Individual]

    def __init__(
            self,
            fitness_function: Callable[[list[float]], float],
            config, # EASettings
            ):
        self.config = config
        self.fitness_function = fitness_function
        self.rng = np.random.default_rng(SEED)

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
        return ind

    def parent_selection(self, population: Population) -> Population:
        # random.shuffle(population)
        # for idx in range(0, len(population) - 1, 2):
        #     ind_i = population[idx]
        #     ind_j = population[idx + 1]

        #     # Compare fitness values
        #     if ind_i.fitness > ind_j.fitness and self.config.is_maximisation:
        #         ind_i.tags = {"ps": True}
        #         ind_j.tags = {"ps": False}
        #     else:
        #         ind_i.tags = {"ps": False}
        #         ind_j.tags = {"ps": True}
        # return population
        pass

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

    def evaluate(self, original_ind: Individual,
                 brain_pop: Population,
                 nn_input_size: int,
                 nn_output_size: int
    ) -> Population:
        
        for ind in brain_pop:
            if ind.requires_eval:
                history = self.run(original_ind, ind, nn_input_size, nn_output_size)
                ind.fitness = self.fitness_function(history)

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

        # Number of actuators and DoFs
        console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

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

        console.log(f"xpos before sim: {tracker.history["xpos"][0]}") # TODO: REMOVE DEBUG

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

            # Records video of the simulation TODO, alongside saving genoptyps
            # case "video":
                # path_to_video_folder = str(DATA / "videos") 
                # video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # # Render with video recorder
                # video_renderer(
                #     model,
                #     data,
                #     duration=30,
                #     video_recorder=video_recorder,
                # )
            
            case _:
                console.log(f"Mode '{mode}' not recognized. No simulation run.")

        # print(20 * "-")
        # print(tracker.history["xpos"][0][-1])
        # print(20 * "-")
        return tracker.history["xpos"][0]

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
        brain_population = [self.create_individual(nn_input_size, nn_output_size, self.config.nn_hidden_layers) for _ in range(self.config.population_size)]
        
        for _ in range(self.config.num_of_generations):
            brain_population = self.evaluate(original_ind, brain_population, nn_input_size, nn_output_size)
            print([ind.fitness for ind in brain_population])

        return sorted(brain_population, key=lambda ind: ind.fitness, reverse=self.config.is_maximisation)[0].genotype
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

        
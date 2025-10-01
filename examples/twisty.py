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
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco
import numpy as np
from rich.console import Console
from mujoco import viewer
import torch
import torch.nn as nn

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (NUM_OF_FACES,
                                                       NUM_OF_ROTATIONS,
                                                       NUM_OF_TYPES_OF_MODULES)
from ariel.body_phenotypes.robogen_lite.constructor import \
    construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder, save_graph_as_json)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.cpg_with_sensory_feedback import \
    CPGSensoryFeedback
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec.a001 import Individual
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner

if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 40
# twisty indexes (rotations of 45, 135, 225, 315 degrees) 
TWIST_I = [1, 3, 5, 7]
# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

def create_individual(con_twisty: bool) -> Individual:
    ind = Individual()
    num_modules = 20

    # "Type" probability space
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    if con_twisty is True:   
        rotation_probability_space = RNG.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
    else:
        # if twisty not true, the twisted angles are given the value zero, so they can't be selected.
        rotation_probability_space = RNG.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
        for i in rotation_probability_space:
            i[TWIST_I] = [0]*4

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )
    # Save the graph to a file
    save_graph_as_json(
        graph,
        DATA / "graph.json",
    )
    ind.genotype = graph
    ind.twisty = con_twisty
    return ind

def main() -> None:
    """Entry point."""
    population_size = 5
    initial_population = []
    for i in range(population_size):
        # twisty is false, no 45 degrees angles
        initial_population.append(create_individual(False))
    for i in range(population_size):
        # twisty is True
        initial_population.append(create_individual(True))
    # Print all nodes
    for ind in initial_population:
        core = construct_mjspec_from_graph(ind.genotype)
        # Simulate the robot
        run(core, ind)


def run(
    robot: CoreModule,
    individual: Individual,
    mode: str = "launcher",
) -> None:
    """Entry point."""
    # BugFix -> "Python exception raised"
    mujoco.set_mjcb_control(None)

    # # MuJoCo configuration
    # viz_options = mujoco.MjvOption()  # visualization of various elements

    # # Visualization of the corresponding model or decoration element
    # viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    # viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Save the model to XML
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Define action specification and set policy
    data.ctrl = RNG.normal(scale=0.1, size=model.nu)

    # Actuators and CPG
    mujoco.set_mjcb_control(None)
    weight_matrix = RNG.uniform(-0.1, 0.1, size=(model.nu, model.nu))
    cpg = CPGSensoryFeedback(
        num_neurons=int(model.nu),
        sensory_term=-0.0,
        _lambda=0.01,
        coupling_weights=weight_matrix,
    )
    cpg.reset()
    # add brain genotype to the individual
    individual.brain_genotype = cpg.c
    console.log("Brain genotype:")
    console.log("Qpos at start:")

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
        controller_callback_function=policy,
        time_steps_per_ctrl_step=1,
        tracker=tracker,
    )

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg))

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

        # Records video of the simulation
        case "video":
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=30,
                video_recorder=video_recorder,
            )
        
        case _:
            console.log(f"Mode '{mode}' not recognized. No simulation run.")

    console.log(f"xpos after sim: {tracker.history["xpos"][0]}") # TODO: REMOVE DEBUG

def policy(
    model: mujoco.MjModel,  # noqa: ARG001
    data: mujoco.MjData,
    cpg: CPGSensoryFeedback,
) -> None:
    """Use feedback term to shift the output of the CPGs."""
    x, _ = cpg.step()
    return x * np.pi / 2

if __name__ == "__main__":
    # Test several times
    main()

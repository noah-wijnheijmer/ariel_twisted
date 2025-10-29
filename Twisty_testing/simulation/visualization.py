import numpy as np
from robot_body.modules.core import CoreModule
from ea_components.individual import Individual
from simulation.environments._simple_flat import SimpleFlatWorld
from utils.tracker import Tracker
from utils.runners import simple_runner
from utils.video_recorder import VideoRecorder
from utils.renderers import video_renderer
from simulation.cpg.sf_cpg import CPGSensoryFeedback, sf_policy
from simulation.cpg.na_cpg import (
    NaCPG, create_fully_connected_adjacency, na_policy
)
from simulation.controller import Controller
from rich.console import Console
from typing import Any
from mujoco import viewer
import mujoco

console = Console()
SEED = 40
RNG = np.random.default_rng(SEED)

def visualize_champ(robot: CoreModule, individual: Individual, spawn_pos: list[float], path_settings: list[Any], brain_type: str, mode: str = "video") -> None: #Visualizes the champion robot after evolution
    """Entry point."""
    # BugFix -> "Python exception raised"
    mujoco.set_mjcb_control(None)

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec, position=spawn_pos) #read as champion_robot.spec. it's just being accessed through the parameter name robot.

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)
    
    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Save the model to XML
    xml = world.spec.to_xml()
    with (path_settings[0] / f"{path_settings[1]}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)
    
    # Actuators and CPG
    mujoco.set_mjcb_control(None)
    if brain_type == "na_cpg":
        adj_dict = create_fully_connected_adjacency(model.nu)
        cpg = NaCPG(adj_dict, angle_tracking=True)
        cpg.reset()
        gen = cpg.get_flat_params()
        # add brain genotype to the individual
        individual.brain_genotype = gen

        # Initialize robot tracker
        mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )
        tracker.setup(world.spec, data)

        ctrl = Controller(
            controller_callback_function=na_policy,
            time_steps_per_ctrl_step=1,
            tracker=tracker,
        )
    elif brain_type == "sf_cpg":
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
            controller_callback_function=sf_policy,
            time_steps_per_ctrl_step=1,
            tracker=tracker,
        )

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg)) # type: ignore

    console.log(f"xpos before sim: {tracker.history['xpos'][0]}") # type: ignore

    match mode:
        # Launches interactive viewer
        case "launcher":
            # Launch viewer with proper camera settings for small robots
            viewer.launch(model, data)
            # console.log("🎥 Viewer launched!")

        # This disables visualisation (fastest option)
        case "simple_runner":
            simple_runner(
                model,
                data,
                duration=30,
            )

        # Records video of the simulation
        case "video":
            path_to_video_folder = str(path_settings[0] / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=30,
                video_recorder=video_recorder,
            )

    # return fitness_function(tracker.history["xpos"])
            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=30,
                video_recorder=video_recorder,
            )
        
        case _:
            console.log(f"Mode '{mode}' not recognized. No simulation run.")

    # return fitness_function(tracker.history["xpos"])
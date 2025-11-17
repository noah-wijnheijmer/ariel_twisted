# Third-party libraries
import time
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import nevergrad as ng


# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# Local libraries
# from ariel.utils.renderers import video_renderer
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from twisty_src.twisty_brain import RobotBrain
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
# from ariel.utils.video_recorder import VideoRecorder

# Keep track of data / history
HISTORY = []

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# -- HYPERPARAMETERS --- #
HIDDEN_LAYERS = [32]
STARTING_POSITION = [0, 0, 0.1]

def fitness_function_basic(history: list[float]) -> float:
    # move forward, direction got trail and errored
    
    _, ys, _ = STARTING_POSITION.copy()
    _, ye, _ = history[-1]

    # maximize the distance
    y_distance = np.abs(ys - ye) # negative y distance = forward movement
    return y_distance


def run(params: list[int], brain: RobotBrain, mode: str ="simple") -> None:
    """Main function to run the simulation with random movements."""

    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    gecko_core = gecko()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=STARTING_POSITION.copy(), correct_for_bounding_box=True)

    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)  # type: ignore

    # Initialize robot tracker
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
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

    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    match mode:
        case "viewer":
            # Launch the viewer
            viewer.launch(
                    model=model,
                    data=data,
            )
        case "simple":
            # Run the simulation
            simple_runner(
                    model=model,
                    data=data,
                    duration=15,
            )
        case "video":
            # Non-default VideoRecorder options
            PATH_TO_VIDEO_FOLDER = "./__videos__"
            video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=30,
                video_recorder=video_recorder,
            )


        case _:
            raise ValueError(f"Mode {mode} not recognized.")

    history  = tracker.history["xpos"][0]  # Get history of the first tracked object

    fitness = fitness_function_basic(history)

    return -fitness


def main():
    robot = gecko()
    
    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(robot.spec, spawn_position=STARTING_POSITION.copy(), correct_for_bounding_box=True)

    model = world.spec.compile()
    data = mj.MjData(model)

    nn_input_size = len(data.qpos)
    nn_output_size = model.nu
    mj.mj_resetData(model, data)

    brain = RobotBrain(
        input_size=nn_input_size,
        output_size=nn_output_size,
        hidden_layers=HIDDEN_LAYERS,
    )

    num_params = len(brain.get_weights_as_vector())
    param = ng.p.Array(shape=(num_params,))

    start_time = time.time()
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=5000, num_workers=1)

    print("Starting optimization...")
    best_brain = optimizer.minimize(lambda x: run(params=x, brain=brain, mode="simple"))
    print(f"Optimization setup took {time.time() - start_time:.2f} seconds.")

    run(params=best_brain.value, brain=brain, mode="video")


if __name__ == "__main__":
    main()

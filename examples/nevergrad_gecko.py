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
EVAL_COUNTER = 0
OPTIMIZER_NAME = "cma"
BUDGET = 5000
NUM_WORKERS = 1


def _set_brain_parameters(brain: RobotBrain, params: npt.ArrayLike) -> None:
    """Apply a flat parameter vector (or Nevergrad container) to the brain."""
    vector = getattr(params, "value", params)
    brain.set_weights_from_vector(np.asarray(vector, dtype=np.float32))

def fitness_function_basic(history: list[npt.NDArray[np.float64]]) -> float:
    """Reward forward motion along the gecko's primary (Y) axis."""
    ys = history[0][1]
    ye = history[-1][1]

    # maximize the distance
    y_distance = np.abs(ys - ye)  # negative delta corresponds to forward motion
    return y_distance


def run(params: npt.ArrayLike, brain: RobotBrain, mode: str ="simple") -> None:
    """Main function to run the simulation with random movements."""

    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    _set_brain_parameters(brain, params)

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
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    print(f"[Eval {EVAL_COUNTER}] Fitness: {fitness:.4f}")

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

    initial_weights = brain.get_weights_as_vector().astype(np.float32)
    param = ng.p.Array(init=initial_weights.copy())

    def _make_optimizer():
        name = OPTIMIZER_NAME.lower()
        common_kwargs = dict(parametrization=param, budget=BUDGET, num_workers=NUM_WORKERS)
        if name == "cma":
            return ng.optimizers.CMA(**common_kwargs)
        if name in {"oneplusone", "1+1"}:
            return ng.optimizers.ParametrizedOnePlusOne(**common_kwargs)
        if name == "tbpsa":
            return ng.optimizers.TBPSA(**common_kwargs)
        return ng.optimizers.NGOpt(**common_kwargs)

    start_time = time.time()
    optimizer = _make_optimizer()

    print("Starting optimization...")
    best_brain = optimizer.minimize(lambda x: run(params=x, brain=brain, mode="simple"))
    print(f"Optimization setup took {time.time() - start_time:.2f} seconds.")

    print("Replaying best brain in video mode...")
    run(params=best_brain.value, brain=brain, mode="video")


if __name__ == "__main__":
    main()

# Third-party libraries
import time
# import matplotlib.pyplot as plt
import mujoco as mj
import mujoco
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import nevergrad as ng
import matplotlib.pyplot as plt
import torch

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# Local libraries
# from ariel.utils.renderers import video_renderer
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency

# Keep track of data / history
HISTORY = []

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# -- HYPERPARAMETERS --- #
STARTING_POSITION = [0, 0, 0.1]
EVAL_COUNTER = 0
OPTIMIZER_NAME = "NGOpt"  # Options: "CMA", "1+1", "TBPSA", "NGOpt"
BUDGET = 500
NUM_WORKERS = 1



def fitness_function_basic(history: list[npt.NDArray[np.float64]]) -> float:
    """Reward forward motion along the gecko's primary (Y) axis."""
    ys = history[0][1]
    ye = history[-1][1]

    # maximize the distance
    y_distance = ys - ye  # negative delta corresponds to forward motion
    return y_distance

def run_with_cpg(flat_cpg_params: npt.ArrayLike, mode: str ="simple") -> None:
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

    # Prepare CPG with new parameters
    cpg = NaCPG(adjacency_dict=create_fully_connected_adjacency(model.nu))
    vector = getattr(flat_cpg_params, "value", flat_cpg_params)
    cpg.set_flat_params(torch.from_numpy(vector))

    def cpg_forward(
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """Forward pass for control inputs."""
        return cpg.forward().detach().numpy()

    # Initialize controller
    ctrl = Controller(
        controller_callback_function=cpg_forward,
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
            tracking_video_renderer(
                model=model,
                data=data,
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

    mj.mj_resetData(model, data)

    cpg = NaCPG(adjacency_dict=create_fully_connected_adjacency(model.nu))

    initial_params = cpg.get_flat_params().numpy().astype(np.float32)
    param = ng.p.Array(init=initial_params.copy())
    
    def _make_optimizer():
        name = OPTIMIZER_NAME.lower()
        common_kwargs = dict(parametrization=param, budget=BUDGET, num_workers=NUM_WORKERS)
        if name == "cma":
            return ng.optimizers.CMA(**common_kwargs)
        if name in {"oneplusone", "1+1"}:
            return ng.optimizers.OnePlusOne(**common_kwargs)
        if name == "tbpsa":
            return ng.optimizers.TBPSA(**common_kwargs)
        return ng.optimizers.NGOpt(**common_kwargs)

    start_time = time.time()
    optimizer = _make_optimizer()

    print("Starting optimization...")
    best_cpg_parameters = optimizer.minimize(run_with_cpg)

    # # Store history
    # fitness_history = []

    # try:
    #     cumulative_fitness = 0.0
    #     for i in range(optimizer.budget):
    #         x = optimizer.ask()
    #         fitness = run_with_cpg(flat_cpg_params=x.value, mode="simple")
    #         optimizer.tell(x, fitness)
    #         cumulative_fitness -= fitness
    #         if (i + 1) % 100 == 0:
    #             avg_fitness = cumulative_fitness / 100
    #             print(f"Iteration {i + 1}, Average Fitness over last 100 evaluations: {avg_fitness:.4f}")
    #             cumulative_fitness = 0.0
    #             fitness_history.append(avg_fitness)
    # except KeyboardInterrupt:
    #     print("Optimization interrupted by user.")
    
    # best_cpg_parameters = optimizer.provide_recommendation()
    # print(f"Optimization setup took {(time.time() - start_time)/60:.2f} minutes.")

    # Plot curve
    # plt.plot(fitness_history)
    # plt.xlabel("Iteration")
    # plt.ylabel("Fitness")
    # plt.title(f"{OPTIMIZER_NAME} Optimization History")
    # plt.show()

    print("Replaying best brain in video mode...")
    run_with_cpg(flat_cpg_params=best_cpg_parameters.value, mode="video")


if __name__ == "__main__":
    main()

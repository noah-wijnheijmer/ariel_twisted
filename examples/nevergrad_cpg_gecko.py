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
import os
import json
from collections.abc import Callable
import datetime

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_untwisted import gecko_untwisted
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_good import gecko_good
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_front import gecko_front
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist import gecko_doubletwist
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist_turtle import gecko_doubletwist_turtle
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# Local libraries
# from ariel.utils.renderers import video_renderer
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from nevergrad_gecko import create_file_name

# Keep track of data / history
HISTORY = []

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# -- HYPERPARAMETERS --- #
STARTING_POSITION = [0, 0, 0.1]
EVAL_COUNTER = 0
OPTIMIZER_NAME = "cma"  # Options: "CMA", "1+1", "TBPSA", "NGOpt", PSO
BUDGET = 26000
NUM_WORKERS = 1

def create_file_name(
        experiment_name: str,
        robot: Callable
    ) -> str:
    
    name_components = []
    name_components.append(experiment_name)
    
    robot_name = robot.__name__
    name_components.append(robot_name)
    
    optimizer_name = OPTIMIZER_NAME.lower()
    name_components.append(optimizer_name)
    
    budget = str(BUDGET)[:-3] + 'k' if BUDGET >= 1000 else str(BUDGET)
    name_components.append(budget)
    
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
        "%Y%m%d_%H%M%S",
    )
    name_components.append(timestamp)
    
    return "_".join(name_components)


def fitness_function_basic(history: list[npt.NDArray[np.float64]]) -> float:
    """Reward forward motion along the gecko's primary (Y) axis."""
    ys = history[0][1]
    ye = history[-1][1]

    # maximize the distance
    y_distance = ys - ye  # negative delta corresponds to forward motion
    return y_distance

def run_with_cpg(
        robot_model: Callable,
        flat_cpg_params: npt.ArrayLike,
        mode: str ="simple",
        video_name: str | None = None
    ) -> None:
    
    """Main function to run the simulation with random movements."""

    if mode not in {"viewer", "simple", "video"}:
        raise ValueError(f"Mode {mode} not recognized. Choose from 'viewer', 'simple', or 'video'.")
    
    if mode == "video" and video_name is None:
        raise ValueError("In 'video' mode, 'video_name' must be provided.")
    
    if mode != "video" and video_name is not None:
        print("Warning: 'video_name' is only used in 'video' mode. Ignoring it.")

    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    gecko_core = robot_model()

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
            video_recorder = VideoRecorder(
                output_folder=PATH_TO_VIDEO_FOLDER,
                file_name=video_name
            )
            video_recorder._add_timestamp_to_file_name = False  # Disable timestamp, handled by custom naming
            
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
    # print(f"Fitness: {fitness:.4f}") # DEBUG    
    return -fitness


def run_cpg_experiment(seed: int, gecko_model: Callable = gecko, experiment_name: str = "cpg_frwd") -> list[float]:
    global RNG
    RNG = np.random.default_rng(seed)
    save_file_name = create_file_name(experiment_name=experiment_name, robot=gecko_model)
    
    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(gecko_model().spec, spawn_position=STARTING_POSITION.copy(), correct_for_bounding_box=True)

    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)

    cpg = NaCPG(adjacency_dict=create_fully_connected_adjacency(model.nu), seed=seed)

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
        if name == "pso":
            return ng.optimizers.PSO(**common_kwargs)
        return ng.optimizers.NGOpt(**common_kwargs)
    
    start_time = time.time()
    optimizer = _make_optimizer()
    optimizer.parametrization.random_state = np.random.RandomState(seed)

    print("Starting optimization...")

    # Store history
    fitness_history = []

    try:
        cumulative_fitness = 0.0
        for i in range(optimizer.budget):
            x = optimizer.ask()
            fitness = run_with_cpg(robot_model=gecko_model, flat_cpg_params=x.value, mode="simple")
            optimizer.tell(x, fitness)
            cumulative_fitness -= fitness
            if (i + 1) % 100 == 0:
                avg_fitness = cumulative_fitness / 100
                print(f"Iteration {i + 1}, Average Fitness over last 100 evaluations: {avg_fitness:.4f}")
                cumulative_fitness = 0.0
                fitness_history.append(avg_fitness)
                # print(f"FITNESS HISTORY: {fitness_history}") # DEBUG
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    
    best_cpg_parameters = optimizer.provide_recommendation()
    print(f"Optimization setup took {(time.time() - start_time)/60:.2f} minutes.")

    print("Replaying best brain in video mode...")
    run_with_cpg(robot_model=gecko_model, flat_cpg_params=best_cpg_parameters.value, mode="video", video_name=save_file_name)

    return fitness_history

if __name__ == "__main__":
    num_runs_per_experiment = 1
    experiment_name = "cpg_frwd"
    
    for gecko_type in [
        # gecko,
        gecko_untwisted,
        gecko_good,
        # gecko_doubletwist,
        # gecko_doubletwist_turtle,
        # gecko_front,
    ]:
        histories = []
        
        for run_idx in range(num_runs_per_experiment):
            seed = SEED
            # seed += run_idx
            print("\n" + "=" * 50)
            print(f"Running experiment for {gecko_type.__name__}, run {run_idx + 1}")
            print("-" * 50)
            print(f"HYPERPARAMETERS:")
            print(f"  Optimizer: {OPTIMIZER_NAME}")
            print(f"  Budget: {BUDGET}")
            print(f"  Num Workers: {NUM_WORKERS}")
            print("-" * 50)
            histories.append(run_cpg_experiment(
                                    seed=seed,
                                    gecko_model=gecko_type,
                                    experiment_name=experiment_name
                                ))

        save_file_name = create_file_name(experiment_name=experiment_name, robot=gecko_type)
        
        # Save fitness history to JSON
        with open(f"./__data__/{save_file_name}.json", "w") as f:
            json.dump(histories, f)
        
        # Plot curve
        # for hist in histories:
        # plt.plot(hist)
        mean_runs = np.zeros(len(histories[0]))
        mean_slope = np.zeros(len(histories[0]))
        for history in histories:
            for i in range(len(history)):
                mean_runs[i] += history[i]
                if i != (len(history)-1):
                    deltax = history[i+1] - history[i]
                    mean_slope[i] += deltax 

        # print(mean_runs)
        for j in range(len(mean_runs)):
            mean_runs[j] = mean_runs[j]/(len(histories))
            mean_slope[j] = mean_slope[j]/(len(histories))
        # print(mean_runs)
        y = mean_slope
        x = np.linspace(1, len(mean_runs), num=len(mean_runs))
        plt.plot(x, y)
        plt.xlabel("Iteration")
        plt.ylabel("average slope")
        plt.title(f"{gecko_type.__name__} optimization slope over Time")
        os.makedirs("./__figures__", exist_ok=True)
        plt.savefig(f"./__figures__/{save_file_name}_slope_curve.png")
        plt.close()
        y = mean_runs
        x = np.linspace(1, len(mean_runs), num=len(mean_runs))
        plt.plot(x, y)
        plt.xlabel("Iteration")
        plt.ylabel("average fitness")
        plt.title(f"{gecko_type.__name__} optimization fitness over Time")
        plt.savefig(f"./__figures__/{save_file_name}_fitness_curve.png")
        plt.close()

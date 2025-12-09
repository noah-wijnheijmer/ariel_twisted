import numpy as np
from robot_body.modules.core import CoreModule
from ea_components.individual import Individual
from simulation.environments._simple_flat import SimpleFlatWorld
from utils.tracker import Tracker
from ariel.utils.renderers import tracking_video_renderer
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
# RNG = np.random.default_rng(SEED)
import numpy as np
import matplotlib.pyplot as plt  # ADD THIS
# ...existing code...

def visualize_champ(robot: CoreModule, individual: Individual, correct_for_bounding: bool, spawn_z: float, spawn_xy: list[float], path_settings: list[Any], brain_type: str, mode: str = "video", video_name: str | None = None) -> list[list[float]]:
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    world.spawn(robot.spec, spawn_z=spawn_z, spawn_xy=spawn_xy, correct_for_bounding_box=correct_for_bounding)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    xml = world.spec.to_xml()
    with (path_settings[0] / f"{path_settings[1]}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # ---------- Brain / Controller setup (define ctrl & cpg) ----------
    tracker = Tracker(mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    if brain_type == "na_cpg":
        adj = create_fully_connected_adjacency(model.nu)
        cpg = NaCPG(adj, angle_tracking=True)
        cpg.reset()
        individual.brain_genotype = cpg.get_flat_params()
        ctrl = Controller(controller_callback_function=na_policy, time_steps_per_ctrl_step=1, tracker=tracker)
    elif brain_type == "sf_cpg":
        # assumes individual.brain_genotype already exists
        weights = individual.brain_genotype
        cpg = CPGSensoryFeedback(num_neurons=int(model.nu), sensory_term=0.0, _lambda=0.01, coupling_weights=weights)
        cpg.reset()
        ctrl = Controller(controller_callback_function=sf_policy, time_steps_per_ctrl_step=1, tracker=tracker)
    else:
        cpg = None
        ctrl = Controller(controller_callback_function=lambda m, d, _: None, time_steps_per_ctrl_step=1, tracker=tracker)
    # ---------------------------------------------------------------

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg))  # type: ignore

    sim_duration = 30.0
    timestep = model.opt.timestep
    steps = int(sim_duration / timestep)
    history: list[list[float]] = []

    def step_and_collect(n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(model, data)
            base_pos = np.array(data.xipos[0], dtype=float)
            if np.isfinite(base_pos).all():
                history.append([float(base_pos[0]), float(base_pos[1]), float(base_pos[2])])

    if mode == "simple_runner":
        step_and_collect(steps)
    elif mode == "video":
        path_to_video_folder = str(path_settings[0] / "videos")
        video_recorder = VideoRecorder(output_folder=path_to_video_folder, file_name=video_name)
        step_and_collect(steps)
        tracking_video_renderer(model, data, duration=sim_duration, video_recorder=video_recorder)
    elif mode == "launcher":
        warmup_steps = int(1.0 / timestep)
        step_and_collect(warmup_steps)
        viewer.launch(model, data)
    else:
        console.log(f"Mode '{mode}' not recognized. No simulation run.")

    return history


def show_qpos_history(history: list[list[float]]) -> None:
    pos_data = np.array(history, dtype=float)
    if pos_data.ndim != 2 or pos_data.shape[1] < 2:
        console.log("[red]History format invalid for plotting.")
        return
    # Drop non-finite rows to avoid plotting NaNs
    pos_data = pos_data[np.isfinite(pos_data).all(axis=1)]
    if pos_data.size == 0:
        console.log("[yellow]No finite samples to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitness_over_generations(
    experiments_data: dict[str, Any],
    repetitions: int, 
    population_name: str = "mixed_twisty"
) -> None:
    """Plot fitness over generations showing max, average, and min values.

    Args:
        experiment_data: Dictionary containing generation statistics
            from the experiment
        population_name: Name of the population key in the data
            (e.g., "mixed_twisty", "twisty", "non_twisty")
    """
    if (
        "generations" not in experiments_data[0]
        or len(experiments_data[0]["generations"]) == 0
    ):
        console.log("No generation data found in experiment data.")
        return

    generations = []
    max_fitness = np.zeros(len(experiments_data[0]["generations"]))
    avg_fitness = np.zeros(len(experiments_data[0]["generations"]))
    i = 0
    for experiment_data in experiments_data:
        maxs = []
        means = []
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

        # Check if the population_name exists in this generation
            if population_name in gen_data:
                stats = gen_data[population_name]
                maxs.append(stats["max"])
                means.append(stats["mean"])
            else:
                msg = (
                    f"Population '{population_name}' "
                    f"not found in generation {gen_data['generation']}"
                )
                console.log(msg)
                return
        i += 1
        for i in range(len(experiment_data["generations"])):
            max_fitness[i] += maxs[i]
            avg_fitness[i] += means[i]
    for i in range(len(max_fitness)):
        max_fitness[i] = max_fitness[i]/repetitions
        avg_fitness[i] = avg_fitness[i]/repetitions
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        generations, max_fitness, marker="o", markersize=4, label="Max", linewidth=2,
    )
    plt.plot(
        generations, avg_fitness, marker="o", markersize=4, label="Mean", linewidth=2,
    )
    plt.xlabel("Generation No.", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    title = f"Fitness over Generations - {population_name.replace('_', ' ').title()}"
    plt.title(title, fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(visible=True, alpha=0.3)
    
    # I force integer ticks on x-axis (half generations don't make sense to me)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(f"{population_name}_fitness.png")
    plt.tight_layout()
    plt.show()
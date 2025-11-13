from robot_body.modules.core import CoreModule
from ea_components.individual import Individual
from simulation.environments._simple_flat import SimpleFlatWorld
from utils.tracker import Tracker
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
import numpy as np
import mujoco as mj
console = Console()
SEED = 40
RNG = np.random.default_rng(SEED)
import matplotlib.pyplot as plt

def visualize_champ(robot: CoreModule, individual: Individual, correct_for_bounding: bool, spawn_z: float, spawn_xy: list[float], path_settings: list[Any], brain_type: str, mode: str = "video") -> list[list[float]]:
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
        weights = individual.brain_genotype
        cpg = CPGSensoryFeedback(num_neurons=int(model.nu), sensory_term=0.0, _lambda=0.01, coupling_weights=weights)
        cpg.reset()
        ctrl = Controller(controller_callback_function=sf_policy, time_steps_per_ctrl_step=1, tracker=tracker)
    else:
        cpg = None
        ctrl = Controller(controller_callback_function=lambda m, d, _: None, time_steps_per_ctrl_step=1, tracker=tracker)
    # ---------------------------------------------------------------

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg))  # type: ignore

    sim_duration = 5.0
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
        video_recorder = VideoRecorder(output_folder=path_to_video_folder)
        step_and_collect(steps)
        video_renderer(model, data, duration=sim_duration, video_recorder=video_recorder)
    elif mode == "launcher":
        warmup_steps = int(1.0 / timestep)
        step_and_collect(warmup_steps)
        viewer.launch(model, data)
    else:
        console.log(f"Mode '{mode}' not recognized. No simulation run.")

    return history

def show_xpos_history(
    history: list[float],
    spawn_position: list[float],
    target_position: list[float],
    *,
    save: bool = True,
    show: bool = True,
) -> None:
    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena(
        load_precompiled=False,
    )

    # Add some objects to the world
    start_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="green_sphere"
            size=".1"
            rgba="0 1 0 1"/>
        </worldbody>
    </mujoco>
    """
    end_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="red_sphere"
            size=".1"
            rgba="1 0 0 1"/>
        </worldbody>
    </mujoco>
    """
    target_box = r"""
    <mujoco>
        <worldbody>
            <geom name="magenta_box"
                size=".1 .1 .1"
                type="box"
                rgba="1 0 1 0.75"/>
        </worldbody>
    </mujoco>
    """
    spawn_box = r"""
    <mujoco>
        <worldbody>
            <geom name="gray_box"
            size=".1 .1 .1"
            type="box"
            rgba="0.5 0.5 0.5 0.5"/>
        </worldbody>
    </mujoco>
    """
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Starting point of robot
    adjustment = np.array((0, 0, target_position[2] + 1))
    world.spawn(
        mj.MjSpec.from_string(start_sphere),
        position=pos_data[0] + (adjustment * 1.5),
        correct_collision_with_floor=False,
    )

    # End point of robot
    world.spawn(
        mj.MjSpec.from_string(end_sphere),
        position=pos_data[-1] + (adjustment * 2),
        correct_collision_with_floor=False,
    )

    # Target position
    world.spawn(
        mj.MjSpec.from_string(target_box),
        position=target_position + adjustment,
        correct_collision_with_floor=False,
    )

    # Spawn position of robot
    world.spawn(
        mj.MjSpec.from_string(spawn_box),
        position=spawn_position,
        correct_collision_with_floor=False,
    )

    # Draw the path of the robot
    smooth = np.linspace(0, 1, len(pos_data))
    inv_smooth = 1 - smooth
    smooth_rise = np.linspace(1.25, 1.95, len(pos_data))
    for i in range(1, len(pos_data)):
        # Get the two points to draw the distance between
        pos_i = pos_data[i]
        pos_j = pos_data[i - 1]

        # Size of the box to represent the distance
        distance = pos_i - pos_j
        minimum_size = 0.05
        geom_size = np.array([
            max(abs(distance[0]) / 2, minimum_size),
            max(abs(distance[1]) / 2, minimum_size),
            max(abs(distance[2]) / 2, minimum_size),
        ])
        geom_size_str: str = f"{geom_size[0]} {geom_size[1]} {geom_size[2]}"

        # Position the box in the middle of the two points
        half_way_point = (pos_i + pos_j) / 2
        geom_pos_str = (
            f"{half_way_point[0]} {half_way_point[1]} {half_way_point[2]}"
        )

        # Smooth color transition from green to red
        geom_rgba = f"{smooth[i]} {inv_smooth[i]} 0 0.75"
        path_box = rf"""
        <mujoco>
            <worldbody>
                <geom name="yellow_sphere"
                    type="box"
                    pos="{geom_pos_str}"
                    size="{geom_size_str}"
                    rgba="{geom_rgba}"
                />
            </worldbody>
        </mujoco>
        """
        world.spawn(
            mj.MjSpec.from_string(path_box),
            position=(adjustment * smooth_rise[i]),
            correct_collision_with_floor=False,
        )

    # Setup the plot
    _, ax = plt.subplots()

    # Add legend to the plot
    plt.rc("legend", fontsize="small")
    red_patch = mpatches.Patch(color="red", label="End Position")
    gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
    green_patch = mpatches.Patch(color="green", label="Start Position")
    magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
    yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
    ax.legend(
        handles=[
            green_patch,
            red_patch,
            magenta_patch,
            gray_patch,
            yellow_patch,
        ],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )

    # Add labels and title
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Title
    plt.title("Robot Path in XY Plane")

    # Render the background image
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
        width=200,
        height=600,
        cam_fovy=8,
        cam_pos=[2.1, 0, 50],
        cam_quat=[-0.7071, 0, 0, 0.7071],
    )

    # Setup background image
    img = plt.imread(save_path)
    ax.imshow(img)

    # Save the figure
    if save:
        fig_path = DATA / "robot_path.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Show results
    if show:
        plt.show()
# def show_xpos_history(history: list[float]) -> None:
#     # Create a tracking camera
#     camera = mj.MjvCamera()
#     camera.type = mj.mjtCamera.mjCAMERA_FREE
#     camera.lookat = [2.5, 0, 0]
#     camera.distance = 10
#     camera.azimuth = 0
#     camera.elevation = -90

#     # Initialize world to get the background
#     mj.set_mjcb_control(None)
#     world = SimpleFlatWorld()
#     model = world.spec.compile()
#     data = mj.MjData(model)
#     save_path = str(DATA / "background.png")
#     single_frame_renderer(
#         model,
#         data,
#         save_path=save_path,
#         save=True,
#     )

#     # Setup background image
#     img = plt.imread(save_path)
#     _, ax = plt.subplots()
#     ax.imshow(img)
#     w, h, _ = img.shape

#     # Convert list of [x,y,z] positions to numpy array
#     pos_data = np.array(history)

#     # Calculate initial position
#     x0, y0 = int(h * 0.483), int(w * 0.815)
#     xc, yc = int(h * 0.483), int(w * 0.9205)
#     ym0, ymc = 0, custom_xy[0]

#     # Convert position data to pixel coordinates
#     pixel_to_dist = -((ymc - ym0) / (yc - y0))
#     pos_data_pixel = [[xc, yc]]
#     for i in range(len(pos_data) - 1):
#         xi, yi, _ = pos_data[i]
#         xj, yj, _ = pos_data[i + 1]
#         xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
#         xn, yn = pos_data_pixel[i]
#         pos_data_pixel.append([xn + int(xd), yn + int(yd)])
#     pos_data_pixel = np.array(pos_data_pixel)

#     # Plot x,y trajectory
#     ax.plot(x0, y0, "kx", label="[0, 0, 0]")
#     ax.plot(xc, yc, "go", label="Start")
#     ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
#     ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

#     # Add labels and title
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     ax.legend()

#     # Title
#     plt.title("Robot Path in XY Plane")

#     # Show results
#     plt.savefig(DATA / "robot_path.png")
#     print("trajectory plot saved")




# def show_qpos_history(history: list[list[float]]) -> None:
#     pos_data = np.array(history, dtype=float)
#     if pos_data.ndim != 2 or pos_data.shape[1] < 2:
#         console.log("[red]History format invalid for plotting.")
#         return
#     # Drop non-finite rows to avoid plotting NaNs
#     pos_data = pos_data[np.isfinite(pos_data).all(axis=1)]
#     if pos_data.size == 0:
#         console.log("[yellow]No finite samples to plot.")
#         return
#     plt.figure(figsize=(10, 6))
#     plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
#     plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
#     plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.title("Robot Path in XY Plane")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def plot_fitness_over_generations(
    experiment_data: dict[str, Any],
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
        "generations" not in experiment_data
        or len(experiment_data["generations"]) == 0
    ):
        console.log("No generation data found in experiment data.")
        return

    generations = []
    max_fitness = []
    avg_fitness = []
    min_fitness = []

    for gen_data in experiment_data["generations"]:
        generations.append(gen_data["generation"])

        # Check if the population_name exists in this generation
        if population_name in gen_data:
            stats = gen_data[population_name]
            max_fitness.append(stats["max"])
            avg_fitness.append(stats["mean"])
            min_fitness.append(stats["min"])
        else:
            msg = (
                f"Population '{population_name}' "
                f"not found in generation {gen_data['generation']}"
            )
            console.log(msg)
            return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        generations, max_fitness, "b-",
        marker="o", markersize=4, label="Max", linewidth=2,
    )
    plt.plot(
        generations, avg_fitness, "purple",
        marker="s", markersize=3, label="Average", linewidth=2,
    )
    plt.plot(
        generations, min_fitness, "gray",
        marker="^", markersize=3, label="Min", linewidth=1.5,
    )

    plt.xlabel("Generation No.", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    title = f"Fitness over Generations - {population_name.replace('_', ' ').title()}"
    plt.title(title, fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(visible=True, alpha=0.3)
    
    # I force integer ticks on x-axis (half generations don't make sense to me)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()

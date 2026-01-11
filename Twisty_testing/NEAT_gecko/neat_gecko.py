# NEAT-based optimization for robot control (Gecko variants)

from __future__ import annotations

import os
import time
from typing import Callable

import mujoco as mj
import numpy as np
from mujoco import viewer
import neat

# Import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_untwisted import (
    gecko_untwisted,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_good import gecko_good
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_front import gecko_front
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist import (
    gecko_doubletwist,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist_turtle import (
    gecko_doubletwist_turtle,
)

# Local libraries
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller

from neat_gecko_helper import fitness_function_basic, create_file_name


# --- RANDOM / EXPERIMENT SETUP --- #
SEED = 42
STARTING_POSITION = [0, 0, 0.1]
OPTIMIZER_NAME = "neat"

# NEAT run settings
GENERATIONS = 50
CONFIG_FILE_NAME = "neat_gecko_config.txt"


def _build_world(robot_model: Callable):
    """Create world, spawn robot, compile model, and attach tracker."""
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    core = robot_model()
    world.spawn(
        core.spec,
        spawn_position=STARTING_POSITION.copy(),
        correct_for_bounding_box=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)  # type: ignore

    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )
    tracker.setup(world.spec, data)

    return world, model, data, tracker


def _make_control_fn(net):
    """Wrap a NEAT network into a controller callback function."""

    def control_fn(model, data):  # signature expected by Controller
        # Basic input: generalized positions (consistent with prior setup)
        inputs = np.asarray(data.qpos, dtype=np.float32)
        outputs = np.asarray(net.activate(inputs), dtype=np.float32)
        # Enforce tanh output range [-1, 1] regardless of internal activations
        outputs = np.tanh(outputs)
        return outputs

    return control_fn


def run(
    robot_model: Callable,
    net,
    mode: str = "simple",
    video_name: str | None = None,
) -> float:
    """Run simulation using a NEAT network; return forward fitness."""
    if mode not in {"viewer", "simple", "video"}:
        raise ValueError(
            f"Mode {mode} not recognized. Choose from 'viewer', 'simple', or 'video'."
        )
    if mode == "video" and video_name is None:
        raise ValueError("In 'video' mode, 'video_name' must be provided.")

    world, model, data, tracker = _build_world(robot_model)

    # Reset network state before each rollout (for recurrent nets)
    if hasattr(net, "reset"):
        net.reset()

    ctrl = Controller(
        controller_callback_function=_make_control_fn(net),
        tracker=tracker,
    )

    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    match mode:
        case "viewer":
            viewer.launch(model=model, data=data)
        case "simple":
            simple_runner(model=model, data=data, duration=15)
        case "video":
            PATH_TO_VIDEO_FOLDER = "./__videos__"
            video_recorder = VideoRecorder(
                output_folder=PATH_TO_VIDEO_FOLDER,
                file_name=video_name,
            )
            video_recorder._add_timestamp_to_file_name = False
            tracking_video_renderer(
                model=model,
                data=data,
                duration=30,
                video_recorder=video_recorder,
            )
        case _:
            raise ValueError(f"Mode {mode} not recognized.")

    history = tracker.history["xpos"][0]
    fitness = fitness_function_basic(history)
    return fitness


class BestFitnessReporter(neat.reporting.BaseReporter):
    """Collect best fitness per generation."""

    def __init__(self) -> None:
        self.best_history: list[float] = []

    def post_evaluate(self, config, population, species, best_genome):
        self.best_history.append(float(getattr(best_genome, "fitness", 0.0)))


def run_neat_experiment(
    seed: int,
    gecko_model: Callable = gecko,
    experiment_name: str = "neat_frwd",
    generations: int = GENERATIONS,
) -> list[float]:
    # Build world once to get IO sizes
    _, model, data, _ = _build_world(gecko_model)
    nn_input_size = len(data.qpos)
    nn_output_size = model.nu

    # Load and patch config
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, CONFIG_FILE_NAME)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Patch input/output sizes to match Mujoco model
    gc = config.genome_config
    gc.num_inputs = nn_input_size
    gc.num_outputs = nn_output_size
    gc.input_keys = [-i for i in range(1, nn_input_size + 1)]
    gc.output_keys = [i for i in range(nn_output_size)]

    # Set RNG seed (for reproducibility)
    rng = np.random.RandomState(seed)
    config.random_state = rng

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    best_reporter = BestFitnessReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)
    pop.add_reporter(best_reporter)

    def eval_genomes(genomes, cfg):
        for gid, genome in genomes:
            if not cfg.genome_config.feed_forward:
                net = neat.nn.RecurrentNetwork.create(genome, cfg)
            else:
                net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            genome.fitness = run(gecko_model, net, mode="simple")

    start = time.time()
    winner = pop.run(eval_genomes, generations)
    elapsed_min = (time.time() - start) / 60.0
    print(f"NEAT optimization ran for {elapsed_min:.2f} minutes.")

    # Replay best genome as video
    if not config.genome_config.feed_forward:
        best_net = neat.nn.RecurrentNetwork.create(winner, config)
    else:
        best_net = neat.nn.FeedForwardNetwork.create(winner, config)

    save_file_name = create_file_name(
        experiment_name=experiment_name,
        robot=gecko_model,
        optimizer_name=OPTIMIZER_NAME,
        generations=generations,
        pop_size=config.pop_size,
    )
    print("Replaying best genome in video mode...")
    _ = run(robot_model=gecko_model, net=best_net, mode="video", video_name=save_file_name)

    return best_reporter.best_history


if __name__ == "__main__":
    num_runs_per_experiment = 1
    experiment_name = "neat_frwd"

    for gecko_type in [
        gecko_doubletwist_turtle,
        # gecko,
        # gecko_untwisted,
        # gecko_good,
        # gecko_doubletwist,
        # gecko_front,
    ]:
        histories: list[list[float]] = []

        for run_idx in range(num_runs_per_experiment):
            seed = SEED
            print("\n" + "=" * 50)
            print(f"Running NEAT for {gecko_type.__name__}, run {run_idx + 1}")
            print("-" * 50)
            print(f"HYPERPARAMETERS:")
            print(f"  Generations: {GENERATIONS}")
            print(f"  Config: {CONFIG_FILE_NAME}")
            print(f"  Seed: {seed}")
            print("-" * 50)
            histories.append(
                run_neat_experiment(
                    seed=seed, gecko_model=gecko_type, experiment_name=experiment_name
                )
            )

        # Save fitness history to JSON
        os.makedirs("./__data__", exist_ok=True)
        out_path = f"./__data__/{gecko_type.__name__}_fitnesses_neat.json"
        try:
            import json

            with open(out_path, "w") as f:
                json.dump(histories, f)
            print(f"Saved fitness histories to {out_path}")
        except Exception as exc:
            print(f"Could not save fitness histories: {exc}")

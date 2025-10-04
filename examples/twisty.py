"""Evolutionary robotics experiment comparing twisty vs non-twisty robots.

This script implements an evolutionary algorithm to compare the performance
of modular robots with twisted orientations (45°, 135°, 225°, 315° rotations)
against standard orthogonal orientations in locomotion tasks.

Author:     Noah, Kacper, Kayra
Date:       2025-01-10
Py Ver:     3.12
OS:         macOS  Sequoia 15.6.1
Hardware:   M4 Pro
Status:     In progress ⚙️

Notes
-----
- Uses tournament selection with elitism
- Evolves probability matrices for body morphology
- Fitness based on distance traveled
- Separate populations for twisty vs non-twisty robots

References
----------
[1] Evolutionary robotics and modular robot morphology evolution

Todo
----
[ ] Add more sophisticated fitness functions
[ ] Implement co-evolution of brain and body?
[x] Add statistical analysis of results

"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import json
import mujoco
import numpy as np
from rich.console import Console
from mujoco import viewer

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder, save_graph_as_json
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.cpg_with_sensory_feedback import (
    CPGSensoryFeedback
)
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
# from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec.a001 import Individual
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner
# from twisty_brain import RobotBrain
# from evotorch.neuroevolution import NEProblem
# from evotorch.algorithms import PGPE
# from evotorch.logging import PandasLogger
from ariel.ec.a000 import IntegerMutator
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

# Global variables
SPAWN_POS = [0, 0, 0.1]
NUM_OF_MODULES = 30
# TARGET_POSITION = [5, 0, 0.5]

# def fitness_function_olympics(history: list[tuple[float, float, float]]) -> float:
#     """Calculate fitness based on robot's trajectory history.

#     xt, yt, zt = TARGET_POSITION
#     xc, yc, zc = history[-1]

#     # Minimize the distance --> maximize the negative distance
#     cartesian_distance = np.sqrt(
#         (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
#     )
#     return -cartesian_distance

def fitness_function_basic(history: list[float]) -> float:
    xs, ys, _ = SPAWN_POS
    xe, ye, _ = history[-1]

    # maximize the distance
    cartesian_distance = np.sqrt(
        (xs - xe) ** 2 + (ys - ye) ** 2,
    )
    return cartesian_distance

def create_individual(con_twisty: bool) -> Individual:
    ind = Individual()
    num_modules = 4

    # "Type" probability space - bias towards HINGE modules for functional robots
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )
    
    # Bias the probabilities to favor HINGE modules over NONE
    # CORE=0, BRICK=1, HINGE=2, NONE=3
    for i in range(num_modules):
        if i == 0:  # Keep core module as is (will be set correctly by decoder)
            continue
        # Boost HINGE probability and reduce NONE probability
        type_probability_space[i, 2] *= 3.0  # HINGE gets 3x weight
        type_probability_space[i, 1] *= 2.0  # BRICK gets 2x weight  
        type_probability_space[i, 3] *= 0.1  # NONE gets very low weight

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
        #Hmmm, this seems incorrect btw
        for i in rotation_probability_space:
            i[TWIST_I] = [0]*4 #This assigns a list, not individual values

    # Store probability matrices in tags for evolution
    ind.tags = {
        "type_probs": type_probability_space.tolist(),
        "conn_probs": conn_probability_space.tolist(),
        "rotation_probs": rotation_probability_space.tolist()
    }

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )
    ind.genotype = graph
    ind.twisty = con_twisty
    return ind


def create_individual_from_matrices(
    type_probs: np.ndarray, 
    conn_probs: np.ndarray, 
    rotation_probs: np.ndarray, 
    twisty: bool
) -> Individual:
    """Create individual from probability matrices."""
    ind = Individual()
    
    # Store in tags
    ind.tags = {
        "type_probs": type_probs.tolist(),
        "conn_probs": conn_probs.tolist(),
        "rotation_probs": rotation_probs.tolist()
    }
    
    # Decode to graph
    hpd = HighProbabilityDecoder(len(type_probs))
    graph = hpd.probability_matrices_to_graph(type_probs, conn_probs, rotation_probs)
    ind.genotype = graph
    ind.twisty = twisty
    
    return ind


def mutate_individual(
    individual: Individual, mutation_rate: float = 0.1
) -> Individual:
    """Mutate probability matrices using ARIEL's float_creep mutation.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation for each matrix type
        
    Returns:
        New mutated individual
    """

    
    # Extract probability matrices from tags
    type_probs = individual.tags["type_probs"]
    conn_probs = individual.tags["conn_probs"]
    rotation_probs = individual.tags["rotation_probs"]
    
    # Apply ARIEL's float_creep mutations
    if RNG.random() < mutation_rate:
        type_probs = IntegerMutator.float_creep(
            individual=type_probs,
            span=0.1,
            mutation_probability=0.3,
        )
        # Ensure values stay in [0, 1] range
        type_probs = np.clip(type_probs, 0, 1).tolist()
    
    if RNG.random() < mutation_rate:
        conn_probs = IntegerMutator.float_creep(
            individual=conn_probs,
            span=0.1,
            mutation_probability=0.3,
        )
        conn_probs = np.clip(conn_probs, 0, 1).tolist()
    
    if RNG.random() < mutation_rate:
        rotation_probs = IntegerMutator.float_creep(
            individual=rotation_probs,
            span=0.1,
            mutation_probability=0.3,
        )
        rotation_probs = np.clip(rotation_probs, 0, 1).tolist()
        
        # Enforce twisty constraint for non-twisty individuals
        if not individual.twisty:
            rotation_probs = np.array(rotation_probs) 
            for i in range(rotation_probs.shape[0]):
                rotation_probs[i, TWIST_I] = 0
    
    # Create new individual with mutated matrices
    return create_individual_from_matrices(
        np.array(type_probs),
        np.array(conn_probs), 
        np.array(rotation_probs),
        individual.twisty,
    )


def tournament_selection(population: list[Individual], tournament_size: int = 3) -> Individual:
    """Tournament selection."""
    tournament = RNG.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda ind: ind.fitness)


def evolve_generation(population: list[Individual], 
                     mutation_rate: float = 0.1,
                     elitism: int = 1) -> list[Individual]:
    """Evolve one generation with tournament selection and elitism."""
    
    # Sort by fitness for elitism
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Next generation
    new_population = []
    
    # Elitism - keep best individual(s)
    new_population.extend(population[:elitism])
    
    # Generate offspring through mutation
    while len(new_population) < len(population):
        parent = tournament_selection(population)
        child = mutate_individual(parent, mutation_rate)
        new_population.append(child)
    
    return new_population


def run_for_fitness(robot: CoreModule, individual: Individual) -> float:
    """Modified run function that returns fitness instead of rendering video."""
    
    # Setup (same as existing run())
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5
    
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    
    # Create CPG controller
    weight_matrix = RNG.uniform(-0.1, 0.1, size=(model.nu, model.nu))
    cpg = CPGSensoryFeedback(
        num_neurons=int(model.nu),
        sensory_term=-0.0,
        _lambda=0.01,
        coupling_weights=weight_matrix,
    )
    cpg.reset()
    individual.brain_genotype = cpg.c
    mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg=cpg))
    
    # Record initial position
    initial_pos = data.xpos[1][:2].copy()
    
    # Run simulation for fitness (no video)
    simulation_time = 30.0  # seconds
    steps = int(simulation_time / model.opt.timestep)
    
    for _ in range(steps):
        mujoco.mj_step(model, data)
    
    # Calculate distance fitness
    final_pos = data.xpos[1][:2]
    fitness = np.linalg.norm(final_pos - initial_pos)
    
    return fitness

def evaluate_population(population: list[Individual]) -> None:
    """Evaluate fitness for all individuals in population."""
    for individual in population:
        try:
            robot = construct_mjspec_from_graph(individual.genotype)
            fitness = run_for_fitness(robot, individual)
            individual.fitness = fitness
            console.log(f"Individual (twisty={individual.twisty}) fitness: {fitness:.3f}")
        except Exception as e:
            console.log(f"Error evaluating individual: {e}")
            individual.fitness = 0.0


def initialize_experiment_data(generations: int, population_size: int) -> dict:
    """Initialize experiment tracking data structure."""
    return {
        "parameters": {
            "generations": generations,
            "population_size": population_size,
            "mutation_rate": 0.3,
            "tournament_size": 2,
            "elitism": True,
        },
        "generations": [],
        "final_statistics": {},
        "champions": {},
    }


def calculate_generation_statistics(
    twisty_population: list[Individual], 
    non_twisty_population: list[Individual], 
    generation: int
) -> dict:
    """Calculate comprehensive statistics for a generation."""
    twisty_fitnesses = [ind.fitness for ind in twisty_population]
    non_twisty_fitnesses = [ind.fitness for ind in non_twisty_population]
    
    return {
        "generation": generation + 1,
        "twisty": {
            "mean": float(np.mean(twisty_fitnesses)),
            "std": float(np.std(twisty_fitnesses)),
            "min": float(min(twisty_fitnesses)),
            "max": float(max(twisty_fitnesses)),
            "median": float(np.median(twisty_fitnesses)),
            "q25": float(np.percentile(twisty_fitnesses, 25)),
            "q75": float(np.percentile(twisty_fitnesses, 75)),
            "all_fitnesses": [float(f) for f in twisty_fitnesses],
        },
        "non_twisty": {
            "mean": float(np.mean(non_twisty_fitnesses)),
            "std": float(np.std(non_twisty_fitnesses)),
            "min": float(min(non_twisty_fitnesses)),
            "max": float(max(non_twisty_fitnesses)),
            "median": float(np.median(non_twisty_fitnesses)),
            "q25": float(np.percentile(non_twisty_fitnesses, 25)),
            "q75": float(np.percentile(non_twisty_fitnesses, 75)),
            "all_fitnesses": [float(f) for f in non_twisty_fitnesses],
        },
    }


def finalize_experiment_data(
    experiment_data: dict,
    best_twisty_fitness: float,
    best_non_twisty_fitness: float,
    champion_type: str,
    champion_fitness: float
) -> None:
    """Calculate final statistics and save experiment data."""
    # Calculate overall statistics
    twisty_all_scores = [
        score for gen in experiment_data["generations"] 
        for score in gen["twisty"]["all_fitnesses"]
    ]
    non_twisty_all_scores = [
        score for gen in experiment_data["generations"] 
        for score in gen["non_twisty"]["all_fitnesses"]
    ]
    
    experiment_data["final_statistics"] = {
        "twisty": {
            "overall_mean": float(np.mean(twisty_all_scores)),
            "overall_std": float(np.std(twisty_all_scores)),
            "overall_min": float(min(twisty_all_scores)),
            "overall_max": float(max(twisty_all_scores)),
            "total_evaluations": len(twisty_all_scores),
            "champion_fitness": float(best_twisty_fitness),
        },
        "non_twisty": {
            "overall_mean": float(np.mean(non_twisty_all_scores)),
            "overall_std": float(np.std(non_twisty_all_scores)),
            "overall_min": float(min(non_twisty_all_scores)),
            "overall_max": float(max(non_twisty_all_scores)),
            "total_evaluations": len(non_twisty_all_scores),
            "champion_fitness": float(best_non_twisty_fitness),
        },
    }
    
    # Calculate performance advantage
    if best_twisty_fitness >= best_non_twisty_fitness:
        if best_non_twisty_fitness > 0:
            performance_advantage = (
                (best_twisty_fitness - best_non_twisty_fitness) 
                / best_non_twisty_fitness
            ) * 100
        else:
            performance_advantage = float("inf")
    else:
        performance_advantage = (
            (best_non_twisty_fitness - best_twisty_fitness) 
            / best_twisty_fitness
        ) * 100
    
    experiment_data["champions"] = {
        "overall_champion": champion_type,
        "champion_fitness": float(champion_fitness),
        "performance_advantage_percent": float(performance_advantage),
        "twisty_champion_fitness": float(best_twisty_fitness),
        "non_twisty_champion_fitness": float(best_non_twisty_fitness),
    }
    
    # Save experiment data
    experiment_filename = f"experiment_data_{champion_type.lower()}_champion.json"
    with open(DATA / experiment_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)
    console.log(f"📈 Complete experiment data saved to: {DATA / experiment_filename}")


def run_evolution_experiment(
    generations: int = 100,
    population_size: int = 100,
    save_evolution_graphs: bool = True,
    sample_diversity_every: int = 10,
) -> tuple[Individual, dict]:
    """Run evolutionary experiment comparing twisty vs non-twisty robots.
    
    Args:
        generations: Number of evolutionary generations to run
        population_size: Size of each population (twisty and non-twisty)  
        save_evolution_graphs: Whether to save champion robot graphs 
        sample_diversity_every: Save population samples (top 3) every N generations 
    
    Returns
    -------
    tuple
        Champion robot and comprehensive experiment data
    """
    console.log("Starting evolutionary experiment...")
    console.log(f"Generations: {generations}, Population size: {population_size}")
    
    # Initialize experiment tracking data
    experiment_data = initialize_experiment_data(generations, population_size)
    
    # Initialize separate populations
    twisty_population = [
        create_individual(con_twisty=True) for _ in range(population_size)
    ]
    non_twisty_population = [
        create_individual(con_twisty=False) for _ in range(population_size)
    ]
    
    # Track best individuals across all generations
    best_twisty_ever = None
    best_non_twisty_ever = None
    best_twisty_fitness = -float('inf')
    best_non_twisty_fitness = -float('inf')
    
    # Evolution loop
    for generation in range(generations):
        console.log(f"\n--- Generation {generation + 1} ---")
        
        # Evaluate fitness for both populations
        console.log("Evaluating twisty population...")
        evaluate_population(twisty_population)
        
        console.log("Evaluating non-twisty population...")
        evaluate_population(non_twisty_population)
        
        # Track best individuals this generation
        current_best_twisty = max(twisty_population, key=lambda x: x.fitness)
        current_best_non_twisty = max(non_twisty_population, key=lambda x: x.fitness)
        
        # Update all-time champions
        if current_best_twisty.fitness > best_twisty_fitness:
            best_twisty_ever = current_best_twisty
            best_twisty_fitness = current_best_twisty.fitness
            console.log(f"[bold blue]NEW TWISTY CHAMPION: {best_twisty_fitness:.3f}")
            
            # Save new champion graph for research analysis (if enabled)
            if save_evolution_graphs:
                champion_dir = DATA / "evolution_champions" / "twisty"
                champion_dir.mkdir(parents=True, exist_ok=True)
                filename = f"gen_{generation + 1}_fitness_{best_twisty_fitness:.3f}.json"
                save_graph_as_json(
                    current_best_twisty.genotype,
                    champion_dir / filename,
                )
            
        if current_best_non_twisty.fitness > best_non_twisty_fitness:
            best_non_twisty_ever = current_best_non_twisty
            best_non_twisty_fitness = current_best_non_twisty.fitness
            console.log(f"[bold red]NEW NON-TWISTY CHAMPION: {best_non_twisty_fitness:.3f}")
            
            # Save new champion graph for research analysis (if enabled)
            if save_evolution_graphs:
                champion_dir = DATA / "evolution_champions" / "non_twisty"
                champion_dir.mkdir(parents=True, exist_ok=True)
                filename = f"gen_{generation + 1}_fitness_{best_non_twisty_fitness:.3f}.json"
                save_graph_as_json(
                    current_best_non_twisty.genotype,
                    champion_dir / filename,
                )
        
        # Calculate comprehensive statistics for scientific analysis
        gen_stats = calculate_generation_statistics(
            twisty_population,
            non_twisty_population,
            generation,
        )

        experiment_data["generations"].append(gen_stats)

        # Optional diversity sampling for morphological studies
        if (
            sample_diversity_every > 0 
            and (generation + 1) % sample_diversity_every == 0
            and save_evolution_graphs
        ):
            diversity_dir = DATA / "population_samples" / f"generation_{generation + 1}"
            diversity_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample top 3 from each population for diversity analysis
            top_twisty = sorted(twisty_population, key=lambda x: x.fitness, reverse=True)[:3]
            top_non_twisty = sorted(non_twisty_population, key=lambda x: x.fitness, reverse=True)[:3]
            
            for i, ind in enumerate(top_twisty):
                filename = f"twisty_top_{i+1}_fitness_{ind.fitness:.3f}.json"
                save_graph_as_json(ind.genotype, diversity_dir / filename)
                
            for i, ind in enumerate(top_non_twisty):
                filename = f"non_twisty_top_{i+1}_fitness_{ind.fitness:.3f}.json"
                save_graph_as_json(ind.genotype, diversity_dir / filename)

        # Report generation statistics
        console.log(f"Generation {generation + 1} Detailed Statistics:")
        twisty_stats = gen_stats["twisty"]
        non_twisty_stats = gen_stats["non_twisty"]
        
        console.log(
            f"  Twisty - Mean: {twisty_stats['mean']:.3f} ± "
            f"{twisty_stats['std']:.3f}"
        )
        console.log(
            f"           Range: [{twisty_stats['min']:.3f}, "
            f"{twisty_stats['max']:.3f}]"
        )
        console.log(
            f"           Median: {twisty_stats['median']:.3f}, "
            f"IQR: [{twisty_stats['q25']:.3f}, {twisty_stats['q75']:.3f}]"
        )
        console.log(
            f"  Non-twisty - Mean: {non_twisty_stats['mean']:.3f} ± "
            f"{non_twisty_stats['std']:.3f}"
        )
        console.log(
            f"               Range: [{non_twisty_stats['min']:.3f}, "
            f"{non_twisty_stats['max']:.3f}]"
        )
        console.log(
            f"               Median: {non_twisty_stats['median']:.3f}, "
            f"IQR: [{non_twisty_stats['q25']:.3f}, "
            f"{non_twisty_stats['q75']:.3f}]"
        )
        
        # Evolve populations (except for last generation)
        if generation < generations - 1:
            twisty_population = evolve_generation(twisty_population)
            non_twisty_population = evolve_generation(non_twisty_population)

    # Determine overall champion
    if best_twisty_fitness >= best_non_twisty_fitness:
        champion = best_twisty_ever
        champion_type = "TWISTY"
        champion_fitness = best_twisty_fitness
    else:
        champion = best_non_twisty_ever
        champion_type = "NON-TWISTY"
        champion_fitness = best_non_twisty_fitness

    # Calculate final experiment statistics and save data
    finalize_experiment_data(
        experiment_data, best_twisty_fitness, best_non_twisty_fitness,
        champion_type, champion_fitness
    )

    console.log(f"[bold green]🏆 OVERALL CHAMPION: {champion_type} robot")
    console.log(f"[bold green]🏆 Champion Fitness: {champion_fitness:.3f}")

    # Save champion graph to file
    champion_filename = f"champion_{champion_type.lower()}_robot.json"
    save_graph_as_json(champion.genotype, DATA / champion_filename)
    console.log(f"📋 Champion robot saved to: {DATA / champion_filename}")
    
    return champion, experiment_data


def main() -> None:
    """Entry point for evolutionary experiment."""
    champion, _ = run_evolution_experiment(
        generations=5,
        population_size=3,
    )
    
    # Render a video of the champion robot
    console.log("\n🎬 Creating champion robot video...")
    champion_robot = construct_mjspec_from_graph(champion.genotype)
    run(champion_robot, champion)


def run(robot: CoreModule, individual: Individual, mode: str = "video") -> None:
    """Entry point."""
    # BugFix -> "Python exception raised"
    mujoco.set_mjcb_control(None)

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

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

    # return fitness_function(tracker.history["xpos"])

def policy(
    model: mujoco.MjModel,  # noqa: ARG001
    data: mujoco.MjData,
    cpg: CPGSensoryFeedback,
) -> np.ndarray:
    """Use feedback term to shift the output of the CPGs."""
    x, _ = cpg.step()
    data.ctrl = x * np.pi / 2


if __name__ == "__main__":
    # Test several times
    main()

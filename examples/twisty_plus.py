"""Evolutionary robotics experiment comparing twisty vs non-twisty robots.

This script implements an evolutionary algorithm to compare the performance
of modular robots with twisted orientations (45°, 135°, 225°, 315° rotations)
against standard orthogonal orientations in locomotion tasks.

Author:     Noah, Kacper, Kayra
Date:       2025-09-10
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
[ ] Make sure that every morphology always has a body (Error evaluating individual: 'CoreModule' object has no attribute 'body'  )

"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
import matplotlib.pyplot as plt
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
    HighProbabilityDecoder, save_graph_as_json, load_graph_from_json
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.sf_cpg import CPGSensoryFeedback, sf_policy
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
# from ariel.simulation.environments import OlympicArena
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
TWIST_I = [1, 3, 5, 7] # We want to experiment with this eventually


# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

# Global variables
EVOLUTION_CONFIG = {
    "generations": 2,
    "population_size": 5,
    "save_evolution_graphs": True,
    "sample_diversity_every": 10,
    "checkpoint_every": 1,  # Save checkpoint every N generations
    "auto_resume": True,    # Automatically resume from checkpoint if found
}
SPAWN_POS = [0, 0, 0.1]  # Low spawn height - bounding box correction will adjust
# NUM_OF_MODULES = 30
TARGET_POSITION = [0, 5, 0.5]


def create_individual(con_twisty: bool) -> Individual:
    ind = Individual()
    num_modules = 20

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
        type_probability_space[i, 2] *= 1.0  # HINGE gets 3x weight
        type_probability_space[i, 1] *= 1.0  # BRICK gets 2x weight  
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
        # If twisty not true, the twisted angles are given the value zero, so they can't be selected.
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
    ind.genotype = [type_probability_space.tolist(), conn_probability_space.tolist(), rotation_probability_space.tolist()]
    ind.graph = graph
    ind.twisty = con_twisty
    return ind


def create_individual_from_matrices(
    type_probs: np.ndarray[Any, Any], 
    conn_probs: np.ndarray[Any, Any], 
    rotation_probs: np.ndarray[Any, Any], 
    twisty: bool
) -> Individual:
    """Create individual from probability matrices."""
    ind = Individual()
    
    # Decode to graph
    hpd = HighProbabilityDecoder(len(type_probs))
    graph = hpd.probability_matrices_to_graph(type_probs, conn_probs, rotation_probs)
    
    ind.genotype = [type_probs.tolist(), conn_probs.tolist(), rotation_probs.tolist()]
    ind.graph = graph
    ind.twisty = twisty
    
    return ind


def float_creep(
    individual: Any,
    span: float,
    mutation_probability: float,
) -> list[float]:
    """Mutate numeric arrays/lists by adding uniform noise within +/- span.

    Accepts array-like input and returns a list of floats with the same shape.
    """
    # Prep
    ind_arr = np.array(individual, dtype=float)
    shape = ind_arr.shape

    # Generate mutation values
    mutator = RNG.uniform(
        low=0,
        high=span,
        size=shape,
    )

    # Include negative mutations
    sub_mask = RNG.choice(
        [-1, 1],
        size=shape,
    )

    # Determine which positions to mutate
    do_mask = RNG.choice(
        [1, 0],
        size=shape,
        p=[mutation_probability, 1 - mutation_probability],
    )
    mutation_mask = mutator * sub_mask * do_mask
    new_genotype = ind_arr + mutation_mask
    return new_genotype.astype(float).tolist()


def mutate_individual(
    individual: Individual, mutation_rate: float = 0.1
) -> Individual:
    """Mutate probability matrices using ARIEL's (old) float_creep mutation.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation for each matrix type
        
    Returns:
        New mutated individual
    """

    
    # Extract probability matrices from tags
    type_probs = individual.genotype[0]
    conn_probs = individual.genotype[1]
    rotation_probs = individual.genotype[2]
    
    # Apply ARIEL's float_creep mutations
    if RNG.random() < mutation_rate:
        type_probs = float_creep(
            individual=type_probs,
            span=0.1,
            mutation_probability=0.3,
        )
        # Ensure values stay in [0, 1] range
        type_probs = np.clip(type_probs, 0, 1).tolist()
    
    if RNG.random() < mutation_rate:
        conn_probs = float_creep(
            individual=conn_probs,
            span=0.1,
            mutation_probability=0.3,
        )
        conn_probs = np.clip(conn_probs, 0, 1).tolist()
    
    if RNG.random() < mutation_rate:
        rotation_probs = float_creep(
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


def crossover_individuals(
    parent1: Individual, 
    parent2: Individual, 
    alpha: float = 0.5,
) -> Individual:
    """Create offspring by crossing over probability matrices from two parents.
    
    Uses Blend Crossover (BLX-alpha) where offspring values are sampled from
    intervals around parent values, allowing exploration beyond parent bounds.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        alpha: Blend factor controlling exploration range (typical: 0.1-0.5)
        
    Returns:
        New individual created through blend crossover
    """
    # Extract probability matrices from both parents
    type_probs1, conn_probs1, rotation_probs1 = parent1.genotype
    type_probs2, conn_probs2, rotation_probs2 = parent2.genotype
    
    # Convert to numpy arrays for easier manipulation
    type1, type2 = np.array(type_probs1), np.array(type_probs2)
    conn1, conn2 = np.array(conn_probs1), np.array(conn_probs2)
    rot1, rot2 = np.array(rotation_probs1), np.array(rotation_probs2)
    
    # Apply blend crossover to each matrix
    child_type = _blend_crossover_matrix(type1, type2, alpha)
    child_conn = _blend_crossover_matrix(conn1, conn2, alpha)
    child_rot = _blend_crossover_matrix(rot1, rot2, alpha)
    
    # Determine twisty status - inherit if either parent is twisty
    twisty = parent1.twisty or parent2.twisty
    
    # Enforce twisty constraint for non-twisty individuals
    if not twisty:
        child_rot[:, TWIST_I] = 0
    
    return create_individual_from_matrices(
        child_type, child_conn, child_rot, twisty
    )


def _blend_crossover_matrix(
    matrix1: np.ndarray[Any, Any], matrix2: np.ndarray[Any, Any], alpha: float,
) -> np.ndarray[Any, Any]:
    """Apply Blend Crossover (BLX-alpha) to two matrices element-wise.
    
    For each gene pair (x1, x2), creates offspring value in interval:
    [min(x1,x2) - alpha*|x1-x2|, max(x1,x2) + alpha*|x1-x2|]
    
    Args:
        matrix1: First parent matrix
        matrix2: Second parent matrix
        alpha: Blend factor controlling exploration range
        
    Returns:
        Offspring matrix with blended values
    """
    # Calculate min, max, and difference for each gene pair
    min_vals = np.minimum(matrix1, matrix2)
    max_vals = np.maximum(matrix1, matrix2)
    diff = np.abs(matrix1 - matrix2)
    
    # Calculate blend interval bounds
    lower_bound = min_vals - alpha * diff
    upper_bound = max_vals + alpha * diff
    
    # Sample uniformly from blend intervals
    child_matrix = RNG.uniform(lower_bound, upper_bound)
    
    # Ensure values stay in [0, 1] range for probability matrices
    child_matrix = np.clip(child_matrix, 0.0, 1.0)
    
    return child_matrix


def evolve_generation(population: list[Individual], 
                                    mutation_rate: float = 0.1,
                                    crossover_rate: float = 0.7,
                                    elitism: int = 1) -> list[Individual]:
    """Evolve one generation with tournament selection and elitism.
    
    Args:
        population: Current population to evolve
        mutation_rate: Probability of mutation for each matrix type
        crossover_rate: Probability of using crossover vs mutation-only
        elitism: Number of best individuals to keep unchanged
        
    Returns:
        New evolved population
    """
    # Sort by fitness for elitism
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Next generation
    new_population = []
    
    # Elitism - keep best individual(s)
    new_population.extend(population[:elitism])
    
    # Generate offspring through crossover and mutation
    while len(new_population) < len(population):
        if RNG.random() < crossover_rate:
            # Crossover: select two parents and create offspring
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover_individuals(parent1, parent2)
            
            # Apply mutation to crossover offspring (optional but recommended)
            if RNG.random() < mutation_rate:
                child = mutate_individual(child, mutation_rate)  # Lower mutation rate for crossover offspring
        else:
            # Mutation only: select one parent and mutate
            parent = tournament_selection(population)
            child = mutate_individual(parent, mutation_rate)
        
        new_population.append(child)
    
    return new_population


def run_for_fitness(robot: CoreModule, individual: Individual) -> float:
    """Modified run function that returns fitness based on distance to target."""
    
    # Setup (same as existing run())
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5
    
    world.spawn(robot.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    
    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")
    
    # Create CPG controller
    adj_dict = create_fully_connected_adjacency(model.nu)
    cpg = NaCPG(adj_dict, angle_tracking=True)
    cpg.reset()
    gen = cpg.get_flat_params()
    individual.brain_genotype = gen
    mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg=cpg))
    
    # Run simulation for target-seeking fitness
    simulation_time = 15.0  # seconds
    steps = int(simulation_time / model.opt.timestep)
    individual.time_alive = 0
    
    for step in range(steps):
        individual.time_alive = step / (1/model.opt.timestep)
        mujoco.mj_step(model, data)
    
    # Calculate fitness based on final distance to target
    final_position = data.xpos[1][:2].copy()  # x, y coordinates only
    target_position = np.array([TARGET_POSITION[0], TARGET_POSITION[1]])  # x, y from target
    
    # Distance to target (lower is better)
    distance_to_target = np.linalg.norm(final_position - target_position)
    
    # Simple inverse distance fitness (higher fitness = closer to target)
    fitness = 1.0 / (1.0 + distance_to_target)
    
    return fitness

def evaluate_population(population: list[Individual]) -> None:
    """Evaluate fitness for all individuals in population."""
    for individual in population:
        try:
            robot = construct_mjspec_from_graph(individual.graph)
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
    mixed_twisty_population: list[Individual], 
    generation: int
) -> dict:
    """Calculate comprehensive statistics for a generation."""
    mixed_twisty_fitnesses = [ind.fitness for ind in mixed_twisty_population]
    
    return {"generation": generation + 1,
        "mixed_twisty": {
            "mean": float(np.mean(mixed_twisty_fitnesses)),
            "std": float(np.std(mixed_twisty_fitnesses)),
            "min": float(min(mixed_twisty_fitnesses)),
            "max": float(max(mixed_twisty_fitnesses)),
            "median": float(np.median(mixed_twisty_fitnesses)),
            "q25": float(np.percentile(mixed_twisty_fitnesses, 25)),
            "q75": float(np.percentile(mixed_twisty_fitnesses, 75)),
            "all_fitnesses": [float(f) for f in mixed_twisty_fitnesses],
        },
    }


def finalize_experiment_data(
    experiment_data: dict,
    best_mixed_twisty_fitness: float,
    champion_type: str,
    champion_fitness: float
) -> None:
    """Calculate final statistics and save experiment data."""
    # Calculate overall statistics
    mixed_twisty_all_scores = [
        score for gen in experiment_data["generations"] 
        for score in gen["mixed_twisty"]["all_fitnesses"]
    ]
    
    experiment_data["final_statistics"] = {
        "non_twisty": {
            "overall_mean": float(np.mean(mixed_twisty_all_scores)),
            "overall_std": float(np.std(mixed_twisty_all_scores)),
            "overall_min": float(min(mixed_twisty_all_scores)),
            "overall_max": float(max(mixed_twisty_all_scores)),
            "total_evaluations": len(mixed_twisty_all_scores),
            "champion_fitness": float(best_mixed_twisty_fitness),
        },
    }
    
    
    experiment_data["champions"] = {
        "overall_champion": champion_type,
        "champion_fitness": float(champion_fitness),
    }
    
    # Save experiment data
    experiment_filename = f"experiment_data_{champion_type.lower()}_champion.json"
    with open(DATA / experiment_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)
    console.log(f"📈 Complete experiment data saved to: {DATA / experiment_filename}")


def save_checkpoint(
    generation: int,
    twisty_population: list[Individual],
    non_twisty_population: list[Individual],
    experiment_data: dict,
    best_twisty_ever: Individual,
    best_non_twisty_ever: Individual,
    best_twisty_fitness: float,
    best_non_twisty_fitness: float,
) -> None:
    """Save complete experiment state to checkpoint file."""
    checkpoint_data = {
        "generation": generation,
        "experiment_data": experiment_data,
        "best_twisty_fitness": best_twisty_fitness,
        "best_non_twisty_fitness": best_non_twisty_fitness,
        "populations": {
            "twisty": [
                {
                    "genotype": ind.genotype,
                    "fitness": ind.fitness,
                    "twisty": ind.twisty,
                } for ind in twisty_population
            ],
            "non_twisty": [
                {
                    "genotype": ind.genotype,
                    "fitness": ind.fitness,
                    "twisty": ind.twisty,
                } for ind in non_twisty_population
            ],
        },
        "champions": {
            "twisty": {
                "genotype": best_twisty_ever.genotype if best_twisty_ever else None,
                "fitness": best_twisty_fitness,
                "twisty": True,
            } if best_twisty_ever else None,
            "non_twisty": {
                "genotype": best_non_twisty_ever.genotype if best_non_twisty_ever else None,
                "fitness": best_non_twisty_fitness,
                "twisty": False,
            } if best_non_twisty_ever else None,
        },
        "timestamp": str(np.datetime64('now')),
        "seed": SEED,
    }
    
    checkpoint_file = DATA / f"checkpoint_gen_{generation}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    console.log(f"💾 Checkpoint saved: {checkpoint_file}")


def load_checkpoint(
    checkpoint_file: Path,
) -> tuple[int, list[Individual], list[Individual], dict, Individual, Individual, float, float]:
    """Load experiment state from checkpoint file."""
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)
    
    generation = checkpoint_data["generation"]
    experiment_data = checkpoint_data["experiment_data"]
    best_twisty_fitness = checkpoint_data["best_twisty_fitness"]
    best_non_twisty_fitness = checkpoint_data["best_non_twisty_fitness"]
    
    # Reconstruct populations
    twisty_population = []
    for ind_data in checkpoint_data["populations"]["twisty"]:
        individual = create_individual_from_matrices(
            np.array(ind_data["genotype"][0]),
            np.array(ind_data["genotype"][1]),
            np.array(ind_data["genotype"][2]),
            ind_data["twisty"]
        )
        individual.fitness = ind_data["fitness"]
        twisty_population.append(individual)
    
    non_twisty_population = []
    for ind_data in checkpoint_data["populations"]["non_twisty"]:
        individual = create_individual_from_matrices(
            np.array(ind_data["genotype"][0]),
            np.array(ind_data["genotype"][1]),
            np.array(ind_data["genotype"][2]),
            ind_data["twisty"]
        )
        individual.fitness = ind_data["fitness"]
        non_twisty_population.append(individual)
    
    # Reconstruct champions
    best_twisty_ever = None
    if checkpoint_data["champions"]["twisty"]:
        champ_data = checkpoint_data["champions"]["twisty"]
        best_twisty_ever = create_individual_from_matrices(
            np.array(champ_data["genotype"][0]),
            np.array(champ_data["genotype"][1]),
            np.array(champ_data["genotype"][2]),
            champ_data["twisty"]
        )
        best_twisty_ever.fitness = champ_data["fitness"]
    
    best_non_twisty_ever = None
    if checkpoint_data["champions"]["non_twisty"]:
        champ_data = checkpoint_data["champions"]["non_twisty"]
        best_non_twisty_ever = create_individual_from_matrices(
            np.array(champ_data["genotype"][0]),
            np.array(champ_data["genotype"][1]),
            np.array(champ_data["genotype"][2]),
            champ_data["twisty"]
        )
        best_non_twisty_ever.fitness = champ_data["fitness"]
    
    console.log(f"📂 Checkpoint loaded: generation {generation}")
    return (generation, twisty_population, non_twisty_population, experiment_data, 
            best_twisty_ever, best_non_twisty_ever, best_twisty_fitness, best_non_twisty_fitness)


def find_latest_checkpoint() -> Path | None:
    """Find the most recent checkpoint file."""
    checkpoint_pattern = "checkpoint_gen_*.json"
    checkpoints = list(DATA.glob(checkpoint_pattern))
    
    if not checkpoints:
        return None
    
    # Sort by generation number
    def get_gen_num(path: Path) -> int:
        return int(path.stem.split('_')[-1])
    
    latest = max(checkpoints, key=get_gen_num)
    console.log(f"🔍 Found latest checkpoint: {latest}")
    return latest


def run_evolution_experiment(
    generations: int = EVOLUTION_CONFIG['generations'],
    population_size: int = EVOLUTION_CONFIG['population_size'],
    save_evolution_graphs: bool = EVOLUTION_CONFIG['save_evolution_graphs'],
    sample_diversity_every: int = EVOLUTION_CONFIG['sample_diversity_every'],
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
    mixed_twisty_population = [
        create_individual(con_twisty=False) for _ in range(population_size)
    ]
    
    twisty_spawnrate = 0.5
    twisty_maxspawn = int(population_size/5)
    # Track best individuals across all generations
    best_mixed_twisty_ever = None
    best_mixed_twisty_fitness = -float('inf')
    twisty_rate = twisty_spawnrate/generations
    # Evolution loop
    for generation in range(generations):
        console.log(f"\n--- Generation {generation + 1} ---")
        
        # Evaluate fitness for both populations
        # console.log("Evaluating twisty population...")
        # evaluate_population(twisty_population)
        twisty_p = twisty_rate*generation
        for i in range(twisty_maxspawn):
            chance = RNG.random()
            if chance <= twisty_p:
                mixed_twisty_population.append(create_individual(True))
        console.log("Evaluating mixed-twisty population...")
        evaluate_population(mixed_twisty_population)
        
        # Track best individuals this generation
        current_best_mixed_twisty = max(mixed_twisty_population, key=lambda x: x.fitness)
            
        if current_best_mixed_twisty.fitness > best_mixed_twisty_fitness:
            best_mixed_twisty_ever = current_best_mixed_twisty
            best_mixed_twisty_fitness = current_best_mixed_twisty.fitness
            console.log(f"[bold red]NEW MIXED-TWISTY CHAMPION: {best_mixed_twisty_fitness:.3f}")
            
            # Save new champion graph for research analysis (if enabled)
            if save_evolution_graphs:
                champion_dir = DATA / "evolution_champions" / "Mixed_twisty"
                champion_dir.mkdir(parents=True, exist_ok=True)
                filename = f"gen_{generation + 1}_fitness_{best_mixed_twisty_fitness:.3f}.json"
                save_graph_as_json(
                    current_best_mixed_twisty.graph,
                    champion_dir / filename,
                )
        
        # Calculate comprehensive statistics for scientific analysis
        gen_stats = calculate_generation_statistics(
            mixed_twisty_population,
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
            top_mixed_twisty = sorted(mixed_twisty_population, key=lambda x: x.fitness, reverse=True)[:3]
                
            for i, ind in enumerate(top_mixed_twisty):
                filename = f"mixed_twisty_top_{i+1}_fitness_{ind.fitness:.3f}.json"
                save_graph_as_json(ind.graph, diversity_dir / filename)

        # Report generation statistics
        console.log(f"Generation {generation + 1} Detailed Statistics:")
        mixed_twisty_stats = gen_stats["mixed_twisty"]
        
        console.log(
            f"  mixed-twisty - Mean: {mixed_twisty_stats['mean']:.3f} ± "
            f"{mixed_twisty_stats['std']:.3f}"
        )
        console.log(
            f"               Range: [{mixed_twisty_stats['min']:.3f}, "
            f"{mixed_twisty_stats['max']:.3f}]"
        )
        console.log(
            f"               Median: {mixed_twisty_stats['median']:.3f}, "
            f"IQR: [{mixed_twisty_stats['q25']:.3f}, "
            f"{mixed_twisty_stats['q75']:.3f}]"
        )
        
        # Evolve populations (except for last generation)
        if generation < generations - 1:
            mixed_twisty_population = evolve_generation(mixed_twisty_population)

    champion = best_mixed_twisty_ever
    if champion.twisty is True:
        champion_type = "TWISTY"
    else:
        champion_type = "NON-TWISTY"
    champion_fitness = best_mixed_twisty_fitness
    # Calculate final experiment statistics and save data
    finalize_experiment_data(
        experiment_data, best_mixed_twisty_fitness,
        champion_type, champion_fitness
    )

    console.log(f"[bold green]🏆 OVERALL CHAMPION: {champion_type} robot")
    console.log(f"[bold green]🏆 Champion Fitness: {champion_fitness:.3f}")

    # Save champion graph to file
    champion_filename = f"champion_{champion_type.lower()}_robot.json"
    save_graph_as_json(champion.graph, DATA / champion_filename)
    console.log(f"📋 Champion robot saved to: {DATA / champion_filename}")
    
    return champion, experiment_data

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
    population_name = "twisty"
    plt.xlabel("Generation No.", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    title = f"Fitness over Generations - {population_name.replace('_', ' ').title()}"
    plt.title(title, fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(visible=True, alpha=0.3)
    
    # I force integer ticks on x-axis (half generations don't make sense to me)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # type: ignore
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Entry point for evolutionary experiment."""
    filename = f"champ_graph_normal.json"
    graph = load_graph_from_json(DATA / filename)
    data_file = "experiment_data_twisty_champion.json"
    load_data = DATA / data_file 
    # Visualize the champion robot - change mode here!
    console.log("\n🏆 Visualizing champion robot...")
    champion_robot = construct_mjspec_from_graph(graph)
    brain_filename = "champ_brain_normal.json"
    load_file = DATA / brain_filename
    with Path(load_file).open("r", encoding="utf-8") as f:
        brain = json.load(f) 
    with Path(load_data).open("r", encoding="utf-8") as f:
        data = json.load(f)
    plot_fitness_over_generations(data, "mixed_twisty")
    # Change mode to "launcher" to see robot in interactive viewer
    # Change mode to "video" to record a video
    run(champion_robot, brain, mode="video")


def run(robot: CoreModule, brain,  mode: str = "video") -> None: #Visualizes the champion robot after evolution
    """Entry point."""
    # BugFix -> "Python exception raised"
    mujoco.set_mjcb_control(None)

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec, position=SPAWN_POS) #read as champion_robot.spec. it's just being accessed through the parameter name robot.

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)
    
    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Save the model to XML
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # Actuators and CPG
    mujoco.set_mjcb_control(None)
    weights = brain
    cpg = CPGSensoryFeedback(num_neurons=int(model.nu), sensory_term=0.0, _lambda=0.01, coupling_weights=weights)
    cpg.reset()

    # Initialize robot tracker
    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )
    tracker.setup(world.spec, data)

    # # Initialize controller
    # ctrl = Controller(
    #     controller_callback_function=policy,
    #     time_steps_per_ctrl_step=1,
    #     tracker=tracker,
    # )

    # mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg))

    # # Initialize robot tracker
    # mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    # name_to_bind = "core"
    # tracker = Tracker(
    #     mujoco_obj_to_find=mujoco_type_to_find,
    #     name_to_bind=name_to_bind,
    # )
    # tracker.setup(world.spec, data)

    # Initialize controller
    ctrl = Controller(
        controller_callback_function=sf_policy,
        time_steps_per_ctrl_step=1,
        tracker=tracker,
    )

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, cpg))

    console.log(f"xpos before sim: {tracker.history['xpos'][0]}") # TODO: REMOVE DEBUG

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
            path_to_video_folder = str(DATA / "videos")
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


if __name__ == "__main__":
    # Test several times
    main()

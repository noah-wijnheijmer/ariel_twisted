"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    if weights is None:
        # Initialize the networks weights randomly for backward compatibility
        w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
        w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
        w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))
    else:
        # Use provided weights from evolution
        w1_size = input_size * hidden_size
        w2_size = hidden_size * hidden_size
        w3_size = hidden_size * output_size
        
        w1 = weights[:w1_size].reshape(input_size, hidden_size)
        w2 = weights[w1_size:w1_size+w2_size].reshape(hidden_size, hidden_size)
        w3 = weights[w1_size+w2_size:w1_size+w2_size+w3_size].reshape(hidden_size, output_size)

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


# === EVOLUTIONARY ALGORITHM FUNCTIONS === #

def create_individual() -> dict[str, Any]:
    """Create an individual with body and brain genes."""
    genotype_size = 64
    
    # Body genes (fixed for this assignment)
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)
    
    # Neural network weights (what we're evolving)
    # We'll calculate exact size during first evaluation
    brain_genes = RNG.normal(0.0, 0.5, size=1000).astype(np.float32)  # Start with large array
    
    return {
        'body': [type_p_genes, conn_p_genes, rot_p_genes],
        'brain': brain_genes,
        'fitness': -float('inf')
    }


def evaluate_individual(individual: dict[str, Any]) -> float:
    """Evaluate fitness of one individual."""
    try:
        # Create robot from body genes
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward(individual['body'])
        
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        robot_graph = hpd.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        )
        
        core = construct_mjspec_from_graph(robot_graph)
        
        # Set up tracker
        tracker = Tracker(
            mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
            name_to_bind="core",
        )
        
        # Create controller with evolved weights
        def controller_with_weights(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
            # Calculate required weights size
            input_size = len(data.qpos)
            hidden_size = 8
            output_size = model.nu
            total_weights = input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size
            
            # Resize brain if needed
            if len(individual['brain']) != total_weights:
                if len(individual['brain']) < total_weights:
                    # Extend with random values
                    extra = RNG.normal(0.0, 0.5, size=total_weights - len(individual['brain']))
                    individual['brain'] = np.concatenate([individual['brain'], extra]).astype(np.float32)
                else:
                    # Truncate
                    individual['brain'] = individual['brain'][:total_weights]
            
            return nn_controller(model, data, individual['brain'])
        
        ctrl = Controller(
            controller_callback_function=controller_with_weights,
            tracker=tracker,
        )
        
        # Run simulation (headless for speed)
        experiment(robot=core, controller=ctrl, mode="simple")
        
        # Calculate fitness
        fitness = fitness_function(tracker.history["xpos"][0])
        return fitness
        
    except Exception as e:
        console.log(f"Error evaluating individual: {e}")
        return -float('inf')


def tournament_selection(population: list[dict[str, Any]], tournament_size: int = 3) -> dict[str, Any]:
    """Select individual using tournament selection."""
    tournament = RNG.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda x: x['fitness'])


def crossover(parent1: dict[str, Any], parent2: dict[str, Any], crossover_rate: float = 0.7) -> tuple[dict[str, Any], dict[str, Any]]:
    """Single-point crossover for brain weights."""
    child1 = {
        'body': parent1['body'].copy(),
        'brain': parent1['brain'].copy(),
        'fitness': -float('inf')
    }
    child2 = {
        'body': parent2['body'].copy(), 
        'brain': parent2['brain'].copy(),
        'fitness': -float('inf')
    }
    
    if RNG.random() < crossover_rate:
        # Crossover brain weights only
        min_length = min(len(parent1['brain']), len(parent2['brain']))
        crossover_point = RNG.integers(1, min_length)
        
        child1['brain'][:crossover_point] = parent2['brain'][:crossover_point]
        child2['brain'][:crossover_point] = parent1['brain'][:crossover_point]
    
    return child1, child2


def mutate(individual: dict[str, Any], mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> dict[str, Any]:
    """Gaussian mutation for brain weights."""
    mutated = {
        'body': individual['body'].copy(),
        'brain': individual['brain'].copy(),
        'fitness': -float('inf')
    }
    
    # Mutate brain weights
    for i in range(len(mutated['brain'])):
        if RNG.random() < mutation_rate:
            mutated['brain'][i] += RNG.normal(0, mutation_strength)
    
    return mutated


def evolutionary_algorithm() -> dict[str, Any]:
    """Main evolutionary algorithm loop."""
    POPULATION_SIZE = 10  # Smaller for demo
    GENERATIONS = 5       # Fewer generations for demo
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    ELITISM = 2  # Keep best 2 individuals
    
    console.log(f"Starting evolution: {POPULATION_SIZE} individuals, {GENERATIONS} generations")
    
    # Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    # Fixed body for all individuals (use same body throughout evolution)
    fixed_body = population[0]['body']
    for individual in population:
        individual['body'] = fixed_body
    
    best_fitness_history = []
    
    for generation in range(GENERATIONS):
        console.log(f"Generation {generation + 1}/{GENERATIONS}")
        
        # Evaluate all individuals
        for i, individual in enumerate(population):
            individual['fitness'] = evaluate_individual(individual)
            if (i + 1) % 5 == 0:
                console.log(f"  Evaluated {i + 1}/{POPULATION_SIZE} individuals")
        
        # Sort by fitness (best first)
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Track best fitness
        best_fitness = population[0]['fitness']
        avg_fitness = np.mean([ind['fitness'] for ind in population])
        best_fitness_history.append(best_fitness)
        
        console.log(f"  Best fitness: {best_fitness:.3f}, Average: {avg_fitness:.3f}")
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(ELITISM):
            elite = {
                'body': population[i]['body'].copy(),
                'brain': population[i]['brain'].copy(),
                'fitness': -float('inf')
            }
            new_population.append(elite)
        
        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            
            new_population.extend([child1, child2])
        
        population = new_population[:POPULATION_SIZE]
    
    # Return best individual
    final_population = population[:]
    for individual in final_population:
        individual['fitness'] = evaluate_individual(individual)
    
    best_individual = max(final_population, key=lambda x: x['fitness'])
    
    console.log(f"Evolution complete! Best fitness: {best_individual['fitness']:.3f}")
    console.log(f"Fitness progression: {best_fitness_history}")
    
    return best_individual


def main() -> None:
    """Entry point."""
    # Run evolutionary algorithm
    best_individual = evolutionary_algorithm()
    
    # Test the best individual with visualization
    console.log("Testing best individual with visualization...")
    
    # Use the best individual's genes
    genotype = best_individual['body']

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # Build the robot
    core = construct_mjspec_from_graph(robot_graph)

    # Set up tracker
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Create controller using the evolved brain weights
    def evolved_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        return nn_controller(model, data, best_individual["brain"])

    ctrl = Controller(
        controller_callback_function=evolved_controller,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl, mode="launcher")

    show_xpos_history(tracker.history["xpos"][0])

    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Best evolved robot fitness: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()


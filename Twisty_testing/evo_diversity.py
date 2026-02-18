from ea_components.individual import Individual, create_individual, evaluate_population
from ea_components.selection import survivor_selection
from ea_components.evolution import evolve_generation
from robot_body.constructor import construct_mjspec_from_graph
from robot_body.hi_prob_decoding import save_population_graph_as_json
from data_storing.data_store import (initialize_experiment_data, calculate_generation_statistics, finalize_experiment_data, save_checkpoint, load_checkpoint)
from simulation.visualization import (visualize_champ, show_qpos_history, plot_fitness_over_generations)
from rich.console import Console
import numpy as np
from typing import Any
from pathlib import Path
import json

console = Console()
# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
DATA_SETTINGS = [DATA, SCRIPT_NAME]
SEED = 41
RNG = np.random.default_rng(SEED)
EVOLUTION_CONFIG = {
    "generations": 20,
    "population_size": 100,
    "save_evolution_graphs": False,
    "sample_diversity_every": 10,
    "checkpoint_every": 1,  # Save checkpoint every N generations
    "load_checkpoint": False,    # Automatically resume from checkpoint if found
    "checkpoint_folder": f"Twisty_testing/checkpoints/experiment_1",
    "checkpoint_gen": 5, # which generation to load from.
    "start_id": 91, # if old graph data should be kept, make it some higher number.
    "twisty_evo": True,
    "mixed": False,
    "p_twisty": 0.5,
}
# if correcting for bounding box, the height will be reduced to zero. Otherwise choose a custom z value for the height.
EVAL_CONFIG = {"correct_for_bounding_box": True, "custom_z": 0.39, "custom_xy": [0, 0] ,"target_pos": [0, 5, 0.5], "brain_type": "sf_cpg", "num_modules": 20}

def run_evolution_experiment(
    seed: int,
    experiment_id: int,
    run_type: str,
    generations: int = EVOLUTION_CONFIG['generations'],
    population_size: int = EVOLUTION_CONFIG['population_size'],
    twisty_evo: bool = EVOLUTION_CONFIG["twisty_evo"],
    mixed_con: bool = EVOLUTION_CONFIG["mixed"],
    save_evolution_graphs: bool = EVOLUTION_CONFIG['save_evolution_graphs'],
    sample_diversity_every: int = EVOLUTION_CONFIG['sample_diversity_every'],
) -> tuple[Individual, dict[str, Any]]:
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
    rng = np.random.default_rng(seed)
    # Initialize experiment tracking data
    experiment_data = initialize_experiment_data(generations, population_size)
    
    # Initialize separate populations
    if EVOLUTION_CONFIG["load_checkpoint"] is True:
        base_filename = f"checkpoint_gen_{EVOLUTION_CONFIG['checkpoint_gen']}.json"
        path = Path(EVOLUTION_CONFIG["checkpoint_folder"])
        checkpoint_path = path / base_filename
        loaded_generation_id, population, _ = load_checkpoint(file_path=checkpoint_path)
        generations = generations - (loaded_generation_id + 1)
        print(generations)
    elif twisty_evo is False:
        population = [
        create_individual(con_twisty=False, id=i, rng=rng, num_modules=EVAL_CONFIG["num_modules"]) for i in range(population_size)]
    elif mixed_con is True:
        population = []
        p_twisty = EVOLUTION_CONFIG["p_twisty"]
        i = 0
        for _ in range(population_size):
            p = rng.random()
            if p >= float(p_twisty):
                population.append(create_individual(con_twisty=True, id=i, rng=rng , num_modules=EVAL_CONFIG["num_modules"]))
            else:
                population.append(create_individual(con_twisty=False, id=i, rng=rng, num_modules=EVAL_CONFIG["num_modules"]))
            i+=1
    else:
        population = [
        create_individual(con_twisty=True, id=i, rng=rng, num_modules=EVAL_CONFIG["num_modules"]) for i in range(population_size)]
    evaluate_population(population, rng, EVAL_CONFIG["correct_for_bounding_box"], EVAL_CONFIG["custom_z"], EVAL_CONFIG["custom_xy"] ,EVAL_CONFIG["target_pos"], EVAL_CONFIG["brain_type"])
    population_graphs = [i.graph for i in population]
    diversity_dir = DATA / "population_graphs" / f"generation_0"
    diversity_dir.mkdir(parents=True, exist_ok=True)
    filename_div = f"rep{experiment_id}_{run_type}"
    save_population_graph_as_json(population_graphs, diversity_dir / filename_div)
    accepted_gens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # Evolution loop
    for generation in range(generations):        
        next_gen = evolve_generation(population, rng=rng)
        evaluate_population(next_gen, rng, EVAL_CONFIG["correct_for_bounding_box"], EVAL_CONFIG["custom_z"], EVAL_CONFIG["custom_xy"] ,EVAL_CONFIG["target_pos"], EVAL_CONFIG["brain_type"])
        population = survivor_selection(next_gen, population_size)
        if generation+1 in accepted_gens:
            population_graphs = [i.graph for i in population]
            diversity_dir = DATA / "population_graphs" / f"generation_{generation+1}"
            diversity_dir.mkdir(parents=True, exist_ok=True)
            filename_div = f"rep{experiment_id}_{run_type}"
            save_population_graph_as_json(population_graphs, diversity_dir / filename_div)
def main() -> None:
    """Entry point for evolutionary experiment."""
    repetitions = 30
    # run_type = "twisted_amphi"
    # for i in range(repetitions):
    #    run_evolution_experiment(seed=SEED+i, experiment_id=i, run_type=run_type, twisty_evo=True, mixed_con=False)
    # run_type = "mixed_amphi"
    # for i in range(repetitions):
    #    run_evolution_experiment(seed=SEED+i, experiment_id=i, run_type=run_type, twisty_evo=True, mixed_con=True)
    run_type = "non-twisty_amphi"
    for i in range(repetitions):
        run_evolution_experiment(seed=SEED+i, experiment_id=i, run_type=run_type, twisty_evo=False, mixed_con=False)

if __name__ == "__main__":
    # Test several times
    main()
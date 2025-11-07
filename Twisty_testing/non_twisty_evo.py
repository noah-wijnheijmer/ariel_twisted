from ea_components.individual import Individual, create_individual, evaluate_population
from ea_components.evolution.evolution import evolve_generation
from robot_body.constructor import construct_mjspec_from_graph
from robot_body.hi_prob_decoding import save_graph_as_json
from data_storing.data_store import (initialize_experiment_data, calculate_generation_statistics, finalize_experiment_data)
from simulation.visualization import visualize_champ
from rich.console import Console
import numpy as np
from typing import Any
from pathlib import Path

console = Console()
# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
DATA_SETTINGS = [DATA, SCRIPT_NAME]
SEED = 40
RNG = np.random.default_rng(SEED)
EVOLUTION_CONFIG = {
    "generations": 10,
    "population_size": 100,
    "save_evolution_graphs": True,
    "sample_diversity_every": 10,
    "checkpoint_every": 1,  # Save checkpoint every N generations
    "auto_resume": True,    # Automatically resume from checkpoint if found
}
# if correcting for bounding box, the height will be reduced to zero. Otherwise choose a custom z value for the height.
EVAL_CONFIG = {"correct_for_bounding_box": True, "custom_z": 0.39, "custom_xy": [0, 0] ,"target_pos": [0, 5, 0.5], "brain_type": "sf_cpg"}

def run_evolution_experiment(
    generations: int = EVOLUTION_CONFIG['generations'],
    population_size: int = EVOLUTION_CONFIG['population_size'],
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
    
    # Initialize experiment tracking data
    experiment_data = initialize_experiment_data(generations, population_size)
    
    # Initialize separate populations
    non_twisty_population = [
        create_individual(con_twisty=False) for _ in range(population_size)
    ]
    evaluate_population(non_twisty_population, EVAL_CONFIG["correct_for_bounding_box"], EVAL_CONFIG["custom_z"], EVAL_CONFIG["custom_xy"] ,EVAL_CONFIG["target_pos"], EVAL_CONFIG["brain_type"])
    # Track best individuals across all generations
    best_non_twisty_ever = max(non_twisty_population, key=lambda x: x.fitness)
    best_non_twisty_fitness = -float('inf')
    # Evolution loop
    for generation in range(generations):
        console.log(f"\n--- Generation {generation + 1} ---")
        console.log("Evaluating non-twisty population...")
        evaluate_population(non_twisty_population, EVAL_CONFIG["correct_for_bounding_box"], EVAL_CONFIG["custom_z"], EVAL_CONFIG["custom_xy"] ,EVAL_CONFIG["target_pos"], EVAL_CONFIG["brain_type"])

        # Track best individuals this generation
        current_best_non_twisty = max(non_twisty_population, key=lambda x: x.fitness)
        print(current_best_non_twisty.fitness)    
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
                    current_best_non_twisty.graph,
                    champion_dir / filename,
                )
        
        # Calculate comprehensive statistics for scientific analysis
        gen_stats = calculate_generation_statistics(
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
            top_non_twisty = sorted(non_twisty_population, key=lambda x: x.fitness, reverse=True)[:3]
                
            for i, ind in enumerate(top_non_twisty):
                filename = f"mixed_twisty_top_{i+1}_fitness_{ind.fitness:.3f}.json"
                save_graph_as_json(ind.graph, diversity_dir / filename)

        # Report generation statistics
        console.log(f"Generation {generation + 1} Detailed Statistics:")
        non_twisty_stats = gen_stats["mixed_twisty"]
        
        console.log(
            f"  mixed-twisty - Mean: {non_twisty_stats['mean']:.3f} ± "
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
            non_twisty_population = evolve_generation(non_twisty_population)

    champion = best_non_twisty_ever
    if champion.twisty is True:
        champion_type = "TWISTY"
    else:
        champion_type = "NON-TWISTY"
    champion_fitness = best_non_twisty_fitness
    # Calculate final experiment statistics and save data
    finalize_experiment_data(
        experiment_data, best_non_twisty_fitness,
        champion_type, champion_fitness, DATA
    )

    console.log(f"[bold green]🏆 OVERALL CHAMPION: {champion_type} robot")
    console.log(f"[bold green]🏆 Champion Fitness: {champion_fitness:.3f}")

    # Save champion graph to file
    champion_filename = f"champion_{champion_type.lower()}_robot.json"
    save_graph_as_json(champion.graph, DATA / champion_filename)
    console.log(f"📋 Champion robot saved to: {DATA / champion_filename}")
    
    return champion, experiment_data
def main() -> None:
    """Entry point for evolutionary experiment."""
    champion, _ = run_evolution_experiment() #You can set some experimental variables here or all the way above in EVOLUTION_CONFIG
    # Visualize the champion robot - change mode here!
    console.log("\n🏆 Visualizing champion robot...")
    champion_robot = construct_mjspec_from_graph(champion.graph)
    # Change mode to "launcher" to see robot in interactive viewer
    # Change mode to "video" to record a video
    visualize_champ(champion_robot, champion, EVAL_CONFIG["correct_for_bounding_box"], EVAL_CONFIG["custom_z"], EVAL_CONFIG["custom_xy"] ,DATA_SETTINGS, EVAL_CONFIG["brain_type"], mode="launcher")

if __name__ == "__main__":
    # Test several times
    main()
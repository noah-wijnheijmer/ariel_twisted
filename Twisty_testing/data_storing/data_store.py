from typing import Any
import json
from robot_body.hi_prob_decoding import load_graph_from_json
from ea_components.individual import Individual, individual_from_dict
import numpy as np
from pathlib import Path
from rich.console import Console
import os
console = Console()

def initialize_experiment_data(generations: int, population_size: int) -> dict[str, Any]:
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
        "final_individuals": {}, 
    }

def calculate_generation_statistics( 
    mixed_twisty_population: list[Individual], 
    generation: int,
    run_type:str,
) -> dict[str, Any]:
    """Calculate comprehensive statistics for a generation."""
    mixed_twisty_fitnesses = [ind.fitness for ind in mixed_twisty_population]
    
    return {"generation": generation + 1,
        f"{run_type}": {
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
    experiment_data: dict[str, Any],
    best_mixed_twisty_fitness: float,
    champion_type: str,
    champion_fitness: float,
    experiment_id:int,
    run_type: str,
    data: Path
) -> None:
    """Calculate final statistics and save experiment data."""
    # Calculate overall statistics
    mixed_twisty_all_scores = [
        score for gen in experiment_data["generations"] 
        for score in gen[f"{run_type}"]["all_fitnesses"]
    ]
    
    experiment_data["final_statistics"] = {
        f"{run_type}": {
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
    experiment_filename = f"experiment_data_{champion_type.lower()}_champion{experiment_id+10}.json"
    with open(data / experiment_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)
    console.log(f"📈 Complete experiment data saved to: {data / experiment_filename}")

def save_checkpoint(
        population: list[Individual],
        generation_id: int,
        config: dict[str, Any],
        folder_path: str
    ):
    """Save a checkpoint of the current population and configuration."""

    checkpoint_data = {
        "generation_id": generation_id,
        "config": config,
        "population": [ind.to_dict(exclude_fields=['graph']) for ind in population],
    }

    base_filename = f"checkpoint_gen_{generation_id}.json"
    directory = Path(folder_path)
    checkpoint_path = directory / base_filename

    if checkpoint_path.exists():
        suffix = 1
        while True:
            candidate = directory / f"checkpoint_gen_{generation_id}_{suffix}.json"
            if not candidate.exists():
                checkpoint_path = candidate
                break
            suffix += 1
    os.makedirs(directory, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    console.log(f"Checkpoint saved to: {checkpoint_path}") # DEBUG
    print(f"Checkpoint saved to: {checkpoint_path}")

def load_checkpoint(file_path: str) -> tuple[int, list[Individual], dict[str, Any]]:
    """Load a checkpoint and return generation id, population, and configuration."""
    with open(file_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    generation_id = checkpoint_data["generation_id"]
    config = checkpoint_data["config"]

    population_data = checkpoint_data["population"]

    # Reconstruct graphs from genotype probabilities
    for ind_data in population_data:

        folder_path = "Twisty_testing/population_data/graphs"
        directory = Path(folder_path)
        path = directory / f"{ind_data["id"]}_graph.json"
        graph = load_graph_from_json(path)
        ind_data["graph"] = graph

    population = [individual_from_dict(ind_data) for ind_data in population_data]

    console.log(f"Checkpoint loaded from: {file_path}")

    return generation_id, population, config
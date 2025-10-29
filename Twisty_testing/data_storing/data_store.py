from typing import Any
import json
from ea_components.individual import Individual
import numpy as np
from pathlib import Path
from rich.console import Console

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
    generation: int
) -> dict[str, Any]:
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
    experiment_data: dict[str, Any],
    best_mixed_twisty_fitness: float,
    champion_type: str,
    champion_fitness: float,
    data: Path
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
    with open(data / experiment_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)
    console.log(f"📈 Complete experiment data saved to: {data / experiment_filename}")
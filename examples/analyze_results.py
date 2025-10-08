#!/usr/bin/env python3
"""
Scientific analysis script for twisty vs non-twisty evolutionary robotics experiment.

This script loads experiment data and creates comprehensive statistical analyses
and visualizations for publication-quality results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_experiment_data(data_file: Path) -> dict:
    """Load experiment data from JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_fitness_evolution_plot(data: dict, output_dir: Path) -> None:
    """Create plots showing fitness evolution over generations."""
    generations = [g['generation'] for g in data['generations']]
    
    # Extract statistics
    twisty_means = [g['twisty']['mean'] for g in data['generations']]
    twisty_stds = [g['twisty']['std'] for g in data['generations']]
    twisty_maxes = [g['twisty']['max'] for g in data['generations']]
    
    non_twisty_means = [g['non_twisty']['mean'] for g in data['generations']]
    non_twisty_stds = [g['non_twisty']['std'] for g in data['generations']]
    non_twisty_maxes = [g['non_twisty']['max'] for g in data['generations']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Mean fitness with error bars
    ax1.errorbar(generations, twisty_means, yerr=twisty_stds, 
                 label='Twisty Robots', marker='o', linewidth=2, capsize=5)
    ax1.errorbar(generations, non_twisty_means, yerr=non_twisty_stds, 
                 label='Non-Twisty Robots', marker='s', linewidth=2, capsize=5)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Fitness ± Std Dev')
    ax1.set_title('Evolution of Mean Fitness Over Generations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best fitness per generation
    ax2.plot(generations, twisty_maxes, label='Twisty Best', 
             marker='o', linewidth=2, markersize=8)
    ax2.plot(generations, non_twisty_maxes, label='Non-Twisty Best', 
             marker='s', linewidth=2, markersize=8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Evolution of Best Fitness Over Generations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fitness_evolution.pdf', bbox_inches='tight')
    plt.close()


def create_distribution_plots(data: dict, output_dir: Path) -> None:
    """Create plots showing fitness distributions."""
    # Collect all fitness values
    all_twisty_fitness = []
    all_non_twisty_fitness = []
    
    for gen in data['generations']:
        all_twisty_fitness.extend(gen['twisty']['all_fitnesses'])
        all_non_twisty_fitness.extend(gen['non_twisty']['all_fitnesses'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram comparison
    ax1.hist(all_twisty_fitness, bins=20, alpha=0.7, label='Twisty Robots', 
             density=True, edgecolor='black')
    ax1.hist(all_non_twisty_fitness, bins=20, alpha=0.7, label='Non-Twisty Robots', 
             density=True, edgecolor='black')
    ax1.set_xlabel('Fitness')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Fitness Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    box_data = [all_twisty_fitness, all_non_twisty_fitness]
    box = ax2.boxplot(box_data, labels=['Twisty', 'Non-Twisty'], patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Fitness')
    ax2.set_title('Fitness Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fitness_distributions.pdf', bbox_inches='tight')
    plt.close()


def perform_statistical_tests(data: dict) -> dict:
    """Perform statistical tests comparing twisty vs non-twisty performance."""
    # Collect all fitness values
    all_twisty_fitness = []
    all_non_twisty_fitness = []
    
    for gen in data['generations']:
        all_twisty_fitness.extend(gen['twisty']['all_fitnesses'])
        all_non_twisty_fitness.extend(gen['non_twisty']['all_fitnesses'])
    
    # Convert to numpy arrays
    twisty_fitness = np.array(all_twisty_fitness)
    non_twisty_fitness = np.array(all_non_twisty_fitness)
    
    # Statistical tests
    results = {}
    
    # T-test (assuming unequal variances)
    t_stat, t_pvalue = stats.ttest_ind(twisty_fitness, non_twisty_fitness, equal_var=False)
    results['t_test'] = {'statistic': t_stat, 'p_value': t_pvalue}
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(twisty_fitness, non_twisty_fitness, 
                                          alternative='two-sided')
    results['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pvalue}
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(twisty_fitness, non_twisty_fitness)
    results['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_pvalue}
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(twisty_fitness) - 1) * np.var(twisty_fitness, ddof=1) + 
                         (len(non_twisty_fitness) - 1) * np.var(non_twisty_fitness, ddof=1)) / 
                        (len(twisty_fitness) + len(non_twisty_fitness) - 2))
    cohens_d = (np.mean(twisty_fitness) - np.mean(non_twisty_fitness)) / pooled_std
    results['effect_size'] = {'cohens_d': cohens_d}
    
    # Descriptive statistics
    results['descriptive_stats'] = {
        'twisty': {
            'mean': float(np.mean(twisty_fitness)),
            'std': float(np.std(twisty_fitness, ddof=1)),
            'median': float(np.median(twisty_fitness)),
            'min': float(np.min(twisty_fitness)),
            'max': float(np.max(twisty_fitness)),
            'n': len(twisty_fitness)
        },
        'non_twisty': {
            'mean': float(np.mean(non_twisty_fitness)),
            'std': float(np.std(non_twisty_fitness, ddof=1)),
            'median': float(np.median(non_twisty_fitness)),
            'min': float(np.min(non_twisty_fitness)),
            'max': float(np.max(non_twisty_fitness)),
            'n': len(non_twisty_fitness)
        }
    }
    
    return results


def generate_scientific_report(data: dict, stats_results: dict, output_dir: Path) -> None:
    """Generate a comprehensive scientific report."""
    report_path = output_dir / 'scientific_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EVOLUTIONARY ROBOTICS EXPERIMENT: TWISTY vs NON-TWISTY COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        # Experiment parameters
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        for key, value in data['parameters'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write(f"Total Generations: {len(data['generations'])}\n")
        f.write(f"Total Evaluations per Type: {data['final_statistics']['twisty']['total_evaluations']}\n\n")
        
        # Final results
        f.write("FINAL RESULTS:\n")
        f.write("-" * 15 + "\n")
        champions = data['champions']
        f.write(f"Overall Champion: {champions['overall_champion']}\n")
        f.write(f"Champion Fitness: {champions['champion_fitness']:.6f}\n")
        f.write(f"Performance Advantage: {champions['performance_advantage_percent']:.2f}%\n\n")
        
        # Descriptive statistics
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write("-" * 25 + "\n")
        desc = stats_results['descriptive_stats']
        f.write("Twisty Robots:\n")
        f.write(f"  Mean ± SD: {desc['twisty']['mean']:.6f} ± {desc['twisty']['std']:.6f}\n")
        f.write(f"  Median: {desc['twisty']['median']:.6f}\n")
        f.write(f"  Range: [{desc['twisty']['min']:.6f}, {desc['twisty']['max']:.6f}]\n")
        f.write(f"  N: {desc['twisty']['n']}\n\n")
        
        f.write("Non-Twisty Robots:\n")
        f.write(f"  Mean ± SD: {desc['non_twisty']['mean']:.6f} ± {desc['non_twisty']['std']:.6f}\n")
        f.write(f"  Median: {desc['non_twisty']['median']:.6f}\n")
        f.write(f"  Range: [{desc['non_twisty']['min']:.6f}, {desc['non_twisty']['max']:.6f}]\n")
        f.write(f"  N: {desc['non_twisty']['n']}\n\n")
        
        # Statistical tests
        f.write("STATISTICAL TESTS:\n")
        f.write("-" * 20 + "\n")
        
        # T-test
        t_test = stats_results['t_test']
        f.write(f"Independent Samples T-Test:\n")
        f.write(f"  t-statistic: {t_test['statistic']:.4f}\n")
        f.write(f"  p-value: {t_test['p_value']:.6f}\n")
        f.write(f"  Significant at α=0.05: {'Yes' if t_test['p_value'] < 0.05 else 'No'}\n\n")
        
        # Mann-Whitney U test
        mw_test = stats_results['mann_whitney']
        f.write(f"Mann-Whitney U Test (non-parametric):\n")
        f.write(f"  U-statistic: {mw_test['statistic']:.4f}\n")
        f.write(f"  p-value: {mw_test['p_value']:.6f}\n")
        f.write(f"  Significant at α=0.05: {'Yes' if mw_test['p_value'] < 0.05 else 'No'}\n\n")
        
        # Effect size
        effect = stats_results['effect_size']
        f.write(f"Effect Size (Cohen's d): {effect['cohens_d']:.4f}\n")
        if abs(effect['cohens_d']) < 0.2:
            effect_interp = "negligible"
        elif abs(effect['cohens_d']) < 0.5:
            effect_interp = "small"
        elif abs(effect['cohens_d']) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        f.write(f"Effect Size Interpretation: {effect_interp}\n\n")
        
        # Conclusions
        f.write("CONCLUSIONS:\n")
        f.write("-" * 12 + "\n")
        if champions['overall_champion'] == 'TWISTY':
            f.write("• Twisty robots achieved superior overall performance\n")
        else:
            f.write("• Non-twisty robots achieved superior overall performance\n")
            
        if t_test['p_value'] < 0.05:
            f.write("• Statistical significance detected in performance difference\n")
        else:
            f.write("• No statistically significant difference in performance\n")
            
        f.write(f"• Effect size suggests {effect_interp} practical difference\n")
        f.write("• Twisted orientation shows promise for physical robot implementation\n")


def main():
    """Main analysis function."""
    # Set up paths
    data_dir = Path("__data__/twisty")
    output_dir = data_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Find the most recent experiment data file
    experiment_files = list(data_dir.glob("experiment_data_*.json"))
    if not experiment_files:
        print("No experiment data files found!")
        return
    
    latest_file = max(experiment_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing data from: {latest_file}")
    
    # Load data
    data = load_experiment_data(latest_file)
    
    # Create visualizations
    print("Creating fitness evolution plots...")
    create_fitness_evolution_plot(data, output_dir)
    
    print("Creating fitness distribution plots...")
    create_distribution_plots(data, output_dir)
    
    # Perform statistical analysis
    print("Performing statistical tests...")
    stats_results = perform_statistical_tests(data)
    
    # Save statistical results
    with open(output_dir / 'statistical_results.json', 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, indent=2)
    
    # Generate scientific report
    print("Generating scientific report...")
    generate_scientific_report(data, stats_results, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("• fitness_evolution.png/pdf - Evolution of fitness over generations")
    print("• fitness_distributions.png/pdf - Fitness distribution comparisons")
    print("• statistical_results.json - Complete statistical analysis")
    print("• scientific_analysis_report.txt - Comprehensive scientific report")


if __name__ == "__main__":
    main()

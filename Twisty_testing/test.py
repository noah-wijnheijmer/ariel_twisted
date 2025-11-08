import ea_components.individual as ind
from data_storing.data_store import save_checkpoint, load_checkpoint
from itertools import zip_longest
# from pathlib import Path

EVOLUTION_CONFIG = {
    "generations": 2,
    "population_size": 5,
    "save_evolution_graphs": True,
    "sample_diversity_every": 10,
    "checkpoint_every": 1,  # Save checkpoint every N generations
    "auto_resume": True,    # Automatically resume from checkpoint if found
}

population = [ind.create_individual(con_twisty=False, num_modules=10) for _ in range(10)]

checkpoint_path = save_checkpoint(generation_id=1, population=population, folder_path="Twisty_testing/checkpoints/experiment_1", config=EVOLUTION_CONFIG)

loaded_generation_id, loaded_population, loaded_config = load_checkpoint(file_path=checkpoint_path)

assert loaded_generation_id == 1, "Generation ID does not match."
assert loaded_config == EVOLUTION_CONFIG, "Configuration does not match."

def summarize_population_differences(original_population, loaded_population):
    differences = []
    for index, (original, loaded) in enumerate(zip_longest(original_population, loaded_population)):
        if original is None or loaded is None:
            differences.append({
                "index": index,
                "type": "length_mismatch",
                "original_exists": original is not None,
                "loaded_exists": loaded is not None,
            })
            continue
        original_dict = original.to_dict()
        loaded_dict = loaded.to_dict()
        field_diffs = {
            field: {"original": original_dict.get(field), "loaded": loaded_dict.get(field)}
            for field in sorted(set(original_dict) | set(loaded_dict))
            if original_dict.get(field) != loaded_dict.get(field)
        }
        if field_diffs:
            differences.append({"index": index, "fields": field_diffs})
    return differences

differences = summarize_population_differences(population, loaded_population)
if differences:
    print("\nPopulation mismatch details:")
    for diff in differences:
        if diff.get("type") == "length_mismatch":
            print(f" - index {diff['index']}: original_exists={diff['original_exists']}, loaded_exists={diff['loaded_exists']}")
            continue
        print(f" - index {diff['index']}:")
        for field, values in diff["fields"].items():
            print(f"   {field}: original={values['original']} | loaded={values['loaded']}")

for i in range(len(population)):
    original_genotype = population[i].genotype
    loaded_genotype = loaded_population[i].genotype
    if original_genotype != loaded_genotype:
        print(f"Genotype mismatch at index {i}:")
        print(f" Original: {original_genotype}")
        print(f" Loaded:   {loaded_genotype}")
    else:
        print(f"Genotype match at index {i}.")
    

assert population == loaded_population, "Loaded population does not match the saved population."
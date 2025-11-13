import numpy as np
from ea_components.individual import Individual, create_individual_from_matrices
from robot_body.body_config import TWISTY_INDEXES
from ea_components.individual import JSONIterable
from typing import Any

SEED = 40
RNG = np.random.default_rng(SEED)

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

def float_creep(
    individual: Individual,
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

def crossover_individuals(
    parent1: Individual, 
    parent2: Individual,
    id: int, 
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
        child_rot[:, TWISTY_INDEXES] = 0
    
    return create_individual_from_matrices(
        child_type, child_conn, child_rot, twisty, id=id
    )


def mutate_individual(
    individual: Individual, id: int, mutation_rate: float = 0.1, 
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
                rotation_probs[i, TWISTY_INDEXES] = 0
    
    # Create new individual with mutated matrices
    return create_individual_from_matrices(
        np.array(type_probs),
        np.array(conn_probs), 
        np.array(rotation_probs),
        individual.twisty,
        id=id
    )

class Crossover:
    @staticmethod
    def one_point(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
    ) -> tuple[JSONIterable, JSONIterable]:
        # Prep
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()

        # Ensure parents have the same length
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)

        # Select crossover point
        crossover_point = RNG.integers(0, len(parent_i_arr))

        # Copy over parents
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()

        # Perform crossover
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]

        # Correct final shape
        child1 = child1.reshape(parent_i_arr_shape).astype(int).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(int).tolist()
        return child1, child2
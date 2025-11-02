import numpy as np
from typing import Any
from networkx import DiGraph
from collections.abc import Hashable, Sequence
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel
from rich.console import Console
from robot_body.body_config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    TWISTY_INDEXES,
)
from robot_body.hi_prob_decoding import HighProbabilityDecoder
from robot_body.constructor import construct_mjspec_from_graph
from ea_components.evaluation.na_evaluation import na_for_fitness
from ea_components.evaluation.sf_evaluation import sf_for_fitness
# Global constants
SEED = 40
RNG = np.random.default_rng(SEED)
console = Console()
# Typing aliases
type JSONPrimitive = str | int | float | bool
type JSONType = JSONPrimitive | Sequence[JSONType] | dict[Hashable, JSONType]
type JSONIterable = Sequence[JSONType] | dict[Hashable, JSONType]


class Individual(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # ------------------------ LIFE TIME ------------------------ #
    alive: bool = Field(default=True, index=True)

    time_of_birth: int = Field(default=-1, index=True)
    time_of_death: int = Field(default=-1, index=True)
    time_alive: int = Field(default=-1, index=True)
    # parents_id removed for now; reintroduce later if needed.

    # ------------------------ FITNESS ------------------------ #
    requires_eval: bool = Field(default=True, index=True)
    fitness_: float | None = Field(default=None, index=True)
    twisty: bool = Field(default=False, index=True)
    
    @property
    def fitness(self) -> float:
        if self.fitness_ is None:
            msg = "Trying to fetch uninitialized data in fitness!\n"
            msg += f"--> {self.fitness_=}"
            raise ValueError(msg)
        return self.fitness_

    @fitness.setter
    def fitness(self, fitness_value: float) -> None:
        if isinstance(fitness_value, float) is False:
            msg = "Trying to assign a non-float value to fitness!\n"
            msg += f"--> {fitness_value=}"
            raise ValueError(msg)
        self.requires_eval = False
        self.fitness_ = fitness_value

    # ------------------------ GENOTYPE ------------------------ #
    requires_init: bool = Field(default=True, index=True)
    graph_: JSONIterable | None = Field(default=None, sa_column=Column(JSON))
    genotype_: JSONIterable | None = Field(default=None, sa_column=Column(JSON))
    brain_genotype_: JSONIterable | None = Field(default=None, sa_column=Column(JSON))

    @property
    def graph(self) -> JSONIterable:
        if self.graph_ is None:
            msg = "Trying to fetch uninitialized data in graph!"
            raise ValueError(msg)
        return self.graph_
    
    @property
    def genotype(self) -> JSONIterable:
        if self.genotype_ is None:
            msg = "Trying to fetch uninitialized data in body genotype!"
            raise ValueError(msg)
        return self.genotype_
    
    @property
    def brain_genotype(self) -> JSONIterable:
        if self.brain_genotype_ is None:
            msg = "Trying to fetch uninitialized data in brain genotype!"
            raise ValueError(msg)
        return self.brain_genotype_
    
    @graph.setter
    def graph(self, individual_graph: JSONIterable) -> None:
        self.requires_init = not bool(individual_graph)
        self.graph_ = individual_graph

    @genotype.setter
    def genotype(self, individual_genotype: JSONIterable) -> None:
        self.requires_init = not bool(individual_genotype)
        self.genotype_ = individual_genotype

    @brain_genotype.setter
    def brain_genotype(self, individual_brain_genotype: JSONIterable) -> None:
        self.brain_genotype_ = individual_brain_genotype

    # ------------------------ TAGS ------------------------
    tags_: dict[JSONType, JSONType] = Field(
        default={},
        sa_column=Column(JSON),
    )

    @property
    def tags(self) -> dict[JSONType, JSONType]:
        return self.tags_

    @tags.setter
    def tags(self, tag: dict[JSONType, JSONType]) -> None:
        self.tags_ = {**self.tags_, **tag}


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
            i[TWISTY_INDEXES] = [0]*4

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

def evaluate_population(population: list[Individual], custom_spawn_pos: bool, spawn_pos: list[float], target_pos: list[float], brain_type: str) -> None:
    """Evaluate fitness for all individuals in population."""
    for individual in population:
        try:
            robot = construct_mjspec_from_graph(individual.graph)
            if brain_type == "na_cpg":
                fitness = na_for_fitness(robot, individual, custom_spawn_pos,spawn_pos, target_pos)
            elif brain_type == "sf_cpg":
                fitness = sf_for_fitness(robot, individual, custom_spawn_pos, spawn_pos, target_pos)
            individual.fitness = fitness # type: ignore
            console.log(f"Individual (twisty={individual.twisty}) fitness: {fitness:.3f}") # type: ignore
        except Exception as e:
            console.log(f"Error evaluating individual: {e}")
            individual.fitness = 0.0
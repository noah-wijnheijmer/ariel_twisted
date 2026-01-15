import numpy as np
from typing import Any
from networkx import DiGraph
from collections.abc import Hashable, Sequence
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel
from pathlib import Path
from rich.console import Console
from robot_body.body_config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    TWISTY_INDEXES,
)
from robot_body.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json
import os
from robot_body.constructor import construct_mjspec_from_graph
from ea_components.evaluation_amphi import run_for_fitness
# Global constants
SEED = 41
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

    def to_dict(self, exclude_fields: list[str] = []) -> dict[str, Any]:
        """Convert Individual to dictionary representation with optional exclusions."""
        data = {
            "id": self.id,
            "alive": self.alive,
            "time_of_birth": self.time_of_birth,
            "time_of_death": self.time_of_death,
            "time_alive": self.time_alive,
            "requires_eval": self.requires_eval,
            "fitness": self.fitness_,
            "twisty": self.twisty,
            "requires_init": self.requires_init,
            "graph": self.graph_,
            "genotype": self.genotype_,
            "brain_genotype": self.brain_genotype_,
            "tags": self.tags_,
        }
        for field in exclude_fields:
            data.pop(field, None)
        return data


def individual_from_dict(data: dict[str, Any]) -> Individual:
    """Create Individual from dictionary representation."""
    ind = Individual()
    ind.id = data.get("id")
    ind.alive = data.get("alive", True)
    ind.time_of_birth = data.get("time_of_birth", -1)
    ind.time_of_death = data.get("time_of_death", -1)
    ind.time_alive = data.get("time_alive", -1)
    ind.requires_eval = data.get("requires_eval", True)
    ind.fitness_ = data.get("fitness")
    ind.twisty = data.get("twisty", False)
    ind.requires_init = data.get("requires_init", True)
    ind.graph_ = data.get("graph")
    ind.genotype_ = data.get("genotype")
    ind.brain_genotype_ = data.get("brain_genotype")
    ind.tags_ = data.get("tags", {})
    return ind

def create_individual(con_twisty: bool, id: int, rng, num_modules: int=20) -> Individual:
    ind = Individual()
    ind.id = id
    # "Type" probability space - bias towards HINGE modules for functional robots
    type_probability_space = rng.random(
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
    conn_probability_space = rng.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )
    type_probability_space[:, 0] = 0
    type_probability_space[0][0] = 1.0
    # print(type_probability_space)
    conn_probability_space[:, 0, :] = 0 
    # print(f"conn: {conn_probability_space[0]}")
    hinge_idxs = []
    brick_idxs = []
    for i in range(len(type_probability_space)):
        mod = np.argmax(type_probability_space[i])
        if mod == 1:
            brick_idxs.append(i)
        elif mod == 2:
            hinge_idxs.append(i)
    # print(brick_idxs)
    # print(hinge_idxs)
    for parent_idx in range(len(conn_probability_space)):
        # print(parent_idx)
        parent = np.argmax(type_probability_space[parent_idx])
        if parent == 1:
            conn_probability_space[parent_idx,brick_idxs, :] = 0
            # print(conn_probability_space[parent_idx])
        elif parent == 2:
            conn_probability_space[parent_idx,hinge_idxs, :] = 0
            # print(conn_probability_space[parent_idx])
    # "Rotation" probability space
    if con_twisty is True:   
        rotation_probability_space = rng.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
    else:
        # If twisty not true, the twisted angles are given the value zero, so they can't be selected.
        rotation_probability_space = rng.random(
            size=(num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
        for i in rotation_probability_space:
            i[TWISTY_INDEXES] = [0]*4

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        type_probability_space.copy(),
        conn_probability_space.copy(),
        rotation_probability_space.copy(),
    )
    folder_path = "Twisty_testing/population_data/graphs"
    directory = Path(folder_path)
    os.makedirs(directory, exist_ok=True)
    path = directory / f"{ind.id}_graph.json"
    save_graph_as_json(graph, path)
    # ===== DEBUG PRINTS =====

    #print()
    #print("Type probability space shape:", type_probability_space.shape)
    #print("Connection probability space shape:", conn_probability_space.shape)
    #print("Rotation probability space shape:", rotation_probability_space.shape)
    #print()

    # ========================

    ind.genotype = [type_probability_space.tolist(), conn_probability_space.tolist(), rotation_probability_space.tolist()]
    ind.graph = graph
    ind.twisty = con_twisty
    return ind

def create_individual_from_matrices(
    type_probs: np.ndarray[Any, Any], 
    conn_probs: np.ndarray[Any, Any], 
    rotation_probs: np.ndarray[Any, Any], 
    twisty: bool,
    id: int
) -> Individual:
    """Create individual from probability matrices."""
    ind = Individual()
    ind.id = id
    # Decode to graph
    conn_probs[:, 0, :] = 0
    type_probs[:, 0] = 0
    type_probs[0][0] = 1.0
    # print(type_probability_space)
    conn_probs[:, 0, :] = 0 
    # print(f"conn: {conn_probability_space[0]}")
    hinge_idxs = []
    brick_idxs = []
    for i in range(len(type_probs)):
        mod = np.argmax(type_probs[i])
        if mod == 1:
            brick_idxs.append(i)
        elif mod == 2:
            hinge_idxs.append(i)
    # print(brick_idxs)
    # print(hinge_idxs)
    for parent_idx in range(len(conn_probs)):
        # print(parent_idx)
        parent = np.argmax(type_probs[parent_idx])
        if parent == 1:
            conn_probs[parent_idx,brick_idxs, :] = 0
            # print(conn_probability_space[parent_idx])
        if parent == 2:
            conn_probs[parent_idx,hinge_idxs, :] = 0
            # print(conn_probability_space[parent_idx])
    hpd = HighProbabilityDecoder(len(type_probs))
    graph = hpd.probability_matrices_to_graph(
        np.array(type_probs, copy=True),
        np.array(conn_probs, copy=True),
        np.array(rotation_probs, copy=True),
    )
    folder_path = "Twisty_testing/population_data/graphs"
    directory = Path(folder_path)
    os.makedirs(directory, exist_ok=True)
    path = directory / f"{ind.id}_graph.json"
    save_graph_as_json(graph, path)
    ind.genotype = [type_probs.tolist(), conn_probs.tolist(), rotation_probs.tolist()]
    ind.graph = graph
    ind.twisty = twisty
    
    return ind

def evaluate_population(population: list[Individual], rng, correct_for_bounding: bool, spawn_z: float, spawn_xy: list[float] ,target_pos: list[float], brain_type: str) -> None:
    """Evaluate fitness for all individuals in population."""
    for individual in population:
        
        robot = construct_mjspec_from_graph(individual.graph)
        if robot is None: # type: ignore
            individual.fitness = 0.0
        fitness = run_for_fitness(robot, individual, correct_for_bounding, rng, spawn_z, spawn_xy ,target_pos, brain_type)
        individual.fitness = fitness # type: ignore
        console.log(f"Individual (twisty={individual.twisty}) fitness: {fitness:.3f}") # type: ignore
        # except Exception as e:
        #     console.log(f"Error evaluating individual: {e}")
        #     individual.fitness = 0.0
if __name__ == "__main__":
    # Test several times
    seed = 50
    rng = np.random.default_rng(seed)
    i = create_individual(False, 1, rng, 20)
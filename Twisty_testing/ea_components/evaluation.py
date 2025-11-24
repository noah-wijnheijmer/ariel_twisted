from robot_body.modules.core import CoreModule
import mujoco
from rich.console import Console
import numpy as np
from typing import Any
from simulation.environments._simple_flat import SimpleFlatWorld
from simulation.cpg.sf_cpg import CPGSensoryFeedback, sf_policy
from simulation.cpg.na_cpg import (
    NaCPG, create_fully_connected_adjacency, na_policy
)

console = Console()
SEED = 40
RNG = np.random.default_rng(SEED)

def run_for_fitness(robot: CoreModule, individual: Any, correct_for_bounding: bool, spawn_z: float, spawn_xy:list[float], target_pos: list[float], brain_type: str = "sf_cpg") -> Any:
    """Modified run function that returns fitness based on distance to target."""
    try:
        if individual.fitness is not None:
            return individual.fitness
    except:   
        # Setup (same as existing run())
        mujoco.set_mjcb_control(None)
        world = SimpleFlatWorld()
    
        for i in range(len(robot.spec.geoms)):
            robot.spec.geoms[i].rgba[-1] = 0.5
    
        world.spawn(robot.spec, spawn_z=spawn_z, spawn_xy=spawn_xy, correct_for_bounding_box=correct_for_bounding)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
    
        # Number of actuators and DoFs
        console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")
    
        # Create CPG controller
        if brain_type == "sf_cpg":
            weight_matrix = RNG.uniform(-0.1, 0.1, size=(model.nu, model.nu))
            cpg = CPGSensoryFeedback(
                num_neurons=int(model.nu),
                sensory_term=-0.0,
                _lambda=0.01,
                coupling_weights=weight_matrix,
            )
            cpg.reset()
            individual.brain_genotype = cpg.c.tolist()
            mujoco.set_mjcb_control(lambda m, d: sf_policy(m, d, cpg=cpg))
        elif brain_type == "na_cpg":
            adj_dict = create_fully_connected_adjacency(model.nu)
            cpg = NaCPG(adj_dict, angle_tracking=True)
            cpg.reset()
            gen = cpg.get_flat_params()
            individual.brain_genotype = gen
            mujoco.set_mjcb_control(lambda m, d: na_policy(m, d, cpg=cpg))
    
        # Run simulation for target-seeking fitness
        simulation_time = 15.0  # seconds
        steps = int(simulation_time / model.opt.timestep)
        individual.time_alive = 0
        fitness = 0
        # Run simulation for fitness (no video)
        simulation_time = 30.0  # seconds
        step_one_sec = int(1/model.opt.timestep)
        steps = int(simulation_time / model.opt.timestep)
        for step in range(steps):
            individual.time_alive = step/step_one_sec
            mujoco.mj_step(model, data)
    
        # Calculate fitness based on final distance to target
        final_position = data.xpos[1][:2].copy()  # x, y coordinates only     
        target_position = np.array([target_pos[0], target_pos[1]])  # x, y from target
        distance_to_target = np.linalg.norm(final_position - target_position)
        fitness = 1.0 / (1.0 + distance_to_target)
        return fitness
"""
    # target_position = np.array([target_pos[0], target_pos[1]])  # x, y from target
    # print(final_position)
    # # Distance to target (lower is better)
    # distance_to_target = np.linalg.norm(final_position - target_position)
    
    # # Simple inverse distance fitness (higher fitness = closer to target)
    # # fitness = 1.0 / (1.0 + distance_to_target)
    
    # return distance_to_target
"""
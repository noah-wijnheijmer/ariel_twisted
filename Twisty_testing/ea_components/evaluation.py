from robot_body.modules.core import CoreModule
import mujoco
from rich.console import Console
import numpy as np
from typing import Any
from simulation.environments._simple_flat import SimpleFlatWorld
from simulation.cpg.na_cpg import (
    NaCPG, create_fully_connected_adjacency, policy
)

console = Console()

def run_for_fitness(robot: CoreModule, individual: Any, spawn_pos: list[float], target_pos: list[float]) -> Any:
    """Modified run function that returns fitness based on distance to target."""
    
    # Setup (same as existing run())
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5
    
    world.spawn(robot.spec, position=spawn_pos)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    
    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")
    
    # Create CPG controller
    adj_dict = create_fully_connected_adjacency(model.nu)
    cpg = NaCPG(adj_dict, angle_tracking=True)
    cpg.reset()
    gen = cpg.get_flat_params()
    individual.brain_genotype = gen
    mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg=cpg))
    
    # Run simulation for target-seeking fitness
    simulation_time = 15.0  # seconds
    steps = int(simulation_time / model.opt.timestep)
    individual.time_alive = 0
    
    for step in range(steps):
        individual.time_alive = step / (1/model.opt.timestep)
        mujoco.mj_step(model, data)
    
    # Calculate fitness based on final distance to target
    final_position = data.xpos[1][:2].copy()  # x, y coordinates only
    target_position = np.array([target_pos[0], target_pos[1]])  # x, y from target
    
    # Distance to target (lower is better)
    distance_to_target = np.linalg.norm(final_position - target_position)
    
    # Simple inverse distance fitness (higher fitness = closer to target)
    fitness = 1.0 / (1.0 + distance_to_target)
    
    return fitness
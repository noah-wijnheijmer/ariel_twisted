"""
Visualize all gecko morphologies for report.
Creates schematic diagrams showing the structure of each gecko variant.
"""

import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import os

# Import all gecko morphologies
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_untwisted import gecko_untwisted
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_good import gecko_good
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_front import gecko_front
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist import gecko_doubletwist
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko_doubletwist_turtle import gecko_doubletwist_turtle

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


def extract_morphology_info(gecko_model):
    """Extract structural information from a gecko morphology."""
    world = SimpleFlatWorld()
    gecko_core = gecko_model()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1], correct_for_bounding_box=True)
    
    model = world.spec.compile()
    data = mj.MjData(model)
    
    info = {
        'name': gecko_model.__name__,
        'num_bodies': model.nbody,
        'num_joints': model.njnt,
        'num_actuators': model.nu,
        'qpos_size': len(data.qpos),
        'bodies': [],
        'joints': [],
        'actuators': [],
    }
    
    # Extract body information
    for i in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if body_name:
            info['bodies'].append(body_name)
    
    # Extract joint information
    for i in range(model.njnt):
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            joint_type = model.jnt_type[i]
            info['joints'].append({'name': joint_name, 'type': joint_type})
    
    # Extract actuator information
    for i in range(model.nu):
        actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name:
            info['actuators'].append(actuator_name)
    
    return info, model, data


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def draw_orientation_arrows(ax, center, quaternion, arrow_length=0.04, linewidth=2):
    """
    Draw orientation arrows (X, Y, Z axes) for a body.
    
    Parameters
    ----------
    ax : Axes3D
        The 3D axes to draw on
    center : array-like
        The center position [x, y, z]
    quaternion : array-like
        Quaternion [w, x, y, z] for rotation
    arrow_length : float
        Length of the arrows
    linewidth : float
        Line width for arrows
    """
    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_rotation_matrix(quaternion)
    
    # Local axis directions
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # Rotate axes to world frame
    x_dir = rot_matrix @ x_axis
    y_dir = rot_matrix @ y_axis
    z_dir = rot_matrix @ z_axis
    
    # Draw arrows
    # X-axis: Red
    ax.quiver(center[0], center[1], center[2],
              x_dir[0] * arrow_length, x_dir[1] * arrow_length, x_dir[2] * arrow_length,
              color='red', arrow_length_ratio=0.3, linewidth=linewidth, alpha=0.8)
    
    # Y-axis: Green
    ax.quiver(center[0], center[1], center[2],
              y_dir[0] * arrow_length, y_dir[1] * arrow_length, y_dir[2] * arrow_length,
              color='lime', arrow_length_ratio=0.3, linewidth=linewidth, alpha=0.8)
    
    # Z-axis: Blue
    ax.quiver(center[0], center[1], center[2],
              z_dir[0] * arrow_length, z_dir[1] * arrow_length, z_dir[2] * arrow_length,
              color='cyan', arrow_length_ratio=0.3, linewidth=linewidth, alpha=0.8)


def draw_cube(ax, center, size, color, quaternion=None, alpha=0.6, edgecolor='black', linewidth=1):
    """
    Draw a 3D rectangular block (cuboid) at given center with given dimensions.
    
    Parameters
    ----------
    ax : Axes3D
        The 3D axes to draw on
    center : array-like
        The center position [x, y, z]
    size : array-like
        The dimensions [dx, dy, dz]
    color : str
        The color of the block
    quaternion : array-like, optional
        Quaternion [w, x, y, z] for rotation. If None, no rotation applied.
    alpha : float
        Transparency
    edgecolor : str
        Edge color
    linewidth : float
        Line width for edges
    """
    x, y, z = center
    dx, dy, dz = size
    
    # Define the 8 vertices of the cube in local coordinates
    local_vertices = np.array([
        [-dx/2, -dy/2, -dz/2],
        [+dx/2, -dy/2, -dz/2],
        [+dx/2, +dy/2, -dz/2],
        [-dx/2, +dy/2, -dz/2],
        [-dx/2, -dy/2, +dz/2],
        [+dx/2, -dy/2, +dz/2],
        [+dx/2, +dy/2, +dz/2],
        [-dx/2, +dy/2, +dz/2],
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None:
        rot_matrix = quaternion_to_rotation_matrix(quaternion)
        rotated_vertices = local_vertices @ rot_matrix.T
    else:
        rotated_vertices = local_vertices
    
    # Translate to world coordinates
    vertices = rotated_vertices + np.array([x, y, z])
    
    # Define the 6 faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # bottom
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create the 3D polygon collection
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edgecolor, linewidths=linewidth)
    ax.add_collection3d(poly)


def render_morphology_3d(gecko_model, save_path):
    """Render a 3D view of the gecko morphology with proper block representations."""
    world = SimpleFlatWorld()
    gecko_core = gecko_model()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1], correct_for_bounding_box=True)
    
    model = world.spec.compile()
    data = mj.MjData(model)
    
    # Step the simulation to stabilize
    for _ in range(100):
        mj.mj_step(model, data)
    
    # Calculate the center of mass for centering
    all_positions = np.array([data.xpos[i] for i in range(model.nbody)])
    center_of_mass = np.mean(all_positions, axis=0)
    
    # Create 3D plot with single isometric view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store arrow information to draw AFTER blocks
    arrow_data = []
    
    # Extract body information and draw blocks FIRST
    for i in range(model.nbody):
        body_pos = data.xpos[i]
        
        # Get body name
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        
        # Determine block type, size, and color based on body name
        if body_name and 'core' in body_name.lower():
            # Core: Large cubic block
            color = '#d32f2f'  # Deep red
            block_size = [0.06, 0.06, 0.06]
        elif body_name and 'brick' in body_name.lower():
            # Brick: Medium rectangular block
            color = '#1976d2'  # Deep blue
            block_size = [0.04, 0.04, 0.04]
        elif body_name and 'hinge' in body_name.lower():
            # Hinge: Small rectangular block
            color = '#388e3c'  # Deep green
            block_size = [0.025, 0.025, 0.04]  # Rectangular for hinge
        elif body_name and ('leg' in body_name.lower() or 'flipper' in body_name.lower()):
            # Leg/Flipper: Medium block
            color = '#0288d1'  # Light blue
            block_size = [0.035, 0.035, 0.035]
        elif body_name and ('neck' in body_name.lower() or 'abdomen' in body_name.lower() 
                            or 'spine' in body_name.lower() or 'butt' in body_name.lower()):
            # Body segments: Medium rectangular block
            color = '#7b1fa2'  # Purple
            block_size = [0.04, 0.04, 0.04]
        else:
            # Unknown: Gray block
            color = '#757575'
            block_size = [0.03, 0.03, 0.03]
        
        # Draw the block with rotation (slightly more transparent)
        quaternion = data.xquat[i]  # Get quaternion for this body
        draw_cube(ax, body_pos, block_size, color, quaternion=quaternion, alpha=0.5, linewidth=0.5)
        
        # Store arrow data for leg components to draw AFTER all blocks
        if body_name and ('leg' in body_name.lower() or 'flipper' in body_name.lower() or 
                          'hinge' in body_name.lower()):
            arrow_data.append((body_pos, quaternion))
    
    # Draw connections between bodies
    for i in range(1, model.nbody):
        parent_id = model.body_parentid[i]
        if parent_id >= 0:
            pos_child = data.xpos[i]
            pos_parent = data.xpos[parent_id]
            ax.plot([pos_parent[0], pos_child[0]], 
                   [pos_parent[1], pos_child[1]], 
                   [pos_parent[2], pos_child[2]], 
                   'k-', alpha=0.2, linewidth=1)
    
    # NOW draw orientation arrows on top of blocks
    for body_pos, quaternion in arrow_data:
        draw_orientation_arrows(ax, body_pos, quaternion, arrow_length=0.05, linewidth=2.5)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'{gecko_model.__name__}', fontsize=16, fontweight='bold', pad=20)
    
    # Set isometric view
    ax.view_init(elev=30, azim=45)
    
    # Center the view on the center of mass
    extent = 0.3  # Range around center
    ax.set_xlim([center_of_mass[0] - extent, center_of_mass[0] + extent])
    ax.set_ylim([center_of_mass[1] - extent, center_of_mass[1] + extent])
    ax.set_zlim([center_of_mass[2] - extent, center_of_mass[2] + extent])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend for orientation arrows
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2.5, label='X-axis'),
        Line2D([0], [0], color='lime', linewidth=2.5, label='Y-axis'),
        Line2D([0], [0], color='cyan', linewidth=2.5, label='Z-axis')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              title='Orientation Arrows', title_fontsize=11)
    
    plt.suptitle(f'{gecko_model.__name__} Morphology', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 3D visualization: {save_path}")


def create_schematic_diagram(gecko_model, info, save_path):
    """Create a simplified schematic diagram of the morphology."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Title
    ax.text(0.5, 0.95, f'{info["name"]} Morphology', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            transform=ax.transAxes)
    
    # Structural information box
    info_text = f"""
Structural Information:
━━━━━━━━━━━━━━━━━━━━━
Bodies:      {info['num_bodies']}
Joints:      {info['num_joints']}
Actuators:   {info['num_actuators']}
Input Size:  {info['qpos_size']}
Output Size: {info['num_actuators']}
    """
    
    ax.text(0.02, 0.88, info_text.strip(), 
            ha='left', va='top', fontsize=11, 
            family='monospace',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Body components list
    y_offset = 0.65
    ax.text(0.02, y_offset, 'Body Components:', 
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    
    y_offset -= 0.04
    for i, body in enumerate(info['bodies'][:20]):  # Limit to first 20
        ax.text(0.02, y_offset - i * 0.025, f'  • {body}', 
                ha='left', va='top', fontsize=9, family='monospace',
                transform=ax.transAxes)
    
    if len(info['bodies']) > 20:
        ax.text(0.02, y_offset - 20 * 0.025, f'  ... and {len(info["bodies"]) - 20} more', 
                ha='left', va='top', fontsize=9, family='monospace', style='italic',
                transform=ax.transAxes)
    
    # Joint types
    y_offset = 0.65
    ax.text(0.52, y_offset, 'Joints & Actuators:', 
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    
    y_offset -= 0.04
    joint_types = {}
    for joint in info['joints']:
        jtype = joint['type']
        joint_types[jtype] = joint_types.get(jtype, 0) + 1
    
    for i, (jtype, count) in enumerate(joint_types.items()):
        type_name = {0: 'Free', 1: 'Ball', 2: 'Slide', 3: 'Hinge'}.get(jtype, 'Unknown')
        ax.text(0.52, y_offset - i * 0.025, f'  • {type_name}: {count}', 
                ha='left', va='top', fontsize=9, family='monospace',
                transform=ax.transAxes)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label='Core Module'),
        mpatches.Patch(color='blue', label='Leg/Flipper'),
        mpatches.Patch(color='green', label='Hinge/Joint'),
        mpatches.Patch(color='gray', label='Other Components')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved schematic diagram: {save_path}")


def create_comparison_plot(all_infos, save_path):
    """Create a comparison plot of all morphologies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    morphology_names = list(all_infos.keys())
    
    for idx, (name, info) in enumerate(all_infos.items()):
        ax = axes[idx]
        
        # Create bar chart of structural properties
        properties = ['Bodies', 'Joints', 'Actuators', 'Input\nSize', 'Output\nSize']
        values = [
            info['num_bodies'],
            info['num_joints'],
            info['num_actuators'],
            info['qpos_size'],
            info['num_actuators']
        ]
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        bars = ax.bar(properties, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(values) * 1.2)
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Gecko Morphology Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved comparison plot: {save_path}")


def create_detailed_stats_table(all_infos, save_path):
    """Create a detailed statistics table."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    morphologies = list(all_infos.keys())
    
    table_data = []
    headers = ['Morphology', 'Bodies', 'Joints', 'Actuators', 'Input Size', 'Output Size', 
               'Hinge Joints', 'Total DOF']
    
    for name, info in all_infos.items():
        # Count hinge joints
        hinge_count = sum(1 for j in info['joints'] if j['type'] == 3)
        
        row = [
            name,
            info['num_bodies'],
            info['num_joints'],
            info['num_actuators'],
            info['qpos_size'],
            info['num_actuators'],
            hinge_count,
            info['qpos_size']  # Approximation
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.18, 0.10, 0.10, 0.12, 0.12, 0.12, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
    
    plt.title('Detailed Morphology Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved statistics table: {save_path}")


def main():
    """Main function to visualize all gecko morphologies."""
    
    # Create output directory
    output_dir = './__morphology_visualizations__'
    os.makedirs(output_dir, exist_ok=True)
    
    # List of all gecko morphologies
    gecko_models = [
        gecko,
        gecko_untwisted,
        gecko_good,
        gecko_doubletwist,
        gecko_doubletwist_turtle,
        gecko_front,
    ]
    
    all_infos = {}
    
    print("="*60)
    print("Visualizing Gecko Morphologies for Report")
    print("="*60)
    
    # Process each morphology
    for gecko_model in gecko_models:
        print(f"\nProcessing: {gecko_model.__name__}")
        print("-" * 40)
        
        # Extract information
        info, model, data = extract_morphology_info(gecko_model)
        all_infos[gecko_model.__name__] = info
        
        # Print summary
        print(f"  Bodies: {info['num_bodies']}")
        print(f"  Joints: {info['num_joints']}")
        print(f"  Actuators: {info['num_actuators']}")
        print(f"  Input Size: {info['qpos_size']}")
        
        # Create 3D visualization
        viz_3d_path = os.path.join(output_dir, f'{gecko_model.__name__}_3d_views.png')
        render_morphology_3d(gecko_model, viz_3d_path)
        
        # Create schematic diagram
        schematic_path = os.path.join(output_dir, f'{gecko_model.__name__}_schematic.png')
        create_schematic_diagram(gecko_model, info, schematic_path)
    
    print("\n" + "="*60)
    print("Creating Comparison Visualizations")
    print("="*60)
    
    # Create comparison plot
    comparison_path = os.path.join(output_dir, 'all_morphologies_comparison.png')
    create_comparison_plot(all_infos, comparison_path)
    
    # Create detailed statistics table
    stats_path = os.path.join(output_dir, 'morphology_statistics_table.png')
    create_detailed_stats_table(all_infos, stats_path)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"All files saved to: {output_dir}")
    print("="*60)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    for name, info in all_infos.items():
        print(f"{name:30} | Bodies: {info['num_bodies']:3} | Joints: {info['num_joints']:3} | "
              f"Actuators: {info['num_actuators']:3} | qpos: {info['qpos_size']:3}")


if __name__ == "__main__":
    main()

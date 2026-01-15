import glob
import os
import time

import open3d as o3d

# Directory where your meshes are saved
mesh_dir = os.getcwd()  # or replace with your mesh folder

# Collect all mesh files (assuming they are named like mesh0.ply, mesh1.ply, ...)
mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "mesh*.ply")))

if not mesh_files:
    raise RuntimeError("No mesh files found!")

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Mesh Movie", width=800, height=600)

# Load first mesh to initialize
mesh = o3d.io.read_triangle_mesh(mesh_files[0])
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

# Animation loop
try:
    for mesh_path in mesh_files:
        # Load next mesh
        new_mesh = o3d.io.read_triangle_mesh(mesh_path)
        new_mesh.compute_vertex_normals()

        # Update the existing geometry
        vis.update_geometry(mesh)
        mesh.vertices = new_mesh.vertices
        mesh.triangles = new_mesh.triangles
        mesh.vertex_normals = new_mesh.vertex_normals
        mesh.triangle_normals = new_mesh.triangle_normals

        # Update renderer
        vis.poll_events()
        vis.update_renderer()

        # Pause between frames (seconds)
        time.sleep(0.5)

finally:
    vis.destroy_window()

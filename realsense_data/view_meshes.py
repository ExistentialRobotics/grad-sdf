import glob
import os

import open3d as o3d

# Directory containing your saved meshes
mesh_dir = os.getcwd()  # or wherever your meshes are
mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "mesh_prior*.ply")))
print(mesh_files)

if not mesh_files:
    raise RuntimeError("No mesh files found!")

# Load the first mesh
current_index = 0
mesh = o3d.io.read_triangle_mesh(mesh_files[current_index])
mesh.compute_vertex_normals()

# Create the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Mesh Viewer", width=800, height=600)
vis.add_geometry(mesh)


# Key callback to swap to the next mesh
def next_mesh(vis):
    global current_index, mesh, mesh_files
    current_index += 1
    if current_index >= len(mesh_files):
        print("Reached last mesh.")
        return

    # Load next mesh
    new_mesh = o3d.io.read_triangle_mesh(mesh_files[current_index])
    new_mesh.compute_vertex_normals()

    # Update mesh data
    mesh.vertices = new_mesh.vertices
    mesh.triangles = new_mesh.triangles
    mesh.vertex_normals = new_mesh.vertex_normals
    mesh.triangle_normals = new_mesh.triangle_normals

    vis.update_geometry(mesh)
    return False  # False = continue running


# Bind the spacebar to advance the mesh
vis.register_key_callback(ord(" "), next_mesh)

# Run the viewer
vis.run()
vis.destroy_window()

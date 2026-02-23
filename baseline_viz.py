"""
Baseline visualization: 18 sliders directly controlling the 18 DOF
(x, y positions of 9 interior vertices) of a 4x4 planar quad mesh.

This is the ground truth — full direct control over every free vertex.
The goal is to later replace these 18 sliders with fewer latent dimensions
that still produce meaningful deformations.
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# =============================================================================
# Mesh definition
# =============================================================================

def make_quad_grid(nx=4, ny=4):
    """
    Create a regular nx×ny quad grid.
    Returns:
        vertices: (num_verts, 3) array — z=0 for planar
        faces: (num_faces, 4) array of vertex indices
        interior_mask: boolean array, True for interior (free) vertices
    """
    num_verts_x = nx + 1  # 5
    num_verts_y = ny + 1  # 5

    # Vertex positions on a regular grid
    vertices = []
    for j in range(num_verts_y):
        for i in range(num_verts_x):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    # Quad faces (each quad defined by 4 vertex indices)
    faces = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * num_verts_x + i
            v1 = v0 + 1
            v2 = v0 + num_verts_x + 1
            v3 = v0 + num_verts_x
            faces.append([v0, v1, v2, v3])
    faces = np.array(faces, dtype=np.int32)

    # Interior mask: vertex is interior if not on the boundary
    interior_mask = np.zeros(len(vertices), dtype=bool)
    for j in range(num_verts_y):
        for i in range(num_verts_x):
            idx = j * num_verts_x + i
            if 0 < i < nx and 0 < j < ny:
                interior_mask[idx] = True

    return vertices, faces, interior_mask


# =============================================================================
# Energy functions
# =============================================================================

def get_all_edges(faces):
    """Extract unique edges from quad faces."""
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            i, j = int(face[k]), int(face[(k + 1) % n])
            edge_set.add((min(i, j), max(i, j)))
    return np.array(sorted(edge_set), dtype=np.int32)


def compute_edge_energy(vertices, edges, rest_lengths):
    """
    Spring energy: E = Σ (||v_i - v_j|| - L_0)^2
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    return np.sum((lengths - rest_lengths) ** 2)


def compute_quad_area_energy(vertices, faces, rest_areas):
    """
    Area preservation: E = Σ (A_q - A_0)^2
    Uses the shoelace formula for quad area (sum of two triangles).
    """
    energy = 0.0
    for idx, face in enumerate(faces):
        v0, v1, v2, v3 = vertices[face]
        # Split quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
        area1 = 0.5 * abs(np.cross(v1[:2] - v0[:2], v2[:2] - v0[:2]))
        area2 = 0.5 * abs(np.cross(v2[:2] - v0[:2], v3[:2] - v0[:2]))
        area = area1 + area2
        energy += (area - rest_areas[idx]) ** 2
    return energy


# =============================================================================
# Main
# =============================================================================

def main():
    # Build mesh
    vertices_rest, faces, interior_mask = make_quad_grid(4, 4)
    edges = get_all_edges(faces)

    # Rest lengths and areas
    rest_lengths = np.linalg.norm(
        vertices_rest[edges[:, 1]] - vertices_rest[edges[:, 0]], axis=1
    )
    rest_areas = np.ones(len(faces), dtype=np.float64)  # unit squares

    # Interior vertex indices and their rest positions
    interior_indices = np.where(interior_mask)[0]
    num_interior = len(interior_indices)  # should be 9
    print(f"Interior vertices: {num_interior} → {2 * num_interior} DOF")
    print(f"Interior indices: {interior_indices}")

    # Current vertex positions (mutable copy)
    vertices = vertices_rest.copy()

    # Slider state: offsets from rest position for each interior vertex DOF
    # Organized as [v0_dx, v0_dy, v1_dx, v1_dy, ..., v8_dx, v8_dy]
    offsets = np.zeros(2 * num_interior, dtype=np.float64)

    # Slider range
    SLIDER_RANGE = 1.5  # allow ±1.5 units of displacement

    # Vertex labels for display
    row_col = {}
    for idx in interior_indices:
        j = idx // 5
        i = idx % 5
        row_col[idx] = (i, j)

    # ---- Polyscope setup ----
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    # Register the mesh
    mesh = ps.register_surface_mesh(
        "quad_grid", vertices, faces, edge_width=2.0
    )
    mesh.set_color((0.2, 0.5, 0.9))

    # Also show vertices as a point cloud for clarity
    cloud = ps.register_point_cloud("vertices", vertices, radius=0.04)
    # Color: boundary = red, interior = green
    colors = np.zeros((len(vertices), 3))
    colors[~interior_mask] = [0.8, 0.2, 0.2]  # boundary: red
    colors[interior_mask] = [0.2, 0.8, 0.2]   # interior: green
    cloud.add_color_quantity("type", colors, enabled=True)

    def callback():
        nonlocal offsets, vertices

        # --- Energy display ---
        energy_edge = compute_edge_energy(vertices, edges, rest_lengths)
        energy_area = compute_quad_area_energy(vertices, faces, rest_areas)

        psim.SetNextWindowPos((10.0, 10.0))
        psim.SetNextWindowSize((350.0, 0.0))
        _, _ = psim.Begin("Energy", True)
        psim.Text(f"Edge energy:  {energy_edge:.6f}")
        psim.Text(f"Area energy:  {energy_area:.6f}")
        psim.Text(f"Total energy: {energy_edge + energy_area:.6f}")
        psim.End()

        # --- Sliders ---
        psim.SetNextWindowPos((10.0, 130.0))
        psim.SetNextWindowSize((350.0, 0.0))
        _, _ = psim.Begin("DOF Sliders (18)", True)

        changed = False
        for k, vid in enumerate(interior_indices):
            col, row = row_col[vid]
            # X offset
            c, val = psim.SliderFloat(
                f"v({col},{row}) dx##{k*2}",
                offsets[2 * k],
                v_min=-SLIDER_RANGE,
                v_max=SLIDER_RANGE,
            )
            if c:
                offsets[2 * k] = val
                changed = True
            # Y offset
            c, val = psim.SliderFloat(
                f"v({col},{row}) dy##{k*2+1}",
                offsets[2 * k + 1],
                v_min=-SLIDER_RANGE,
                v_max=SLIDER_RANGE,
            )
            if c:
                offsets[2 * k + 1] = val
                changed = True

        # Reset button
        if psim.Button("Reset to rest"):
            offsets[:] = 0.0
            changed = True

        psim.End()

        # --- Update mesh if anything changed ---
        if changed:
            vertices = vertices_rest.copy()
            for k, vid in enumerate(interior_indices):
                vertices[vid, 0] += offsets[2 * k]
                vertices[vid, 1] += offsets[2 * k + 1]
            mesh.update_vertex_positions(vertices)
            cloud.update_point_positions(vertices)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

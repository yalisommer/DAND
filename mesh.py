"""
Shared mesh definition and energy functions for the Planar Quad Toy project.
Provides both NumPy (for visualization) and PyTorch (for training) variants.
Supports 2D (planar) and 3D (with bending) modes.
"""

import numpy as np
import torch


# =============================================================================
# Mesh construction
# =============================================================================

def make_quad_grid(nx=4, ny=4):
    """
    Create a regular nx×ny quad grid.
    Returns:
        vertices: (num_verts, 3) float64 array — z=0 for planar
        faces: (num_faces, 4) int32 array of vertex indices
        interior_mask: boolean array, True for interior (free) vertices
    """
    num_verts_x = nx + 1
    num_verts_y = ny + 1

    vertices = []
    for j in range(num_verts_y):
        for i in range(num_verts_x):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * num_verts_x + i
            v1 = v0 + 1
            v2 = v0 + num_verts_x + 1
            v3 = v0 + num_verts_x
            faces.append([v0, v1, v2, v3])
    faces = np.array(faces, dtype=np.int32)

    interior_mask = np.zeros(len(vertices), dtype=bool)
    for j in range(num_verts_y):
        for i in range(num_verts_x):
            idx = j * num_verts_x + i
            if 0 < i < nx and 0 < j < ny:
                interior_mask[idx] = True

    return vertices, faces, interior_mask


def make_open_box(n=2):
    """
    Create an open box (cube with missing bottom face), each face subdivided n×n.
    The cube spans [0, n] in each axis. Bottom lip is at z=0.

    For n=2: 25 unique vertices, 20 quad faces, 8 bottom-lip vertices.

    Returns:
        vertices: (num_verts, 3) float64 array
        faces: (num_faces, 4) int32 array of vertex indices
        interior_mask: boolean array, True for non-bottom-lip vertices
        bottom_lip_mask: boolean array, True for vertices at z=0
    """
    vertex_map = {}  # (x, y, z) -> index

    def add_vertex(x, y, z):
        key = (float(x), float(y), float(z))
        if key not in vertex_map:
            vertex_map[key] = len(vertex_map)
        return vertex_map[key]

    faces_list = []

    def add_face_quads(grid):
        """grid[j][i] = vertex index, j=row, i=col. Adds n×n quads."""
        for j in range(n):
            for i in range(n):
                faces_list.append([grid[j][i], grid[j][i+1],
                                   grid[j+1][i+1], grid[j+1][i]])

    s = n  # side length

    # Top face (z=s): rows along y, cols along x
    grid = [[add_vertex(i, j, s) for i in range(s+1)] for j in range(s+1)]
    add_face_quads(grid)

    # Front face (y=0): rows along z (bottom-up), cols along x
    grid = [[add_vertex(i, 0, j) for i in range(s+1)] for j in range(s+1)]
    add_face_quads(grid)

    # Back face (y=s): rows along z (bottom-up), cols along x (reversed for outward normal)
    grid = [[add_vertex(s-i, s, j) for i in range(s+1)] for j in range(s+1)]
    add_face_quads(grid)

    # Left face (x=0): rows along z (bottom-up), cols along y (reversed for outward normal)
    grid = [[add_vertex(0, s-i, j) for i in range(s+1)] for j in range(s+1)]
    add_face_quads(grid)

    # Right face (x=s): rows along z (bottom-up), cols along y
    grid = [[add_vertex(s, i, j) for i in range(s+1)] for j in range(s+1)]
    add_face_quads(grid)

    num_verts = len(vertex_map)
    vertices = np.zeros((num_verts, 3), dtype=np.float64)
    for (x, y, z), idx in vertex_map.items():
        vertices[idx] = [x, y, z]

    faces_arr = np.array(faces_list, dtype=np.int32)

    bottom_lip_mask = vertices[:, 2] < 1e-8
    interior_mask = ~bottom_lip_mask

    return vertices, faces_arr, interior_mask, bottom_lip_mask


def make_semicircle_tri(n_segments=8, radius=2.0):
    """
    Create a simple 2D semicircle fan mesh made of triangles.

    - Vertex 0: center at (0, 0, 0)
    - Vertices 1..n_segments+1: points along semicircle from angle 0..π
    - Faces: triangles (center, i, i+1) forming a fan

    Returns:
        vertices: (num_verts, 3) float64 array
        faces: (num_faces, 3) int32 array of vertex indices
        interior_mask: boolean array (all True — no anchored boundary here)
    """
    vertices = []
    # Center
    vertices.append([0.0, 0.0, 0.0])
    # Arc points
    for k in range(n_segments + 1):
        theta = np.pi * k / n_segments  # 0..π
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        vertices.append([x, y, 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for k in range(n_segments):
        # Triangle: center (0), arc[k+1], arc[k+2]
        faces.append([0, k + 1, k + 2])
    faces = np.array(faces, dtype=np.int32)

    interior_mask = np.ones(len(vertices), dtype=bool)
    return vertices, faces, interior_mask


def get_all_edges(faces):
    """Extract unique edges from quad faces."""
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            i, j = int(face[k]), int(face[(k + 1) % n])
            edge_set.add((min(i, j), max(i, j)))
    return np.array(sorted(edge_set), dtype=np.int32)


# =============================================================================
# NumPy energy functions (for visualization)
# =============================================================================

def compute_edge_energy_np(vertices, edges, rest_lengths):
    """Spring energy: E = Σ (||v_i - v_j|| - L_0)^2. Works for 2D or 3D."""
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    return np.sum((lengths - rest_lengths) ** 2)


def compute_quad_area_energy_np(vertices, faces, rest_areas):
    """Area preservation: E = Σ (A_face - A_0)^2. Works for tris or quads, 2D or 3D."""
    energy = 0.0
    for idx, face in enumerate(faces):
        if len(face) == 4:
            v0, v1, v2, v3 = vertices[face]
            # Use 3D cross product magnitude for area (two triangles)
            cross1 = np.cross(v1 - v0, v2 - v0)
            area1 = 0.5 * np.linalg.norm(cross1)
            cross2 = np.cross(v2 - v0, v3 - v0)
            area2 = 0.5 * np.linalg.norm(cross2)
            area = area1 + area2
        elif len(face) == 3:
            v0, v1, v2 = vertices[face]
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross)
        else:
            raise ValueError(f"Unsupported face size {len(face)} for area energy")
        energy += (area - rest_areas[idx]) ** 2
    return energy


def compute_planarity_energy_np(vertices, faces):
    """
    Per-quad planarity energy (NumPy).
    For each quad (v0,v1,v2,v3), planarity violation = volume of tetrahedron.
    E = Σ [ (v3-v0) · ((v1-v0) × (v2-v0)) ]^2
    Zero when all 4 vertices are coplanar.
    """
    energy = 0.0
    for face in faces:
        if len(face) == 4:
            v0, v1, v2, v3 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            vol = np.dot(v3 - v0, normal)
            energy += vol ** 2
        elif len(face) == 3:
            # Triangles are always planar; treat planarity energy as zero.
            continue
        else:
            raise ValueError(f"Unsupported face size {len(face)} for planarity energy")
    return energy


def compute_flatness_penalty_np(vertices):
    """Direct z² flatness penalty (NumPy). E = Σ z_i²"""
    return np.sum(vertices[:, 2] ** 2)


def compute_max_z_deviation_np(vertices):
    """Max absolute z-coordinate (for display)."""
    return np.max(np.abs(vertices[:, 2]))


# =============================================================================
# PyTorch energy functions (for training — differentiable)
# =============================================================================

def compute_edge_energy_torch(all_verts, edges, rest_lengths):
    """
    Spring energy in PyTorch. Works for (N, 2) or (N, 3) vertex arrays.
    """
    v0 = all_verts[edges[:, 0]]
    v1 = all_verts[edges[:, 1]]
    lengths = torch.norm(v1 - v0, dim=1)
    return torch.sum((lengths - rest_lengths) ** 2)


def compute_quad_area_energy_torch(all_verts, faces, rest_areas):
    """
    Area preservation energy in PyTorch. Works for tris or quads, (N, 2) or (N, 3).
    """
    dim = all_verts.shape[1]
    if faces.shape[1] == 4:
        v0 = all_verts[faces[:, 0]]
        v1 = all_verts[faces[:, 1]]
        v2 = all_verts[faces[:, 2]]
        v3 = all_verts[faces[:, 3]]

        if dim == 2:
            # 2D cross product (scalar)
            cross1 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                     (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
            area1 = 0.5 * torch.abs(cross1)
            cross2 = (v2[:, 0] - v0[:, 0]) * (v3[:, 1] - v0[:, 1]) - \
                     (v2[:, 1] - v0[:, 1]) * (v3[:, 0] - v0[:, 0])
            area2 = 0.5 * torch.abs(cross2)
        else:
            # 3D cross product (vector)
            cross1 = torch.cross(v1 - v0, v2 - v0, dim=1)
            area1 = 0.5 * torch.norm(cross1, dim=1)
            cross2 = torch.cross(v2 - v0, v3 - v0, dim=1)
            area2 = 0.5 * torch.norm(cross2, dim=1)
        areas = area1 + area2
    elif faces.shape[1] == 3:
        v0 = all_verts[faces[:, 0]]
        v1 = all_verts[faces[:, 1]]
        v2 = all_verts[faces[:, 2]]
        if dim == 2:
            cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                    (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
            areas = 0.5 * torch.abs(cross)
        else:
            cross = torch.cross(v1 - v0, v2 - v0, dim=1)
            areas = 0.5 * torch.norm(cross, dim=1)
    else:
        raise ValueError(f"Unsupported face size {faces.shape[1]} for area energy")

    return torch.sum((areas - rest_areas) ** 2)


def compute_planarity_energy_torch(all_verts_3d, faces):
    """
    Per-quad planarity energy (PyTorch, differentiable).
    For each quad (v0,v1,v2,v3): violation = (v3-v0) · ((v1-v0) × (v2-v0))
    This is the signed volume of the tetrahedron — zero when coplanar.
    E = Σ vol_i^2
    """
    if faces.shape[1] == 4:
        v0 = all_verts_3d[faces[:, 0]]  # (F, 3)
        v1 = all_verts_3d[faces[:, 1]]
        v2 = all_verts_3d[faces[:, 2]]
        v3 = all_verts_3d[faces[:, 3]]

        normal = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        vol = torch.sum((v3 - v0) * normal, dim=1)      # (F,)
        return torch.sum(vol ** 2)
    elif faces.shape[1] == 3:
        # Triangles are always planar; treat planarity energy as zero.
        return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)
    else:
        raise ValueError(f"Unsupported face size {faces.shape[1]} for planarity energy")


def compute_flatness_penalty_torch(all_verts_3d):
    """
    Direct z² flatness penalty (PyTorch, differentiable).
    Penalizes ANY out-of-plane displacement: E = Σ z_i²
    Unlike planarity, this also penalizes rigid z-translation and tilting.
    """
    return torch.sum(all_verts_3d[:, 2] ** 2)


def assemble_vertices_torch(interior_xy, boundary_xy, interior_indices, num_verts=25):
    """
    Assemble full vertex 2D positions from interior (predicted) and boundary (fixed).
    """
    batched = interior_xy.dim() == 2
    if not batched:
        interior_xy = interior_xy.unsqueeze(0)

    B = interior_xy.shape[0]
    device = interior_xy.device
    full = torch.zeros(B, num_verts, 2, device=device, dtype=interior_xy.dtype)

    boundary_mask = torch.ones(num_verts, dtype=torch.bool)
    boundary_mask[interior_indices] = False
    boundary_indices = torch.where(boundary_mask)[0]
    full[:, boundary_indices] = boundary_xy.unsqueeze(0).expand(B, -1, -1)

    interior_reshaped = interior_xy.view(B, 9, 2)
    full[:, interior_indices] = interior_reshaped

    if not batched:
        full = full.squeeze(0)

    return full

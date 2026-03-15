"""
Shared mesh definition and energy functions for the Planar Quad Toy project.
Provides both NumPy (for visualization) and PyTorch (for training) variants.
Supports 2D (planar) and 3D (with bending) modes.
"""

import os
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


def make_hemisphere_tri(n_lat=4, n_lon=8, radius=2.0):
    """
    Create a triangulated hemisphere (upper half of a sphere) mesh.

    - Sphere of given radius centered at the origin.
    - Latitude θ ∈ [0, π/2] (0 = north pole, π/2 = equator at z=0).
    - Longitude φ ∈ [0, 2π).
    - Faces: triangles made by splitting lat-long quads.

    Returns:
        vertices:        (num_verts, 3) float64 array
        faces:           (num_faces, 3) int32 array of vertex indices
        interior_mask:   boolean array, True for non-rim vertices
        bottom_lip_mask: boolean array, True for equator vertices (z≈0), to be anchored
    """
    vertices = []
    # Build latitude rings with n_lon samples each; use wraparound in faces
    # so the equator forms a closed circle with no gap.
    for j in range(n_lat + 1):
        theta = (np.pi / 2.0) * j / n_lat  # 0 .. π/2
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for i in range(n_lon):
            phi = 2.0 * np.pi * i / n_lon  # 0 .. 2π (wraps via modulo)
            x = radius * sin_t * np.cos(phi)
            y = radius * sin_t * np.sin(phi)
            z = radius * cos_t
            vertices.append([x, y, z])
    vertices = np.array(vertices, dtype=np.float64)

    def vid(j, i):
        return j * n_lon + i

    faces = []
    for j in range(n_lat):
        for i in range(n_lon):
            i_next = (i + 1) % n_lon
            v00 = vid(j, i)
            v01 = vid(j, i_next)
            v10 = vid(j + 1, i)
            v11 = vid(j + 1, i_next)
            # Two triangles per quad strip
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int32)

    # Bottom lip is the equator ring (j == n_lat, z ≈ 0)
    z_vals = vertices[:, 2]
    bottom_lip_mask = np.abs(z_vals) < 1e-6
    interior_mask = ~bottom_lip_mask

    return vertices, faces, interior_mask, bottom_lip_mask


_Y_UP_TO_Z_UP = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)


def load_obj(path, anchor_bottom_z=True, z_threshold_eps=0.02):
    """
    Load a mesh from an OBJ file, rotate from Y-up to Z-up, and detect floor.

    OBJ files use Y-up by convention (top of the mesh points in +Y, which
    faces the camera in Polyscope's default view). This function applies a
    fixed 90° rotation around X so that +Y → +Z, placing the mesh upright
    with Z as the vertical axis. Floor vertices (boundary verts at Z-min)
    are identified and frozen for training.

    Returns:
        vertices:        (num_verts, 3) float64 array  (Z-up, floor at z≈0)
        faces:           (num_faces, 3|4) int32 array of vertex indices
        interior_mask:   boolean array, True for non-floor vertices
        bottom_lip_mask: boolean array, True for floor vertices (frozen)
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")

    vertices = []
    raw_faces = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = []
                for p in parts[1:]:
                    v = int(p.split("/")[0])
                    idxs.append(v - 1 if v > 0 else v + len(vertices))
                raw_faces.append(idxs)

    vertices = np.array(vertices, dtype=np.float64)

    # Rotate from Y-up (OBJ convention) to Z-up (training convention)
    if len(vertices) > 0:
        vertices = (vertices @ _Y_UP_TO_Z_UP.T)

    if len(raw_faces) > 0 and all(len(f) == 4 for f in raw_faces):
        faces = np.array(raw_faces, dtype=np.int32)
    else:
        faces_tris = []
        for idxs in raw_faces:
            if len(idxs) == 3:
                faces_tris.append(idxs)
            elif len(idxs) >= 4:
                for k in range(1, len(idxs) - 1):
                    faces_tris.append([idxs[0], idxs[k], idxs[k + 1]])
        faces = np.array(faces_tris, dtype=np.int32)

    if anchor_bottom_z and len(vertices) > 0:
        boundary = _get_boundary_set(faces)
        z_min = vertices[:, 2].min()
        diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        eps = max(z_threshold_eps, 0.02 * diag)

        bottom_lip_mask = np.zeros(len(vertices), dtype=bool)
        for i in boundary:
            if vertices[i, 2] <= z_min + eps:
                bottom_lip_mask[i] = True

        # Translate so floor is at z = 0
        if bottom_lip_mask.any():
            vertices[:, 2] -= vertices[bottom_lip_mask, 2].mean()

        interior_mask = ~bottom_lip_mask
    else:
        bottom_lip_mask = np.zeros(len(vertices), dtype=bool)
        interior_mask = np.ones(len(vertices), dtype=bool)

    return vertices, faces, interior_mask, bottom_lip_mask


def _get_boundary_set(faces):
    """Return set of vertex indices that lie on a boundary edge (shared by only 1 face)."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    for face in faces:
        n = len(face)
        for k in range(n):
            e = tuple(sorted([int(face[k]), int(face[(k + 1) % n])]))
            edge_count[e] += 1
    boundary = set()
    for (a, b), c in edge_count.items():
        if c == 1:
            boundary.add(a)
            boundary.add(b)
    return boundary


def detect_floor_plane(vertices, faces, min_verts=6):
    """
    Detect the floor of an architectural mesh by finding which bounding-box
    face has the most boundary vertices sitting directly on it.

    Algorithm:
        1. Compute the set of boundary vertices (edges shared by only 1 face).
        2. Sweep eps from tight to loose (0.001 to 0.04 of bbox diagonal).
        3. At each eps, count boundary verts within eps of each of the 6
           axis-aligned bounding-box faces.
        4. Stop as soon as some face collects >= min_verts boundary vertices.
        5. Ties are broken by preferring the lower coordinate (ground = bottom).

    Returns:
        floor_axis:  int (0=X, 1=Y, 2=Z)
        floor_end:   'min' or 'max'
        floor_mask:  boolean array over all vertices, True for floor verts
    """
    boundary = _get_boundary_set(faces)
    bnd_mask = np.zeros(len(vertices), dtype=bool)
    for i in boundary:
        bnd_mask[i] = True

    diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    if diag < 1e-12:
        mask = np.zeros(len(vertices), dtype=bool)
        return 2, "min", mask

    eps_fracs = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
    best_axis, best_end, best_eps = 2, "min", eps_fracs[-1] * diag

    for frac in eps_fracs:
        eps = frac * diag
        top_count = 0
        top_axis = 0
        top_end = "min"

        for axis in range(3):
            vmin = vertices[:, axis].min()
            vmax = vertices[:, axis].max()

            n_min = np.sum(bnd_mask & (np.abs(vertices[:, axis] - vmin) < eps))
            n_max = np.sum(bnd_mask & (np.abs(vertices[:, axis] - vmax) < eps))

            # Prefer lower coordinate on ties (floor is at the bottom)
            if n_min > top_count or (n_min == top_count and n_min > 0):
                top_count = n_min
                top_axis = axis
                top_end = "min"
            if n_max > top_count:
                top_count = n_max
                top_axis = axis
                top_end = "max"

        if top_count >= min_verts:
            best_axis = top_axis
            best_end = top_end
            best_eps = eps
            break

    # Collect floor vertices: boundary verts on the winning face
    if best_end == "min":
        val = vertices[:, best_axis].min()
    else:
        val = vertices[:, best_axis].max()
    floor_mask = bnd_mask & (np.abs(vertices[:, best_axis] - val) < best_eps)

    return best_axis, best_end, floor_mask


def _rotation_floor_to_z(floor_axis, floor_end):
    """
    3x3 rotation matrix that maps the detected floor axis/end to Z-min,
    so the mesh stands upright with the floor on the Z=0 ground plane.
    """
    if floor_axis == 2:
        if floor_end == "min":
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    if floor_axis == 1:
        if floor_end == "min":
            # Y-min → Z-min:  z' = y
            return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        # Y-max → Z-min:  z' = -y
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    # floor_axis == 0
    if floor_end == "min":
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)


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
    """Spring energy: mean per-edge (||v_i - v_j|| - L_0)^2. Works for 2D or 3D."""
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    if len(lengths) == 0:
        return 0.0
    return np.mean((lengths - rest_lengths) ** 2)


def compute_quad_area_energy_np(vertices, faces, rest_areas):
    """Area preservation: mean per-face (A_face - A_0)^2. Works for tris or quads, 2D or 3D."""
    energy = 0.0
    n_faces = len(faces)
    if n_faces == 0:
        return 0.0
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
    return energy / float(n_faces)


def compute_planarity_energy_np(vertices, faces):
    """
    Per-quad planarity energy (NumPy).
    For each quad (v0,v1,v2,v3), planarity violation = volume of tetrahedron.
    E = Σ [ (v3-v0) · ((v1-v0) × (v2-v0)) ]^2
    Zero when all 4 vertices are coplanar.
    """
    energy = 0.0
    n_quads = 0
    for face in faces:
        if len(face) == 4:
            v0, v1, v2, v3 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            vol = np.dot(v3 - v0, normal)
            energy += vol ** 2
            n_quads += 1
        elif len(face) == 3:
            # Triangles are always planar; treat planarity energy as zero.
            continue
        else:
            raise ValueError(f"Unsupported face size {len(face)} for planarity energy")
    if n_quads == 0:
        return 0.0
    return energy / float(n_quads)


def compute_diag_planarity_metric_np(vertices, faces, eps=1e-8):
    """
    Diagonal-based planarity metric per face (NumPy).

    For each quad (v0, v1, v2, v3), we consider the two diagonals:
      d1: v0 → v2
      d2: v1 → v3

    We measure the shortest distance between the two infinite lines defined by
    these diagonals, and normalize by the average diagonal length:

        n      = d1 × d2
        dist   = | (v1 - v0) · n | / (||n|| + eps)
        avg_d  = 0.5 * (|d1| + |d2|)
        m_face = 100 * dist / (avg_d + eps)

    - For any planar quad, the diagonals intersect in a plane so dist = 0.
    - For a warped quad, the diagonals become skew in 3D and dist > 0.

    Returns:
        metrics: (num_faces,) array, percentage per face (0 for non-quad faces).
    """
    metrics = np.zeros(len(faces), dtype=np.float64)
    for fi, face in enumerate(faces):
        if len(face) != 4:
            continue
        v0, v1, v2, v3 = vertices[face]

        d1 = v2 - v0
        d2 = v3 - v1
        n = np.cross(d1, d2)
        norm_n = np.linalg.norm(n)
        if norm_n < eps:
            continue

        w = v1 - v0
        dist = abs(np.dot(w, n)) / (norm_n + eps)

        d1_len = np.linalg.norm(d1)
        d2_len = np.linalg.norm(d2)
        avg_len = 0.5 * (d1_len + d2_len)
        if avg_len < eps:
            continue

        metrics[fi] = 100.0 * dist / (avg_len + eps)

    # Clamp tiny numerical noise to exactly zero so perfectly planar meshes
    # (like the rest grid) show uniform zero.
    metrics[np.abs(metrics) < 1e-6] = 0.0
    return metrics


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
    if edges.numel() == 0:
        return torch.tensor(0.0, device=all_verts.device, dtype=all_verts.dtype)
    v0 = all_verts[edges[:, 0]]
    v1 = all_verts[edges[:, 1]]
    lengths = torch.norm(v1 - v0, dim=1)
    return torch.mean((lengths - rest_lengths) ** 2)


def compute_quad_area_energy_torch(all_verts, faces, rest_areas):
    """
    Area preservation energy in PyTorch. Works for tris or quads, (N, 2) or (N, 3).
    """
    dim = all_verts.shape[1]
    if faces.numel() == 0:
        return torch.tensor(0.0, device=all_verts.device, dtype=all_verts.dtype)

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

    return torch.mean((areas - rest_areas) ** 2)


def compute_planarity_energy_torch(all_verts_3d, faces):
    """
    Per-quad planarity energy (PyTorch, differentiable).
    For each quad (v0,v1,v2,v3): violation = (v3-v0) · ((v1-v0) × (v2-v0))
    This is the signed volume of the tetrahedron — zero when coplanar.
    E = Σ vol_i^2
    """
    if faces.shape[1] == 4:
        if faces.shape[0] == 0:
            return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)
        v0 = all_verts_3d[faces[:, 0]]  # (F, 3)
        v1 = all_verts_3d[faces[:, 1]]
        v2 = all_verts_3d[faces[:, 2]]
        v3 = all_verts_3d[faces[:, 3]]

        normal = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        vol = torch.sum((v3 - v0) * normal, dim=1)      # (F,)
        return torch.mean(vol ** 2)
    elif faces.shape[1] == 3:
        # Triangles are always planar; treat planarity energy as zero.
        return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)
    else:
        raise ValueError(f"Unsupported face size {faces.shape[1]} for planarity energy")


def compute_planarity_energy_per_sample_torch(verts_batch, faces):
    """
    Per-sample planarity energy for a batch of meshes.
    verts_batch: (B, N, 3) — batch of vertex positions.
    faces: (F, 4) or (F, 3) face indices (shared topology).
    Returns: (B,) tensor of per-sample mean planarity violations.
    """
    if faces.shape[1] == 3:
        return torch.zeros(verts_batch.shape[0], device=verts_batch.device,
                           dtype=verts_batch.dtype)
    if faces.shape[1] != 4:
        raise ValueError(f"Unsupported face size {faces.shape[1]} for planarity energy")
    if faces.shape[0] == 0:
        return torch.zeros(verts_batch.shape[0], device=verts_batch.device,
                           dtype=verts_batch.dtype)
    v0 = verts_batch[:, faces[:, 0], :]  # (B, F, 3)
    v1 = verts_batch[:, faces[:, 1], :]
    v2 = verts_batch[:, faces[:, 2], :]
    v3 = verts_batch[:, faces[:, 3], :]
    normal = torch.cross(v1 - v0, v2 - v0, dim=2)   # (B, F, 3)
    vol = torch.sum((v3 - v0) * normal, dim=2)       # (B, F)
    return torch.mean(vol ** 2, dim=1)                # (B,)


def compute_diag_planarity_energy_torch(all_verts_3d, faces, eps=1e-8):
    """
    Diagonal-based planarity energy in PyTorch.

    Mirrors compute_diag_planarity_metric_np (without the 100x factor),
    and returns Σ m_face² over quads.

    For each quad (v0, v1, v2, v3):
      - d1    = v2 - v0
      - d2    = v3 - v1
      - n     = d1 × d2
      - dist  = |(v1-v0) · n| / (||n|| + eps)
      - avg_d = 0.5 * (|d1| + |d2|)

      m_face = dist / (avg_d + eps)

    The energy is Σ m_face² over all quads. Triangles are ignored.
    """
    if faces.shape[1] != 4:
        return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)

    v0 = all_verts_3d[faces[:, 0]]
    v1 = all_verts_3d[faces[:, 1]]
    v2 = all_verts_3d[faces[:, 2]]
    v3 = all_verts_3d[faces[:, 3]]

    d1 = v2 - v0
    d2 = v3 - v1
    n = torch.cross(d1, d2, dim=1)   # (F, 3)
    norm_n = torch.norm(n, dim=1)    # (F,)

    w = v1 - v0
    vol = torch.sum(w * n, dim=1).abs()
    dist = vol / (norm_n + eps)

    d1_len = torch.norm(d1, dim=1)
    d2_len = torch.norm(d2, dim=1)
    avg_len = 0.5 * (d1_len + d2_len)

    metric = torch.zeros_like(dist)
    valid = avg_len >= eps
    metric[valid] = dist[valid] / (avg_len[valid] + eps)

    # Zero out tiny numerical noise so exactly planar quads contribute 0.
    metric = torch.where(metric.abs() < 1e-8,
                         torch.zeros_like(metric),
                         metric)

    return torch.mean(metric ** 2)


def compute_diag_planarity_energy_torch_old(all_verts_3d, faces, eps=1e-8):
    """
    Original diagonal-based planarity energy (older, line–line formula).

    For each quad (v0, v1, v2, v3), we consider diagonals:
      d1: v0 → v2 (direction u)
      d2: v1 → v3 (direction v)

    Using the standard closest-point formula between two lines, we compute the
    distance between the infinite lines defined by these diagonals, then
    normalize by the average diagonal length:

        m_face = dist(lines(d1, d2)) / (0.5 * (|d1| + |d2|) + eps)

    The energy is Σ m_face² over all quads. Triangles are ignored.
    """
    if faces.shape[1] != 4:
        return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)

    v0 = all_verts_3d[faces[:, 0]]
    v1 = all_verts_3d[faces[:, 1]]
    v2 = all_verts_3d[faces[:, 2]]
    v3 = all_verts_3d[faces[:, 3]]

    u = v2 - v0  # diag 1 direction
    v = v3 - v1  # diag 2 direction
    w0 = v0 - v1

    a = (u * u).sum(dim=1)
    b = (u * v).sum(dim=1)
    c = (v * v).sum(dim=1)
    d = (u * w0).sum(dim=1)
    e = (v * w0).sum(dim=1)
    denom = a * c - b * b

    stable = (denom.abs() >= eps) & (a >= eps) & (c >= eps)
    if not stable.any():
        return torch.tensor(0.0, device=all_verts_3d.device, dtype=all_verts_3d.dtype)

    s = torch.zeros_like(denom)
    t = torch.zeros_like(denom)
    s[stable] = (b[stable] * e[stable] - c[stable] * d[stable]) / (denom[stable] + eps)
    t[stable] = (a[stable] * e[stable] - b[stable] * d[stable]) / (denom[stable] + eps)

    closest_vec = w0 + s.unsqueeze(1) * u - t.unsqueeze(1) * v
    dist = torch.norm(closest_vec, dim=1)

    d1_len = torch.norm(u, dim=1)
    d2_len = torch.norm(v, dim=1)
    avg_len = 0.5 * (d1_len + d2_len)

    metric = torch.zeros_like(dist)
    valid = avg_len >= eps
    metric[valid] = dist[valid] / (avg_len[valid] + eps)

    return torch.mean(metric ** 2)


def compute_flatness_penalty_torch(all_verts_3d):
    """
    Direct z² flatness penalty (PyTorch, differentiable).
    Penalizes ANY out-of-plane displacement: E = Σ z_i²
    Unlike planarity, this also penalizes rigid z-translation and tilting.
    """
    return torch.sum(all_verts_3d[:, 2] ** 2)


def compute_edge_inequality_10_torch(all_verts, edges, rest_lengths, low=0.9, high=1.1, eps=1e-8):
    """
    Inequality penalty: penalize edge length outside [low*L0, high*L0] (default ±10%).
    Per edge: ratio = L/L0; penalty = ReLU(ratio - high)² + ReLU(low - ratio)².
    all_verts: (V, 3) or (B, V, 3). Returns scalar or (B,) per-sample mean over edges.
    """
    if edges.numel() == 0:
        out = torch.tensor(0.0, device=all_verts.device, dtype=all_verts.dtype)
        if all_verts.dim() == 3:
            out = out.unsqueeze(0).expand(all_verts.shape[0])
        return out
    batched = all_verts.dim() == 3
    if not batched:
        all_verts = all_verts.unsqueeze(0)
    B = all_verts.shape[0]
    v0 = all_verts[:, edges[:, 0], :]   # (B, E, 3)
    v1 = all_verts[:, edges[:, 1], :]
    lengths = torch.norm(v1 - v0, dim=2)   # (B, E)
    L0 = rest_lengths.to(lengths.device)
    if L0.dim() == 1:
        L0 = L0.unsqueeze(0).expand(B, -1)
    ratio = lengths / (L0 + eps)
    over = torch.relu(ratio - high)
    under = torch.relu(low - ratio)
    penalty = (over ** 2 + under ** 2).mean(dim=1)   # (B,)
    return penalty.squeeze(0) if not batched else penalty


def compute_mesh_width_height_torch(all_verts):
    """
    Mesh extent along x (width) and y (height). Differentiable.
    all_verts: (V, 3) or (B, V, 3). Returns width, height as scalars or (B,) tensors.
    """
    if all_verts.dim() == 2:
        w = all_verts[:, 0].max() - all_verts[:, 0].min()
        h = all_verts[:, 1].max() - all_verts[:, 1].min()
        return w, h
    else:
        # (B, V, 3)
        x = all_verts[:, :, 0]
        y = all_verts[:, :, 1]
        width = x.max(dim=1)[0] - x.min(dim=1)[0]
        height = y.max(dim=1)[0] - y.min(dim=1)[0]
        return width, height


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

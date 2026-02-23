"""
Latent space visualization: Load a trained neural subspace model and explore
the learned deformation space with d latent sliders.

Supports "anchored", "free", "free3d", and "stiffFree3d" modes.
Supports "grid" (flat 4x4) and "box" (open cube) meshes.
Includes a "roof mode" that adds walls and a floor beneath the mesh (grid only).

Usage:
    python3 latent_viz.py --model checkpoints/anchored/d6/model.pt
    python3 latent_viz.py --model checkpoints/free/d6/model.pt
    python3 latent_viz.py --model checkpoints/free3d/d6/wp1.0/model.pt
    python3 latent_viz.py --model checkpoints/stiffFree3d/d6/kxy1.0_kz0.1/model.pt
    python3 latent_viz.py --model checkpoints/stiffFree3d/box/d6/kxy1.0_kz0.5/model.pt
"""

import argparse
import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

from mesh import (
    make_quad_grid,
    make_open_box,
    make_semicircle_tri,
    get_all_edges,
    compute_edge_energy_np,
    compute_quad_area_energy_np,
    compute_planarity_energy_np,
    compute_flatness_penalty_np,
    compute_max_z_deviation_np,
)
from train import SubspaceDecoder


MATERIALS = ["clay", "wax", "candy", "flat", "mud", "ceramic", "jade"]


def get_boundary_edges(nx=4, ny=4):
    """Get ordered perimeter edges of an nx×ny quad grid (vertex grid is (nx+1)×(ny+1))."""
    nvx = nx + 1
    boundary_edges = []
    for i in range(nx):
        boundary_edges.append((i, i + 1))
    for j in range(ny):
        boundary_edges.append((j * nvx + nx, (j + 1) * nvx + nx))
    for i in range(nx, 0, -1):
        boundary_edges.append((ny * nvx + i, ny * nvx + i - 1))
    for j in range(ny, 0, -1):
        boundary_edges.append((j * nvx, (j - 1) * nvx))
    return boundary_edges


def build_wall_geometry(roof_verts, boundary_edges, floor_z, is_3d):
    """
    Build wall + floor mesh geometry.
    Returns (wall_verts, wall_faces) where:
      - wall_verts[0:N] = roof vertex positions
      - wall_verts[N:2N] = floor copies (same xy, z=floor_z)
      - wall_faces = quads connecting roof boundary edges to floor + 1 floor quad
    """
    N = len(roof_verts)
    floor_verts = roof_verts.copy()
    if is_3d:
        floor_verts[:, 2] = floor_z
    else:
        floor_verts[:, 1] = floor_z

    wall_verts = np.vstack([roof_verts, floor_verts])

    wall_faces = []
    for a, b in boundary_edges:
        wall_faces.append([a, b, b + N, a + N])

    corners_top = [0, 4, 24, 20]
    floor_quad = [c + N for c in corners_top]
    wall_faces.append(floor_quad)

    return wall_verts, np.array(wall_faces, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--slider_range", type=float, default=3.0,
                        help="Range for latent sliders (±)")
    args = parser.parse_args()

    # --- Load model ---
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    num_layers = checkpoint["num_layers"]
    output_dim = checkpoint.get("output_dim", 18)
    mode = checkpoint.get("mode", "anchored")
    mesh_type = checkpoint.get("mesh", "grid")

    model = SubspaceDecoder(
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model: mode={mode}, mesh={mesh_type}, latent_dim={latent_dim}, "
          f"output_dim={output_dim}, hidden={hidden_dim}, layers={num_layers}")
    if mode == "free3d":
        wp = checkpoint.get("w_planarity", "?")
        print(f"  w_planarity={wp}")
    if mode == "stiffFree3d":
        k_xy = checkpoint.get("k_xy", "?")
        k_z = checkpoint.get("k_z", "?")
        print(f"  k_xy={k_xy}, k_z={k_z}")

    # --- Mesh setup ---
    bottom_lip_mask = None
    if mesh_type == "box":
        vertices_rest, faces, interior_mask, bottom_lip_mask = make_open_box(2)
    elif mesh_type == "semiTri":
        vertices_rest, faces, interior_mask = make_semicircle_tri()
    else:
        vertices_rest, faces, interior_mask = make_quad_grid(4, 4)

    num_verts = len(vertices_rest)
    edges = get_all_edges(faces)

    rest_lengths = np.linalg.norm(
        vertices_rest[edges[:, 1]] - vertices_rest[edges[:, 0]], axis=1
    )
    # Compute actual rest areas per face
    rest_areas_list = []
    for face in faces:
        v0, v1, v2, v3 = vertices_rest[face]
        a1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        a2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
        rest_areas_list.append(a1 + a2)
    rest_areas = np.array(rest_areas_list, dtype=np.float64)

    interior_indices = np.where(interior_mask)[0]
    vertices = vertices_rest.copy()

    q_seed_np = np.array(checkpoint["q_seed"], dtype=np.float32)

    # Displacement mask for box mesh (freeze z of bottom lip)
    disp_mask_np = None
    if bottom_lip_mask is not None:
        disp_mask_np = np.ones(num_verts * 3, dtype=np.float32)
        for idx in np.where(bottom_lip_mask)[0]:
            disp_mask_np[idx * 3 + 2] = 0.0

    is_3d = mode in ("free3d", "stiffFree3d")

    # Latent vector state
    z_values = np.zeros(latent_dim, dtype=np.float32)
    SLIDER_RANGE = args.slider_range

    # Build edge segment list from quad faces for the separate curve network
    edge_segments = []
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            a, b = int(face[k]), int(face[(k + 1) % n])
            key = (min(a, b), max(a, b))
            if key not in edge_set:
                edge_set.add(key)
                edge_segments.append([a, b])
    edge_segments = np.array(edge_segments, dtype=np.int32)

    # Boundary edges for roof mode walls (grid only)
    boundary_edges = get_boundary_edges(4, 4) if mesh_type == "grid" else []
    roof_supported = mesh_type == "grid"

    # Appearance state
    viz_state = {
        "transparency": 1.0,
        "material_idx": 0,
        "mesh_color": (0.2, 0.5, 0.9),
        "edge_color": (0.1, 0.1, 0.1),
        "bg_color": (1.0, 1.0, 1.0),
        "show_vertices": True,
        "show_edges": True,
        "edge_radius": 0.002,
        "first_frame": True,
        "roof_mode": False,
        "wall_color": (0.85, 0.75, 0.6),
        "wall_height": 2.0,
        "show_floor": False,
        "floor_color": (0.3, 0.7, 0.3),
    }

    # --- Decode function ---
    @torch.no_grad()
    def decode(z_np):
        z_t = torch.from_numpy(z_np).float().unsqueeze(0)
        mlp_out = model(z_t).squeeze(0).numpy()
        if disp_mask_np is not None:
            mlp_out = mlp_out * disp_mask_np
        pred = q_seed_np + mlp_out

        verts = vertices_rest.copy()
        if mode == "anchored":
            n_interior = len(interior_indices)
            interior_xy = pred.reshape(n_interior, 2)
            for k, vid in enumerate(interior_indices):
                verts[vid, 0] = interior_xy[k, 0]
                verts[vid, 1] = interior_xy[k, 1]
        elif mode == "free":
            all_xy = pred.reshape(-1, 2)
            verts[:, 0] = all_xy[:, 0]
            verts[:, 1] = all_xy[:, 1]
        elif mode in ("free3d", "stiffFree3d"):
            verts[:, :] = pred.reshape(-1, 3)
        return verts

    # --- Mode labels ---
    mode_labels = {
        "anchored": "Anchored Mesh",
        "free": "Free Mesh (2D)",
        "free3d": "Free-3D Mesh",
        "stiffFree3d": "Stiff-Free3D",
    }
    mode_label = mode_labels.get(mode, mode)
    if mesh_type != "grid":
        mode_label = f"[{mesh_type}] {mode_label}"
    if mode == "stiffFree3d":
        mode_label += f" (kxy={checkpoint.get('k_xy','?')}, kz={checkpoint.get('k_z','?')})"

    # --- Polyscope setup ---
    ps.init()
    ps.set_up_dir("z_up" if (is_3d or mesh_type == "box") else "y_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color(viz_state["bg_color"])

    ps.set_build_gui(False)
    ps.set_open_imgui_window_for_user_callback(False)

    # Face mesh (transparency applies here only)
    mesh_ps = ps.register_surface_mesh("quad_grid", vertices, faces, edge_width=0.0)
    mesh_ps.set_color(viz_state["mesh_color"])

    # Separate edge curve network (always opaque)
    edges_ps = ps.register_curve_network("edges", vertices, edge_segments,
                                          radius=viz_state["edge_radius"])
    edges_ps.set_color(viz_state["edge_color"])

    # Vertex point cloud
    cloud = ps.register_point_cloud("vertices", vertices, radius=0.04)
    colors = np.zeros((num_verts, 3))
    if mode == "anchored":
        colors[~interior_mask] = [0.8, 0.2, 0.2]
        colors[interior_mask] = [0.2, 0.8, 0.2]
    elif bottom_lip_mask is not None:
        colors[:] = [0.2, 0.8, 0.2]
        colors[bottom_lip_mask] = [0.8, 0.2, 0.2]
    else:
        colors[:] = [0.2, 0.8, 0.2]
    cloud.add_color_quantity("type", colors, enabled=True)

    # Freeze scene extents so walls/floor don't bloat the bounding box
    # (which would shrink auto-scaled vertex/edge sizes).
    ps.set_automatically_compute_scene_extents(False)

    # Roof mode: walls mesh (grid only — box mesh is already a box)
    walls_ps = None
    floor_ps = None

    if roof_supported:
        floor_z = -viz_state["wall_height"]
        wall_verts, wall_faces = build_wall_geometry(vertices, boundary_edges, floor_z, is_3d)
        walls_ps = ps.register_surface_mesh("walls", wall_verts, wall_faces, edge_width=1.0)
        walls_ps.set_color(viz_state["wall_color"])
        walls_ps.set_material("flat")
        walls_ps.set_enabled(False)

        # Floor plane for roof mode (below the walls)
        floor_size = 7.5
        cx, cy = 2.0, 2.0
        floor_eps = 0.01
        if is_3d:
            floor_plane_verts = np.array([
                [cx - floor_size, cy - floor_size, floor_z - floor_eps],
                [cx + floor_size, cy - floor_size, floor_z - floor_eps],
                [cx + floor_size, cy + floor_size, floor_z - floor_eps],
                [cx - floor_size, cy + floor_size, floor_z - floor_eps],
            ])
        else:
            floor_plane_verts = np.array([
                [cx - floor_size, floor_z - floor_eps, -floor_size],
                [cx + floor_size, floor_z - floor_eps, -floor_size],
                [cx + floor_size, floor_z - floor_eps,  floor_size],
                [cx - floor_size, floor_z - floor_eps,  floor_size],
            ])
        floor_plane_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
        floor_ps = ps.register_surface_mesh("floor_plane", floor_plane_verts, floor_plane_faces,
                                             edge_width=0.0)
        floor_ps.set_color(viz_state["floor_color"])
        floor_ps.set_material("flat")
        floor_ps.set_enabled(False)

    elif mesh_type == "box":
        # Floor plane at z=0 (the anchored bottom-lip plane)
        n_box = 2
        floor_size = 7.5
        cx, cy = n_box / 2.0, n_box / 2.0
        floor_eps = 0.01
        floor_plane_verts = np.array([
            [cx - floor_size, cy - floor_size, -floor_eps],
            [cx + floor_size, cy - floor_size, -floor_eps],
            [cx + floor_size, cy + floor_size, -floor_eps],
            [cx - floor_size, cy + floor_size, -floor_eps],
        ])
        floor_plane_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
        floor_ps = ps.register_surface_mesh("floor_plane", floor_plane_verts, floor_plane_faces,
                                             edge_width=0.0)
        floor_ps.set_color(viz_state["floor_color"])
        floor_ps.set_material("flat")
        floor_ps.set_enabled(False)

    def update_floor_plane():
        """Update floor plane height to match wall bottom (grid roof mode only)."""
        if not roof_supported:
            return
        fz = -viz_state["wall_height"]
        eps = 0.01
        fv = floor_plane_verts.copy()
        if is_3d:
            fv[:, 2] = fz - eps
        else:
            fv[:, 1] = fz - eps
        floor_ps.update_vertex_positions(fv)

    def update_walls():
        """Rebuild wall geometry from current roof vertices."""
        if not roof_supported:
            return
        floor_z = -viz_state["wall_height"]
        wv, _ = build_wall_geometry(vertices, boundary_edges, floor_z, is_3d)
        walls_ps.update_vertex_positions(wv)
        if viz_state["show_floor"]:
            update_floor_plane()

    def callback():
        nonlocal z_values, vertices

        first = viz_state["first_frame"]

        # =============================================================
        # TOP-LEFT: Latent sliders
        # =============================================================
        if first:
            psim.SetNextWindowPos((10.0, 10.0))
            psim.SetNextWindowSize((320.0, 0.0))
        _, _ = psim.Begin(f"Latent Space (d={latent_dim})", True)

        psim.Text(f"Mode: {mode_label}")
        psim.Separator()

        changed = False
        for i in range(latent_dim):
            c, val = psim.SliderFloat(
                f"z[{i}]##{i}",
                float(z_values[i]),
                v_min=-SLIDER_RANGE,
                v_max=SLIDER_RANGE,
            )
            if c:
                z_values[i] = val
                changed = True

        if psim.Button("Reset to origin"):
            z_values[:] = 0.0
            changed = True

        psim.SameLine()
        if psim.Button("Random sample"):
            z_values[:] = np.random.randn(latent_dim).astype(np.float32)
            changed = True

        psim.End()

        # =============================================================
        # TOP-RIGHT: Energy readouts
        # =============================================================
        energy_edge = compute_edge_energy_np(vertices, edges, rest_lengths)
        energy_area = compute_quad_area_energy_np(vertices, faces, rest_areas)

        if first:
            psim.SetNextWindowPos((780.0, 10.0))
            psim.SetNextWindowSize((280.0, 0.0))
        _, _ = psim.Begin("Energy", True)
        psim.Text(f"Edge energy:  {energy_edge:.6f}")
        psim.Text(f"Area energy:  {energy_area:.6f}")
        psim.Text(f"Total energy: {energy_edge + energy_area:.6f}")
        if mode in ("free3d", "stiffFree3d"):
            e_planar = compute_planarity_energy_np(vertices, faces)
            max_z = compute_max_z_deviation_np(vertices)
            z_sum_sq = compute_flatness_penalty_np(vertices)
            psim.Separator()
            psim.Text(f"Planarity:  {e_planar:.6f}")
            psim.Text(f"Max |z|:    {max_z:.6f}")
            psim.Text(f"Sum z^2:    {z_sum_sq:.6f}")
        psim.End()

        # =============================================================
        # BOTTOM-LEFT: Appearance controls
        # =============================================================
        if first:
            psim.SetNextWindowPos((10.0, 400.0))
            psim.SetNextWindowSize((320.0, 0.0))
        _, _ = psim.Begin("Appearance", True)

        # Material dropdown
        cur_mat = MATERIALS[viz_state["material_idx"]]
        if psim.BeginCombo("Material", cur_mat):
            for idx, mat_name in enumerate(MATERIALS):
                _, selected = psim.Selectable(mat_name, viz_state["material_idx"] == idx)
                if selected:
                    viz_state["material_idx"] = idx
                    mesh_ps.set_material(mat_name)
                    mesh_ps.set_color(viz_state["mesh_color"])
            psim.EndCombo()

        # Face color
        c_col, new_col = psim.ColorEdit3("Face color", viz_state["mesh_color"])
        if c_col:
            viz_state["mesh_color"] = new_col
            mesh_ps.set_color(new_col)
            mesh_ps.set_material(MATERIALS[viz_state["material_idx"]])

        # Face transparency (edges stay opaque)
        c_t, new_t = psim.SliderFloat("Face transparency", viz_state["transparency"],
                                       v_min=0.0, v_max=1.0)
        if c_t:
            viz_state["transparency"] = new_t
            mesh_ps.set_transparency(new_t)

        psim.Separator()

        # Edge color
        c_ec, new_ec = psim.ColorEdit3("Edge color", viz_state["edge_color"])
        if c_ec:
            viz_state["edge_color"] = new_ec
            edges_ps.set_color(new_ec)

        # Edge radius
        c_er, new_er = psim.SliderFloat("Edge radius", viz_state["edge_radius"],
                                         v_min=0.001, v_max=0.01)
        if c_er:
            viz_state["edge_radius"] = new_er
            edges_ps.set_radius(new_er)

        # Toggle edges
        c_se, new_se = psim.Checkbox("Show edges", viz_state["show_edges"])
        if c_se:
            viz_state["show_edges"] = new_se
            edges_ps.set_enabled(new_se)

        # Toggle vertices
        c_sv, new_sv = psim.Checkbox("Show vertices", viz_state["show_vertices"])
        if c_sv:
            viz_state["show_vertices"] = new_sv
            cloud.set_enabled(new_sv)

        psim.Separator()

        if roof_supported:
            # Roof mode (grid mesh only)
            c_rm, new_rm = psim.Checkbox("Roof mode", viz_state["roof_mode"])
            if c_rm:
                viz_state["roof_mode"] = new_rm
                walls_ps.set_enabled(new_rm)
                floor_ps.set_enabled(new_rm and viz_state["show_floor"])
                if new_rm:
                    update_walls()

            if viz_state["roof_mode"]:
                c_wc, new_wc = psim.ColorEdit3("Wall color", viz_state["wall_color"])
                if c_wc:
                    viz_state["wall_color"] = new_wc
                    walls_ps.set_color(new_wc)

                c_wh, new_wh = psim.SliderFloat("Wall height", viz_state["wall_height"],
                                                  v_min=0.5, v_max=5.0)
                if c_wh:
                    viz_state["wall_height"] = new_wh
                    update_walls()

                c_sf, new_sf = psim.Checkbox("Show floor", viz_state["show_floor"])
                if c_sf:
                    viz_state["show_floor"] = new_sf
                    floor_ps.set_enabled(new_sf)
                    if new_sf:
                        update_floor_plane()

                if viz_state["show_floor"]:
                    c_fc, new_fc = psim.ColorEdit3("Floor color", viz_state["floor_color"])
                    if c_fc:
                        viz_state["floor_color"] = new_fc
                        floor_ps.set_color(new_fc)

            psim.Separator()

        if mesh_type == "box" and floor_ps is not None:
            c_sf, new_sf = psim.Checkbox("Show floor", viz_state["show_floor"])
            if c_sf:
                viz_state["show_floor"] = new_sf
                floor_ps.set_enabled(new_sf)

            if viz_state["show_floor"]:
                c_fc, new_fc = psim.ColorEdit3("Floor color", viz_state["floor_color"])
                if c_fc:
                    viz_state["floor_color"] = new_fc
                    floor_ps.set_color(new_fc)

            psim.Separator()

        # Background color
        c_bg, new_bg = psim.ColorEdit3("Background", viz_state["bg_color"])
        if c_bg:
            viz_state["bg_color"] = new_bg
            ps.set_background_color(new_bg)

        psim.End()

        # =============================================================
        # Update mesh on latent change
        # =============================================================
        if changed:
            vertices = decode(z_values)
            mesh_ps.update_vertex_positions(vertices)
            edges_ps.update_node_positions(vertices)
            cloud.update_point_positions(vertices)
            if roof_supported and viz_state["roof_mode"]:
                update_walls()

        viz_state["first_frame"] = False

    ps.set_user_callback(callback)

    # Initial decode at z=0
    vertices = decode(z_values)
    mesh_ps.update_vertex_positions(vertices)
    edges_ps.update_node_positions(vertices)
    cloud.update_point_positions(vertices)

    ps.show()


if __name__ == "__main__":
    main()

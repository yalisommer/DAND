"""
Training script: Learn a neural subspace for quad meshes.

Meshes (--mesh):
  - "grid":  4×4 planar quad grid (25 verts, 16 faces)
  - "box":   open box / cube missing bottom (25 verts @ n=2, 20 faces)
             bottom-lip vertices have z frozen at 0

Modes (--mode):
  - "anchored":      boundary fixed, 9 interior verts × 2D (18 DOF) [grid only]
  - "free":          all verts × 2D (50 DOF) [grid only]
  - "free3d":        all verts × 3D with volumetric planarity penalty
  - "stiffFree3d":   all verts × 3D with anisotropic stiffness
  - "structural3d":  NEW — drops all rest-state stiffness springs.
                     Energy = planarity + anti-collapse + Dirichlet (weak) +
                              anti-triviality.
                     Anti-triviality term: w_trivial * rho^2 * mean(
                         ||z_i||^2 / (||f(z_i) - q_seed||^2 + eps)
                     )
                     This penalises the network for mapping large-z inputs to
                     configurations close to the rest state — forcing the latent
                     space to "use" its geometry as z moves away from the origin.

Implements the approach from "Data-Free Learning of Reduced-Order Kinematics":
  - MLP decoder: z ∈ R^d  →  vertex displacements from rest state
  - Residual seeded training: f(z) = q_seed + ρ·MLP(z), ρ: 0→1
  - Loss = E_pot + λ·metric_reg + w_anchor·anchor_loss
"""

import argparse
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

from mesh import (
    make_quad_grid,
    make_open_box,
    make_semicircle_tri,
    make_hemisphere_tri,
    load_obj,
    get_all_edges,
    compute_edge_energy_torch,
    compute_quad_area_energy_torch,
    compute_planarity_energy_torch,
    compute_diag_planarity_energy_torch,
    compute_diag_planarity_energy_torch_old,
    compute_edge_inequality_10_torch,
    compute_mesh_width_height_torch,
    assemble_vertices_torch,
    get_face_adjacency,
    compute_inter_face_dirichlet_torch,
    compute_area_anticollapse_torch,
)


# =============================================================================
# MLP Decoder
# =============================================================================

class SubspaceDecoder(nn.Module):
    def __init__(self, latent_dim=6, output_dim=18, hidden_dim=64, num_layers=5):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# =============================================================================
# Seeded subspace map
# =============================================================================

def seeded_forward(model, z, q_seed, rho, disp_mask=None):
    mlp_out = model(z)
    if disp_mask is not None:
        mlp_out = mlp_out * disp_mask.unsqueeze(0)
    q_seed_expanded = q_seed.unsqueeze(0).expand_as(mlp_out)
    return q_seed_expanded + rho * mlp_out


# =============================================================================
# Loss functions — existing modes (unchanged)
# =============================================================================

def potential_energy_anchored(pred_batch, boundary_xy, interior_indices,
                              edges, rest_lengths, faces, rest_areas,
                              w_edge=1.0, w_area=0.5):
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = assemble_vertices_torch(pred_batch[i], boundary_xy, interior_indices)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        energies.append(w_edge * e_edge + w_area * e_area)
    return torch.stack(energies)


def potential_energy_free(pred_batch, edges, rest_lengths, faces, rest_areas,
                          w_edge=1.0, w_area=0.5):
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 2)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        energies.append(w_edge * e_edge + w_area * e_area)
    return torch.stack(energies)


def _optional_penalty_terms(all_verts, edges, rest_lengths, faces,
                            w_inverse_diag=0.0, w_edge_length=0.0,
                            w_edge_inequality_10=0.0, w_width=0.0, w_height=0.0):
    extra = 0.0
    if w_inverse_diag > 0.0:
        e_diag = compute_diag_planarity_energy_torch(all_verts, faces)
        extra = extra - w_inverse_diag * e_diag
    if w_edge_length > 0.0:
        extra = extra + w_edge_length * compute_edge_energy_torch(all_verts, edges, rest_lengths)
    if w_edge_inequality_10 > 0.0:
        extra = extra + w_edge_inequality_10 * compute_edge_inequality_10_torch(
            all_verts, edges, rest_lengths)
    if w_width > 0.0 or w_height > 0.0:
        width, height = compute_mesh_width_height_torch(all_verts)
        extra = extra - w_width * width - w_height * height
    return extra


def potential_energy_free3d(pred_batch, edges, rest_lengths, faces, rest_areas,
                            w_edge=1.0, w_area=0.5, w_planarity=10.0,
                            w_diag_planarity=0.0, w_inverse_diag=0.0,
                            w_edge_length=0.0, w_edge_inequality_10=0.0,
                            w_width=0.0, w_height=0.0):
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 3)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e_planar = compute_planarity_energy_torch(all_verts, faces)
        e_diag = compute_diag_planarity_energy_torch(all_verts, faces) if w_diag_planarity > 0.0 else 0.0
        extra = _optional_penalty_terms(
            all_verts, edges, rest_lengths, faces,
            w_inverse_diag=w_inverse_diag, w_edge_length=w_edge_length,
            w_edge_inequality_10=w_edge_inequality_10, w_width=w_width, w_height=w_height)
        energies.append(w_edge * e_edge + w_area * e_area + w_planarity * e_planar
                        + w_diag_planarity * e_diag + extra)
    return torch.stack(energies)


def potential_energy_stiff3d(pred_batch, q_seed, edges, rest_lengths, faces, rest_areas,
                             w_edge=1.0, w_area=0.5, k_xy=1.0, k_z=1.0,
                             w_diag_planarity=0.0, w_inverse_diag=0.0,
                             w_edge_length=0.0, w_edge_inequality_10=0.0,
                             w_width=0.0, w_height=0.0):
    B = pred_batch.shape[0]
    rest_verts = q_seed.view(-1, 3)
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 3)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e_diag = compute_diag_planarity_energy_torch(all_verts, faces) if w_diag_planarity > 0.0 else 0.0
        disp = all_verts - rest_verts
        e_xy = torch.sum(disp[:, :2] ** 2)
        e_z = torch.sum(disp[:, 2] ** 2)
        extra = _optional_penalty_terms(
            all_verts, edges, rest_lengths, faces,
            w_inverse_diag=w_inverse_diag, w_edge_length=w_edge_length,
            w_edge_inequality_10=w_edge_inequality_10, w_width=w_width, w_height=w_height)
        energies.append(w_edge * e_edge + w_area * e_area + k_xy * e_xy + k_z * e_z
                        + w_diag_planarity * e_diag + extra)
    return torch.stack(energies)


def potential_energy_old_diag(pred_batch, edges, rest_lengths, faces, rest_areas,
                              w_edge=1.0, w_area=0.5, w_old_diag=1.0,
                              w_inverse_diag=0.0, w_edge_length=0.0,
                              w_edge_inequality_10=0.0, w_width=0.0, w_height=0.0):
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 3)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e_diag_old = compute_diag_planarity_energy_torch_old(all_verts, faces)
        extra = _optional_penalty_terms(
            all_verts, edges, rest_lengths, faces,
            w_inverse_diag=w_inverse_diag, w_edge_length=w_edge_length,
            w_edge_inequality_10=w_edge_inequality_10, w_width=w_width, w_height=w_height)
        energies.append(w_edge * e_edge + w_area * e_area + w_old_diag * e_diag_old + extra)
    return torch.stack(energies)


# =============================================================================
# structural3d energy (NEW)
# =============================================================================

def potential_energy_structural3d(
    pred_batch,
    rest_verts,
    edges, rest_lengths, faces, adj_pairs, rest_areas,
    w_planarity=50.0,
    w_anticollapse=10.0,
    anticollapse_eps=0.01,
    w_dirichlet=0.05,
    w_edge=0.0,
    w_area=0.0,
):
    """
    Structural energy for the 'structural3d' mode.

    Terms
    -----
    1. Planarity     — keep quads flat (dominant term)
    2. Anti-collapse — one-sided unsigned area penalty, fires only near collapse
    3. Dirichlet     — weak inter-face smoothness (default 0.05)
    4. Optional weak rest-state anchors (w_edge / w_area) for scale stability

    The anti-triviality term lives in the training loop (not here) because it
    depends on both z and f(z).
    """
    is_quad   = (faces.shape[1] == 4)
    v0_rest   = rest_verts  # (N, 3)

    # Batch-vectorised: reshape pred_batch to (B, N, 3)
    B      = pred_batch.shape[0]
    N      = rest_verts.shape[0]
    verts_batch = pred_batch.view(B, N, 3)   # (B, N, 3)

    # --- Planarity (vectorised over batch) ---
    if is_quad and w_planarity > 0.0:
        from mesh import compute_planarity_energy_per_sample_torch
        e_planar = compute_planarity_energy_per_sample_torch(verts_batch, faces)  # (B,)
    else:
        e_planar = torch.zeros(B, device=pred_batch.device, dtype=pred_batch.dtype)

    # --- Anti-collapse and Dirichlet: loop over batch (each call is vectorised over faces) ---
    e_anticollapse_list = []
    e_dirichlet_list    = []
    e_edge_list         = []
    e_area_list         = []

    for i in range(B):
        all_verts = verts_batch[i]   # (N, 3)

        if w_anticollapse > 0.0:
            e_anticollapse_list.append(compute_area_anticollapse_torch(
                all_verts, faces, anticollapse_eps))
        if w_dirichlet > 0.0:
            e_dirichlet_list.append(compute_inter_face_dirichlet_torch(
                all_verts, faces, v0_rest, adj_pairs))
        if w_edge > 0.0:
            e_edge_list.append(compute_edge_energy_torch(all_verts, edges, rest_lengths))
        if w_area > 0.0:
            e_area_list.append(compute_quad_area_energy_torch(all_verts, faces, rest_areas))

    def _stack_or_zeros(lst):
        if lst:
            return torch.stack(lst)
        return torch.zeros(B, device=pred_batch.device, dtype=pred_batch.dtype)

    e_anticollapse = _stack_or_zeros(e_anticollapse_list)
    e_dirichlet    = _stack_or_zeros(e_dirichlet_list)
    e_edge         = _stack_or_zeros(e_edge_list)
    e_area         = _stack_or_zeros(e_area_list)

    return (w_planarity    * e_planar
            + w_anticollapse * e_anticollapse
            + w_dirichlet    * e_dirichlet
            + w_edge         * e_edge
            + w_area         * e_area)


# =============================================================================
# Anti-triviality loss (structural3d only)
# =============================================================================

def anti_triviality_loss(f_z, q_seed, z, rho, eps=1e-4):
    """
    Penalises the network for mapping large-z inputs to configurations that
    are still close to the rest state -- after removing translation and scale
    as free escape routes.

    Translation removal: both f(z) and q_seed are reshaped to (B,N,3) and
    per-sample centroids are subtracted. A pure rigid translation contributes
    zero to the denominator, so the network cannot satisfy the term cheaply
    by sliding the mesh.

    Scale removal: the displacement norm is divided by the RMS of the
    centroid-removed rest vertices. Uniform expansion/contraction scales
    the denominator proportionally, so growing/shrinking the mesh also does
    not satisfy the term cheaply.

    Parameters
    ----------
    f_z    : (B, output_dim) predicted configurations  (output_dim = N*3)
    q_seed : (output_dim,) rest-state configuration
    z      : (B, latent_dim) latent vectors
    rho    : float, current rho value (0->1)
    eps    : denominator stabiliser
    """
    if rho < 1e-6:
        return torch.tensor(0.0, device=f_z.device, dtype=f_z.dtype)

    B = f_z.shape[0]
    N = f_z.shape[1] // 3

    # Reshape to vertex arrays
    f_verts = f_z.view(B, N, 3)                               # (B, N, 3)
    q_verts = q_seed.view(N, 3)                               # (N, 3)

    # Remove translation: subtract per-sample centroid
    f_centered = f_verts - f_verts.mean(dim=1, keepdim=True)  # (B, N, 3)
    q_centered = q_verts - q_verts.mean(dim=0, keepdim=True)  # (N, 3)

    # Scale reference: RMS of rest-state vertex positions (centroid-removed)
    scale_sq = torch.clamp((q_centered ** 2).mean(), min=eps)

    # Shape displacement (translation- and scale-invariant)
    disp            = f_centered - q_centered.unsqueeze(0)    # (B, N, 3)
    disp_norm_sq    = (disp ** 2).sum(dim=(1, 2)) / N         # (B,)
    disp_normalised = disp_norm_sq / scale_sq                 # dimensionless (B,)

    # Latent norm (normalised by latent dimensionality)
    z_norm_sq = (z ** 2).sum(dim=1) / z.shape[1]              # (B,)

    penalty = z_norm_sq / (disp_normalised + eps)              # (B,)

    # Gate by rho^2: inactive during warmup, fully active by end of training
    return (rho ** 2) * penalty.mean()


# =============================================================================
# Axis-aligned diversity loss (option 3)
# =============================================================================

def axis_diversity_loss(model, q_seed, rho, latent_dim, alpha=1.0, eps=1e-6, disp_mask=None):
    """
    Penalises cosine similarity between the shape displacements produced by
    probing each latent axis independently.

    For each basis vector e_k (k = 0..latent_dim-1) we evaluate:
        f_k = seeded_forward(model, alpha * e_k, q_seed, rho)
        d_k = centroid- and scale-removed displacement of f_k from q_seed

    Then for every pair (i, j), i != j:
        penalty += cos_sim(d_i, d_j)^2

    This directly targets the failure mode where different sliders produce
    the same deformation type -- it forces each axis to produce a distinct
    direction of shape change.

    Parameters
    ----------
    model      : SubspaceDecoder
    q_seed     : (output_dim,) rest-state configuration
    rho        : float, current schedule value (0->1)
    latent_dim : int
    alpha      : probe magnitude along each axis. Should be ~O(sigma) so the
                 probe reaches a region of the latent space where the network
                 produces meaningful deformations (not just near-zero warmup).
    eps        : normalisation stabiliser
    disp_mask  : optional (output_dim,) freeze mask, same as in seeded_forward

    Returns
    -------
    scalar tensor
    """
    if rho < 1e-6:
        return torch.tensor(0.0, device=q_seed.device, dtype=q_seed.dtype)

    N = q_seed.shape[0] // 3

    # Rest-state centroid and scale (precomputed once per call, cheap)
    q_verts    = q_seed.view(N, 3)
    q_centroid = q_verts.mean(dim=0, keepdim=True)           # (1, 3)
    q_centered = q_verts - q_centroid                        # (N, 3)
    scale_sq   = torch.clamp((q_centered ** 2).mean(), min=eps)

    # Probe each axis: (latent_dim, output_dim)
    device = q_seed.device
    dtype  = q_seed.dtype
    disps  = []

    for k in range(latent_dim):
        z_k = torch.zeros(1, latent_dim, device=device, dtype=dtype)
        z_k[0, k] = alpha

        with torch.no_grad() if False else torch.enable_grad():
            f_k = seeded_forward(model, z_k, q_seed, rho, disp_mask=disp_mask)  # (1, output_dim)

        f_verts    = f_k.view(N, 3)                          # (N, 3)
        f_centered = f_verts - f_verts.mean(dim=0, keepdim=True)  # (N, 3)

        # Scale-normalised shape displacement
        disp_k  = (f_centered - q_centered).reshape(-1)      # (N*3,)
        disp_k  = disp_k / (scale_sq.sqrt() + eps)           # dimensionless
        disps.append(disp_k)

    D = torch.stack(disps, dim=0)   # (latent_dim, N*3)

    # L2-normalise each displacement vector for cosine similarity
    norms = torch.norm(D, dim=1, keepdim=True).clamp(min=eps)
    D_hat = D / norms               # (latent_dim, N*3)

    # Gram matrix of cosine similarities
    G = D_hat @ D_hat.T             # (latent_dim, latent_dim)

    # Penalise all off-diagonal entries (squared cosine similarity)
    mask   = ~torch.eye(latent_dim, dtype=torch.bool, device=device)
    penalty = (G[mask] ** 2).mean()

    # Gate: ramp to full strength by halfway through training, then plateau.
    # Prevents the term from growing unboundedly as rho->1 competes with planarity.
    diversity_rho = min(rho * 2.0, 1.0)
    return diversity_rho * penalty


# =============================================================================
# Metric-preserving regularizer
# =============================================================================

def metric_preserving_loss(f_z, z, sigma_rho):
    B = f_z.shape[0]
    if B < 2 or sigma_rho < 1e-8:
        return torch.tensor(0.0, device=f_z.device)

    diff_f = f_z.unsqueeze(0) - f_z.unsqueeze(1)
    dist_f = torch.norm(diff_f, dim=2)
    diff_z = z.unsqueeze(0) - z.unsqueeze(1)
    dist_z = torch.norm(diff_z, dim=2)

    mask   = ~torch.eye(B, dtype=torch.bool, device=f_z.device)
    dist_f = dist_f[mask]
    dist_z = dist_z[mask]

    eps       = 1e-8
    ratio     = dist_f / (sigma_rho * dist_z + eps)
    log_ratio = torch.log(ratio + eps)
    return (log_ratio ** 2).mean()


# =============================================================================
# Training
# =============================================================================

def train(args):
    device    = torch.device("cpu")
    mode      = args.mode
    mesh_type = args.mesh
    print(f"Training mode: {mode}, mesh: {mesh_type}")
    print(f"  latent_dim={args.latent_dim}, device={device}")

    # ------------------------------------------------------------------ mesh --
    disp_mask          = None
    bottom_lip_mask_np = None
    mesh_file_path     = getattr(args, "mesh_file", None)

    if mesh_file_path:
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for OBJ mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = load_obj(mesh_file_path)
        mesh_type = "obj"
    elif mesh_type == "grid":
        vertices_rest, faces_np, interior_mask = make_quad_grid(4, 4)
    elif mesh_type == "box":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for box mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = make_open_box(2)
    elif mesh_type == "semiTri":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for semiTri mesh")
        vertices_rest, faces_np, interior_mask = make_semicircle_tri()
    elif mesh_type == "hemiTri":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for hemiTri mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = make_hemisphere_tri()
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    num_verts    = len(vertices_rest)
    edges_np     = get_all_edges(faces_np)
    interior_indices_np = np.where(interior_mask)[0]
    boundary_indices_np = np.where(~interior_mask)[0]

    edges        = torch.from_numpy(edges_np).long().to(device)
    faces        = torch.from_numpy(faces_np).long().to(device)
    rest_lengths = torch.from_numpy(
        np.linalg.norm(
            vertices_rest[edges_np[:, 1]] - vertices_rest[edges_np[:, 0]], axis=1
        )
    ).float().to(device)

    rest_areas_list = []
    for face in faces_np:
        if len(face) == 4:
            v0, v1, v2, v3 = vertices_rest[face]
            a1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            a2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
            rest_areas_list.append(a1 + a2)
        elif len(face) == 3:
            v0, v1, v2 = vertices_rest[face]
            rest_areas_list.append(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
        else:
            raise ValueError(f"Unsupported face size {len(face)}")
    rest_areas = torch.tensor(rest_areas_list, dtype=torch.float32, device=device)

    adj_pairs_np      = get_face_adjacency(faces_np)
    adj_pairs         = torch.from_numpy(adj_pairs_np).long().to(device)
    rest_verts_tensor = torch.from_numpy(vertices_rest).float().to(device)
    mean_rest_area    = float(rest_areas.mean().item())

    if bottom_lip_mask_np is not None and bottom_lip_mask_np.any():
        mask_np     = np.ones(num_verts * 3, dtype=np.float32)
        lip_indices = np.where(bottom_lip_mask_np)[0]
        for idx in lip_indices:
            mask_np[idx * 3 + 2] = 0.0
        disp_mask   = torch.from_numpy(mask_np).to(device)
        num_frozen  = int(bottom_lip_mask_np.sum())
        mesh_label  = "OBJ" if mesh_type == "obj" else mesh_type.capitalize()
        print(f"  {mesh_label} mesh: {num_verts} verts, {num_frozen} bottom-lip "
              f"(z frozen), {num_verts - num_frozen} fully free")

    # --------------------------------------------------------- mode setup ----
    if mode == "anchored":
        output_dim       = len(interior_indices_np) * 2
        interior_indices = torch.from_numpy(interior_indices_np).long().to(device)
        boundary_xy      = torch.from_numpy(
            vertices_rest[boundary_indices_np, :2]).float().to(device)
        q_seed = torch.from_numpy(
            vertices_rest[interior_indices_np, :2].flatten()).float().to(device)
        print(f"  Anchored: {len(interior_indices_np)} interior verts, {output_dim} DOF")

    elif mode == "free":
        output_dim       = num_verts * 2
        interior_indices = None
        boundary_xy      = None
        q_seed = torch.from_numpy(vertices_rest[:, :2].flatten()).float().to(device)
        print(f"  Free: all {num_verts} verts, {output_dim} DOF (2D)")

    elif mode in ("free3d", "old_diag_penalty", "stiffFree3d", "structural3d"):
        output_dim       = num_verts * 3
        interior_indices = None
        boundary_xy      = None
        q_seed = torch.from_numpy(vertices_rest.flatten()).float().to(device)

        if mode == "free3d":
            print(f"  Free3D: {num_verts} verts, {output_dim} DOF (3D)")
            print(f"  w_planarity={args.w_planarity}")
        elif mode == "old_diag_penalty":
            print(f"  OldDiagPenalty (3D): {num_verts} verts, {output_dim} DOF (3D)")
        elif mode == "stiffFree3d":
            print(f"  StiffFree3D: {num_verts} verts, {output_dim} DOF (3D)")
            print(f"  k_xy={args.k_xy}, k_z={args.k_z}")
        elif mode == "structural3d":
            if args.anticollapse_eps <= 0.0:
                args.anticollapse_eps = 0.15 * mean_rest_area
                print(f"  anticollapse_eps auto-set to {args.anticollapse_eps:.5f} "
                      f"(15% of mean rest area {mean_rest_area:.5f})")
            print(f"  Structural3D: {num_verts} verts, {output_dim} DOF (3D)")
            print(f"  w_planarity={args.w_planarity}  w_anticollapse={args.w_anticollapse}  "
                  f"anticollapse_eps={args.anticollapse_eps:.5f}")
            print(f"  w_dirichlet={args.w_dirichlet}  "
                  f"w_trivial={args.w_trivial}  trivial_eps={args.trivial_eps}")
            print(f"  w_diversity={args.w_diversity}  diversity_alpha={args.diversity_alpha}")
            print(f"  {len(adj_pairs_np)} adjacent face pairs")
            if args.w_edge > 0.0:
                print(f"  w_edge={args.w_edge} (weak rest-state anchor)")
            if args.w_area > 0.0:
                print(f"  w_area={args.w_area} (weak rest-state anchor)")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ----------------------------------------------------------- model ------
    model = SubspaceDecoder(
        latent_dim=args.latent_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params}")

    # -------------------------------------------------------- optimizer -----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_steps, gamma=0.5)

    # ------------------------------------------------------- training loop --
    print(f"\nTraining for {args.num_steps} steps...")
    print(f"  σ={args.sigma}, λ={args.lam}, w_anchor={args.w_anchor}")
    print(f"  batch_size={args.batch_size}, hidden={args.hidden_dim}, "
          f"layers={args.num_layers}")
    print()

    log_interval = 1000
    start_time   = time.time()
    losses_log   = []

    for step in range(1, args.num_steps + 1):
        rho       = min(step / args.num_steps, 1.0)
        sigma_rho = args.sigma * rho

        z   = torch.randn(args.batch_size, args.latent_dim, device=device)
        f_z = seeded_forward(model, z, q_seed, rho, disp_mask=disp_mask)

        # ------------------------------------------- potential energy -------
        if mode == "anchored":
            e_pot = potential_energy_anchored(
                f_z, boundary_xy, interior_indices,
                edges, rest_lengths, faces, rest_areas, w_edge=1.0, w_area=0.5)
        elif mode == "free":
            e_pot = potential_energy_free(
                f_z, edges, rest_lengths, faces, rest_areas, w_edge=1.0, w_area=0.5)
        elif mode == "free3d":
            e_pot = potential_energy_free3d(
                f_z, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5, w_planarity=args.w_planarity,
                w_diag_planarity=getattr(args, "w_diag_planarity", 0.0),
                w_inverse_diag=getattr(args, "w_inverse_diag", 0.0),
                w_edge_length=getattr(args, "w_edge_length", 0.0),
                w_edge_inequality_10=getattr(args, "w_edge_inequality_10", 0.0),
                w_width=getattr(args, "w_width", 0.0),
                w_height=getattr(args, "w_height", 0.0))
        elif mode == "old_diag_penalty":
            e_pot = potential_energy_old_diag(
                f_z, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5,
                w_old_diag=getattr(args, "w_diag_planarity", 1.0),
                w_inverse_diag=getattr(args, "w_inverse_diag", 0.0),
                w_edge_length=getattr(args, "w_edge_length", 0.0),
                w_edge_inequality_10=getattr(args, "w_edge_inequality_10", 0.0),
                w_width=getattr(args, "w_width", 0.0),
                w_height=getattr(args, "w_height", 0.0))
        elif mode == "stiffFree3d":
            e_pot = potential_energy_stiff3d(
                f_z, q_seed, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5, k_xy=args.k_xy, k_z=args.k_z,
                w_diag_planarity=getattr(args, "w_diag_planarity", 0.0),
                w_inverse_diag=getattr(args, "w_inverse_diag", 0.0),
                w_edge_length=getattr(args, "w_edge_length", 0.0),
                w_edge_inequality_10=getattr(args, "w_edge_inequality_10", 0.0),
                w_width=getattr(args, "w_width", 0.0),
                w_height=getattr(args, "w_height", 0.0))
        else:  # structural3d
            e_pot = potential_energy_structural3d(
                f_z, rest_verts_tensor,
                edges, rest_lengths, faces, adj_pairs, rest_areas,
                w_planarity=args.w_planarity,
                w_anticollapse=args.w_anticollapse,
                anticollapse_eps=args.anticollapse_eps,
                w_dirichlet=args.w_dirichlet,
                w_edge=args.w_edge,
                w_area=args.w_area)

        loss_energy = e_pot.mean()

        # --------------------------------------- anti-triviality (new) ------
        if mode == "structural3d" and args.w_trivial > 0.0:
            loss_trivial = args.w_trivial * anti_triviality_loss(
                f_z, q_seed, z, rho, eps=args.trivial_eps)
        else:
            loss_trivial = torch.tensor(0.0, device=device)

        # --------------------------------------- axis diversity (option 3) --
        if mode == "structural3d" and args.w_diversity > 0.0:
            loss_diversity = args.w_diversity * axis_diversity_loss(
                model, q_seed, rho, args.latent_dim,
                alpha=args.diversity_alpha,
                disp_mask=disp_mask)
        else:
            loss_diversity = torch.tensor(0.0, device=device)

        # ---------------------------------------- metric regularizer -------
        loss_metric = metric_preserving_loss(f_z, z, sigma_rho)

        # --------------------------------------------- anchor loss ---------
        z_zero      = torch.zeros(1, args.latent_dim, device=device)
        mlp_at_zero = model(z_zero)
        if disp_mask is not None:
            mlp_at_zero = mlp_at_zero * disp_mask.unsqueeze(0)
        loss_anchor = (mlp_at_zero ** 2).sum()

        # ---------------------------------------------------- total --------
        loss = (loss_energy
                + loss_trivial
                + loss_diversity
                + args.lam * loss_metric
                + args.w_anchor * loss_anchor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_now  = optimizer.param_groups[0]['lr']
            log_entry = {
                "step": step, "rho": rho,
                "loss": loss.item(),
                "loss_energy": loss_energy.item(),
                "loss_trivial": loss_trivial.item(),
                "loss_diversity": loss_diversity.item(),
                "loss_metric": loss_metric.item(),
                "loss_anchor": loss_anchor.item(),
                "lr": lr_now,
            }
            losses_log.append(log_entry)
            print(f"  step {step:6d}/{args.num_steps} | ρ={rho:.3f} | "
                  f"loss={loss.item():.6f} "
                  f"(E={loss_energy.item():.4f} "
                  f"T={loss_trivial.item():.4f} "
                  f"M={loss_metric.item():.4f} "
                  f"A={loss_anchor.item():.4f})"
                  f" | lr={lr_now:.2e} | {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    # ------------------------------------------------------------ save -----
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pt")
    save_dict  = {
        "model_state_dict": model.state_dict(),
        "latent_dim":       args.latent_dim,
        "output_dim":       output_dim,
        "hidden_dim":       args.hidden_dim,
        "num_layers":       args.num_layers,
        "mode":             mode,
        "mesh":             mesh_type,
        "num_verts":        num_verts,
        "q_seed":           q_seed.cpu().numpy().tolist(),
        "sigma":            args.sigma,
        "lam":              args.lam,
        "w_anchor":         args.w_anchor,
    }
    if mesh_file_path:
        save_dict["mesh_file"] = os.path.abspath(mesh_file_path)
    if mode == "structural3d":
        save_dict.update({
            "w_planarity":      args.w_planarity,
            "w_anticollapse":   args.w_anticollapse,
            "anticollapse_eps": args.anticollapse_eps,
            "w_dirichlet":      args.w_dirichlet,
            "w_trivial":        args.w_trivial,
            "trivial_eps":      args.trivial_eps,
        })
        if args.w_edge > 0.0: save_dict["w_edge"] = args.w_edge
        if args.w_area > 0.0: save_dict["w_area"] = args.w_area
        save_dict["w_diversity"]     = args.w_diversity
        save_dict["diversity_alpha"] = args.diversity_alpha
    if mode == "free3d":
        save_dict["w_planarity"] = args.w_planarity
    if mode == "stiffFree3d":
        save_dict["k_xy"] = args.k_xy
        save_dict["k_z"]  = args.k_z

    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(losses_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    if getattr(args, "plot_loss", False):
        try:
            import matplotlib.pyplot as plt
            steps        = [e["step"]          for e in losses_log]
            total_loss   = [e["loss"]           for e in losses_log]
            energy_loss  = [e["loss_energy"]    for e in losses_log]
            trivial_loss = [e["loss_trivial"]   for e in losses_log]
            metric_loss  = [e["loss_metric"]    for e in losses_log]
            anchor_loss  = [e["loss_anchor"]    for e in losses_log]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, total_loss,   label="total loss")
            ax.plot(steps, energy_loss,  label="energy")
            ax.plot(steps, trivial_loss, label="anti-triviality")
            ax.plot(steps, metric_loss,  label="metric")
            ax.plot(steps, anchor_loss,  label="anchor")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title("Training loss breakdown")
            ax.legend()
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            plot_path = os.path.join(args.output_dir, "loss_plot.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate loss plot: {e}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train neural subspace for planar quad mesh")

    parser.add_argument("--mesh", type=str, default="grid",
                        choices=["grid", "box", "semiTri", "hemiTri"])
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="anchored",
                        choices=["anchored", "free", "free3d", "stiffFree3d",
                                 "old_diag_penalty", "structural3d"])

    parser.add_argument("--latent_dim",  type=int,   default=6)
    parser.add_argument("--hidden_dim",  type=int,   default=64)
    parser.add_argument("--num_layers",  type=int,   default=5)
    parser.add_argument("--num_steps",   type=int,   default=50000)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lr_decay_steps", type=int, default=12500)

    parser.add_argument("--sigma",    type=float, default=0.5)
    parser.add_argument("--lam",      type=float, default=1.0)
    parser.add_argument("--w_anchor", type=float, default=10.0)

    parser.add_argument("--w_planarity",      type=float, default=10.0)
    parser.add_argument("--w_diag_planarity", type=float, default=0.0)
    parser.add_argument("--w_inverse_diag",   type=float, default=0.0)
    parser.add_argument("--w_edge_length",    type=float, default=0.0)
    parser.add_argument("--w_edge_inequality_10", type=float, default=0.0)
    parser.add_argument("--w_width",          type=float, default=0.0)
    parser.add_argument("--w_height",         type=float, default=0.0)
    parser.add_argument("--k_xy", type=float, default=1.0)
    parser.add_argument("--k_z",  type=float, default=1.0)

    # structural3d
    parser.add_argument("--w_anticollapse",  type=float, default=10.0)
    parser.add_argument("--anticollapse_eps", type=float, default=-1.0,
                        help="Auto-set to 15%% of mean rest area if <= 0")
    parser.add_argument("--w_dirichlet",     type=float, default=0.05,
                        help="Inter-face smoothness weight. Kept weak by default "
                             "to avoid suppressing folds/inversions.")
    parser.add_argument("--w_edge",          type=float, default=0.0)
    parser.add_argument("--w_area",          type=float, default=0.0)

    # anti-triviality
    parser.add_argument("--w_trivial", type=float, default=1.0,
                        help="[structural3d] Weight for anti-triviality term. "
                             "Penalises large ||z|| mapping to near-rest configurations. "
                             "Gated by rho^2 so inactive during warmup.")
    parser.add_argument("--trivial_eps", type=float, default=1e-4,
                        help="[structural3d] Epsilon in anti-triviality denominator "
                             "to prevent division by zero.")

    # axis diversity
    parser.add_argument("--w_diversity", type=float, default=1.0,
                        help="[structural3d] Weight for axis-aligned diversity loss. "
                             "Penalises cosine similarity between per-axis shape displacements.")
    parser.add_argument("--diversity_alpha", type=float, default=1.0,
                        help="[structural3d] Probe magnitude along each latent axis. "
                             "Set to ~O(sigma) so probes reach meaningful deformation regions.")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_loss",  action="store_true")

    args = parser.parse_args()

    if args.output_dir is None:
        def _fmt(x):
            return f"{x:.0f}" if x == int(x) else str(x)
        parts = ["checkpoints", args.mode]
        if args.mesh_file:
            parts.append("obj_" + os.path.splitext(os.path.basename(args.mesh_file))[0])
        elif args.mesh != "grid":
            parts.append(args.mesh)
        parts.append(f"d{args.latent_dim}")
        if args.mode == "structural3d":
            parts.append(f"wp{_fmt(args.w_planarity)}")
            parts.append(f"wac{_fmt(args.w_anticollapse)}")
            parts.append(f"wd{_fmt(args.w_dirichlet)}")
            parts.append(f"wt{_fmt(args.w_trivial)}")
            if args.w_diversity > 0.0:
                parts.append(f"wd{_fmt(args.w_diversity)}")
        elif args.mode == "free3d":
            parts.append(f"wp{args.w_planarity}")
        elif args.mode == "stiffFree3d":
            parts.append(f"kxy{args.k_xy}_kz{args.k_z}")
        args.output_dir = os.path.join(*parts)

    print(f"Output directory: {args.output_dir}")
    train(args)
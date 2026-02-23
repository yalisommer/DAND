"""
Training script: Learn a neural subspace for quad meshes.

Meshes (--mesh):
  - "grid":  4×4 planar quad grid (25 verts, 16 faces)
  - "box":   open box / cube missing bottom (25 verts @ n=2, 20 faces)
             bottom-lip vertices have z frozen at 0

Modes (--mode):
  - "anchored":     boundary fixed, 9 interior verts × 2D (18 DOF) [grid only]
  - "free":         all verts × 2D (50 DOF) [grid only]
  - "free3d":       all verts × 3D with volumetric planarity penalty
  - "stiffFree3d":  all verts × 3D with anisotropic stiffness
                    Energy includes k_xy·Σ(Δxy²) + k_z·Σ(z²).

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
    get_all_edges,
    compute_edge_energy_torch,
    compute_quad_area_energy_torch,
    compute_planarity_energy_torch,
    assemble_vertices_torch,
)


# =============================================================================
# MLP Decoder (used for ALL modes)
# =============================================================================

class SubspaceDecoder(nn.Module):
    """
    MLP that maps latent z ∈ R^d to vertex displacement vector.
    Architecture: hidden layers with ELU activation (following the paper).
    """

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
# Seeded subspace map (residual formulation)
# =============================================================================

def seeded_forward(model, z, q_seed, rho, disp_mask=None):
    """
    Residual formulation:  f_θ(z) = q_seed + ρ * MLP_θ(z)

    - At ρ=0 (start of training): f(z) = q_seed  (rest state, regardless of z)
    - At ρ=1 (end / inference):   f(z) = q_seed + MLP(z)
    - Anchor loss ensures MLP(0) ≈ 0, so f(0) ≈ q_seed.
    - disp_mask: optional (output_dim,) tensor of 0/1 to freeze certain DOFs
                 (e.g. z-component of bottom-lip vertices for the box mesh).
    """
    mlp_out = model(z)
    if disp_mask is not None:
        mlp_out = mlp_out * disp_mask.unsqueeze(0)
    q_seed_expanded = q_seed.unsqueeze(0).expand_as(mlp_out)
    return q_seed_expanded + rho * mlp_out


# =============================================================================
# Loss functions
# =============================================================================

def potential_energy_anchored(pred_batch, boundary_xy, interior_indices,
                              edges, rest_lengths, faces, rest_areas,
                              w_edge=1.0, w_area=0.5):
    """Potential energy for ANCHORED mode."""
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = assemble_vertices_torch(
            pred_batch[i], boundary_xy, interior_indices
        )
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        energies.append(w_edge * e_edge + w_area * e_area)
    return torch.stack(energies)


def potential_energy_free(pred_batch, edges, rest_lengths, faces, rest_areas,
                          w_edge=1.0, w_area=0.5):
    """Potential energy for FREE mode (2D)."""
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 2)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        energies.append(w_edge * e_edge + w_area * e_area)
    return torch.stack(energies)


def potential_energy_free3d(pred_batch, edges, rest_lengths, faces, rest_areas,
                            w_edge=1.0, w_area=0.5, w_planarity=10.0):
    """Potential energy for FREE3D mode (3D with volumetric planarity penalty)."""
    B = pred_batch.shape[0]
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 3)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e_planar = compute_planarity_energy_torch(all_verts, faces)
        energies.append(w_edge * e_edge + w_area * e_area + w_planarity * e_planar)
    return torch.stack(energies)


def potential_energy_stiff3d(pred_batch, q_seed, edges, rest_lengths, faces, rest_areas,
                             w_edge=1.0, w_area=0.5, k_xy=1.0, k_z=1.0):
    """
    Potential energy for STIFFFREE3D mode with anisotropic displacement springs.
    E = edge + area + k_xy·Σ(Δxᵢ²+Δyᵢ²) + k_z·Σzᵢ²

    Both xy and z displacement are penalized.  The RATIO k_z/k_xy determines
    whether the model prefers in-plane or out-of-plane deformation:
      k_z < k_xy  →  z is relatively cheap  →  model discovers bending
      k_z = k_xy  →  isotropic (no preference)
      k_z > k_xy  →  z is expensive  →  stays flat
    """
    B = pred_batch.shape[0]
    rest_verts = q_seed.view(-1, 3)
    energies = []
    for i in range(B):
        all_verts = pred_batch[i].view(-1, 3)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        disp = all_verts - rest_verts
        e_xy = torch.sum(disp[:, :2] ** 2)
        e_z = torch.sum(disp[:, 2] ** 2)
        energies.append(w_edge * e_edge + w_area * e_area + k_xy * e_xy + k_z * e_z)
    return torch.stack(energies)


def metric_preserving_loss(f_z, z, sigma_rho):
    """
    Metric-preserving regularizer. Uses SQUARED log to penalize both collapse and stretching.
    """
    B = f_z.shape[0]
    if B < 2 or sigma_rho < 1e-8:
        return torch.tensor(0.0, device=f_z.device)

    diff_f = f_z.unsqueeze(0) - f_z.unsqueeze(1)
    dist_f = torch.norm(diff_f, dim=2)

    diff_z = z.unsqueeze(0) - z.unsqueeze(1)
    dist_z = torch.norm(diff_z, dim=2)

    mask = ~torch.eye(B, dtype=torch.bool, device=f_z.device)
    dist_f = dist_f[mask]
    dist_z = dist_z[mask]

    eps = 1e-8
    ratio = dist_f / (sigma_rho * dist_z + eps)
    log_ratio = torch.log(ratio + eps)

    return (log_ratio ** 2).mean()


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cpu")
    mode = args.mode
    mesh_type = args.mesh
    print(f"Training mode: {mode}, mesh: {mesh_type}")
    print(f"  latent_dim={args.latent_dim}, device={device}")

    # --- Mesh setup ---
    disp_mask = None  # displacement mask for frozen DOFs (None = all free)
    bottom_lip_mask_np = None

    if mesh_type == "grid":
        vertices_rest, faces_np, interior_mask = make_quad_grid(4, 4)
    elif mesh_type == "box":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for box mesh (use free3d or stiffFree3d)")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = make_open_box(2)
    elif mesh_type == "semiTri":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' is not supported for semiTri mesh (use free3d or stiffFree3d)")
        vertices_rest, faces_np, interior_mask = make_semicircle_tri()
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    num_verts = len(vertices_rest)
    edges_np = get_all_edges(faces_np)

    interior_indices_np = np.where(interior_mask)[0]
    boundary_indices_np = np.where(~interior_mask)[0]

    edges = torch.from_numpy(edges_np).long().to(device)
    faces = torch.from_numpy(faces_np).long().to(device)

    rest_lengths_np = np.linalg.norm(
        vertices_rest[edges_np[:, 1]] - vertices_rest[edges_np[:, 0]], axis=1
    )
    rest_lengths = torch.from_numpy(rest_lengths_np).float().to(device)

    # Compute actual rest areas per face (tris or quads)
    rest_areas_list = []
    for face in faces_np:
        if len(face) == 4:
            v0, v1, v2, v3 = vertices_rest[face]
            a1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            a2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
            rest_areas_list.append(a1 + a2)
        elif len(face) == 3:
            v0, v1, v2 = vertices_rest[face]
            a = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            rest_areas_list.append(a)
        else:
            raise ValueError(f"Unsupported face size {len(face)} when computing rest areas")
    rest_areas = torch.tensor(rest_areas_list, dtype=torch.float32, device=device)

    # Build displacement mask for box mesh (freeze z-component of bottom lip)
    if bottom_lip_mask_np is not None:
        mask_np = np.ones(num_verts * 3, dtype=np.float32)
        lip_indices = np.where(bottom_lip_mask_np)[0]
        for idx in lip_indices:
            mask_np[idx * 3 + 2] = 0.0  # zero out z-component
        disp_mask = torch.from_numpy(mask_np).to(device)
        num_frozen = int(bottom_lip_mask_np.sum())
        print(f"  Box mesh: {num_verts} verts, {num_frozen} bottom-lip (z frozen), "
              f"{num_verts - num_frozen} fully free")

    # --- Mode-specific setup ---
    if mode == "anchored":
        output_dim = len(interior_indices_np) * 2
        interior_indices = torch.from_numpy(interior_indices_np).long().to(device)
        boundary_xy = torch.from_numpy(
            vertices_rest[boundary_indices_np, :2]
        ).float().to(device)
        q_seed = torch.from_numpy(
            vertices_rest[interior_indices_np, :2].flatten()
        ).float().to(device)
        print(f"  Anchored: {len(interior_indices_np)} interior vertices, {output_dim} DOF")

    elif mode == "free":
        output_dim = num_verts * 2
        interior_indices = None
        boundary_xy = None
        q_seed = torch.from_numpy(
            vertices_rest[:, :2].flatten()
        ).float().to(device)
        print(f"  Free: all {num_verts} vertices, {output_dim} DOF (2D)")

    elif mode == "free3d":
        output_dim = num_verts * 3
        interior_indices = None
        boundary_xy = None
        q_seed = torch.from_numpy(
            vertices_rest.flatten()
        ).float().to(device)
        print(f"  Free3D: {num_verts} vertices, {output_dim} DOF (3D)")
        print(f"  w_planarity={args.w_planarity}")

    elif mode == "stiffFree3d":
        output_dim = num_verts * 3
        interior_indices = None
        boundary_xy = None
        q_seed = torch.from_numpy(
            vertices_rest.flatten()
        ).float().to(device)
        print(f"  StiffFree3D: {num_verts} vertices, {output_dim} DOF (3D)")
        print(f"  k_xy={args.k_xy}, k_z={args.k_z} (ratio k_z/k_xy={args.k_z/args.k_xy:.2f})")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Model (same architecture for all modes) ---
    model = SubspaceDecoder(
        latent_dim=args.latent_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_steps, gamma=0.5
    )

    # --- Training loop ---
    print(f"\nTraining for {args.num_steps} steps...")
    print(f"  σ={args.sigma}, λ={args.lam}, w_anchor={args.w_anchor}")
    print(f"  batch_size={args.batch_size}, hidden={args.hidden_dim}, layers={args.num_layers}")
    print()

    log_interval = 1000
    start_time = time.time()
    losses_log = []

    for step in range(1, args.num_steps + 1):
        rho = min(step / args.num_steps, 1.0)
        sigma_rho = args.sigma * rho

        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        f_z = seeded_forward(model, z, q_seed, rho, disp_mask=disp_mask)

        # --- Potential energy ---
        if mode == "anchored":
            e_pot = potential_energy_anchored(
                f_z, boundary_xy, interior_indices,
                edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5,
            )
        elif mode == "free":
            e_pot = potential_energy_free(
                f_z, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5,
            )
        elif mode == "free3d":
            e_pot = potential_energy_free3d(
                f_z, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5, w_planarity=args.w_planarity,
            )
        else:  # stiffFree3d
            e_pot = potential_energy_stiff3d(
                f_z, q_seed, edges, rest_lengths, faces, rest_areas,
                w_edge=1.0, w_area=0.5, k_xy=args.k_xy, k_z=args.k_z,
            )
        loss_energy = e_pot.mean()

        # --- Metric-preserving regularizer ---
        loss_metric = metric_preserving_loss(f_z, z, sigma_rho)

        # --- Anchor loss: MLP(0) → 0 so z=0 maps to q_seed ---
        z_zero = torch.zeros(1, args.latent_dim, device=device)
        mlp_at_zero = model(z_zero)
        if disp_mask is not None:
            mlp_at_zero = mlp_at_zero * disp_mask.unsqueeze(0)
        loss_anchor = (mlp_at_zero ** 2).mean()

        # --- Total loss ---
        loss = loss_energy + args.lam * loss_metric + args.w_anchor * loss_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            log_entry = {
                "step": step, "rho": rho,
                "loss": loss.item(),
                "loss_energy": loss_energy.item(),
                "loss_metric": loss_metric.item(),
                "loss_anchor": loss_anchor.item(),
                "lr": lr_now,
            }

            losses_log.append(log_entry)
            print(f"  step {step:6d}/{args.num_steps} | ρ={rho:.3f} | "
                  f"loss={loss.item():.6f} (E={loss_energy.item():.4f} "
                  f"M={loss_metric.item():.4f} A={loss_anchor.item():.4f})"
                  f" | lr={lr_now:.2e} | {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    # --- Save model ---
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pt")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "output_dim": output_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "mode": mode,
        "mesh": mesh_type,
        "num_verts": num_verts,
        "q_seed": q_seed.cpu().numpy().tolist(),
        "sigma": args.sigma,
        "lam": args.lam,
        "w_anchor": args.w_anchor,
    }
    if mode == "free3d":
        save_dict["w_planarity"] = args.w_planarity
    if mode == "stiffFree3d":
        save_dict["k_xy"] = args.k_xy
        save_dict["k_z"] = args.k_z
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(losses_log, f, indent=2)
    print(f"Training log saved to {log_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural subspace for planar quad mesh")

    # Mesh
    parser.add_argument("--mesh", type=str, default="grid",
                        choices=["grid", "box", "semiTri"],
                        help="Mesh type: 'grid' (flat 4x4), 'box' (open cube, n=2), "
                             "or 'semiTri' (triangulated semicircle fan)")

    # Mode
    parser.add_argument("--mode", type=str, default="anchored",
                        choices=["anchored", "free", "free3d", "stiffFree3d"],
                        help="Training mode")

    # Model
    parser.add_argument("--latent_dim", type=int, default=6, help="Latent space dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden layer width")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of hidden layers")

    # Training
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay_steps", type=int, default=12500, help="Steps between LR decay")

    # Loss hyperparameters
    parser.add_argument("--sigma", type=float, default=0.5, help="Subspace scale σ")
    parser.add_argument("--lam", type=float, default=1.0, help="Metric-preserving weight λ")
    parser.add_argument("--w_anchor", type=float, default=10.0,
                        help="Anchor loss weight: penalizes MLP(0) != 0 so z=0 → rest state")

    # Free3D-specific
    parser.add_argument("--w_planarity", type=float, default=10.0,
                        help="Planarity penalty weight (higher = flatter mesh)")

    # StiffFree3D-specific
    parser.add_argument("--k_xy", type=float, default=1.0,
                        help="In-plane (xy) displacement stiffness")
    parser.add_argument("--k_z", type=float, default=1.0,
                        help="Out-of-plane (z) displacement stiffness. "
                             "Ratio k_z/k_xy controls bending preference")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")

    args = parser.parse_args()

    # Auto-generate output_dir
    if args.output_dir is None:
        parts = ["checkpoints", args.mode]
        if args.mesh != "grid":
            parts.append(args.mesh)
        parts.append(f"d{args.latent_dim}")
        if args.mode == "free3d":
            parts.append(f"wp{args.w_planarity}")
        if args.mode == "stiffFree3d":
            parts.append(f"kxy{args.k_xy}_kz{args.k_z}")
        args.output_dir = os.path.join(*parts)
    print(f"Output directory: {args.output_dir}")

    train(args)

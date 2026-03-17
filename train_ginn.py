"""
Variant 2 — Geometry-Informed Neural Network (GINN) training.

Replaces the metric-preserving regularizer with a formal diversity constraint,
and treats planarity as a hard feasibility check that gates which samples
contribute to the diversity objective.

Theory:
  min_θ  Objective(f_θ)             — potential energy
  s.t.   Feasibility(f_θ(z)) ≤ ε   — planarity ≤ ε
         Diversity(f_θ)   ≥ δ       — pairwise spread ≥ δ

  Diversity is enforced via its own ALM inequality dual variable.
  Feasibility gates which samples enter the energy objective.

Reference: Knöbelreiter et al. (2024) — arxiv.org/abs/2402.14009

Shares mesh infrastructure, SubspaceDecoder, and seeded_forward from
train.py / mesh.py verbatim.
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
    compute_planarity_energy_per_sample_torch,
    compute_diag_planarity_energy_torch,
    assemble_vertices_torch,
)

from train import SubspaceDecoder, seeded_forward


# =============================================================================
# Per-sample helpers
# =============================================================================

def _base_energy_per_sample(f_z, mode, q_seed, edges, rest_lengths, faces,
                            rest_areas, k_xy, k_z):
    """Edge + area (+ displacement springs for stiffFree3d). Returns (B,)."""
    B = f_z.shape[0]
    rest_verts = q_seed.view(-1, 3) if mode == "stiffFree3d" else None
    energies = []
    for i in range(B):
        dim = 3 if mode not in ("anchored", "free") else 2
        all_verts = f_z[i].view(-1, dim)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e = 1.0 * e_edge + 0.5 * e_area
        if mode == "stiffFree3d":
            disp = all_verts - rest_verts
            e = e + k_xy * torch.sum(disp[:, :2] ** 2) + k_z * torch.sum(disp[:, 2] ** 2)
        energies.append(e)
    return torch.stack(energies)


def diversity_loss(f_z):
    """Mean pairwise L2 distance in output space. Scalar."""
    B = f_z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=f_z.device)
    diff = f_z.unsqueeze(0) - f_z.unsqueeze(1)        # (B, B, D)
    dists = torch.norm(diff, dim=2)                    # (B, B)
    mask = ~torch.eye(B, dtype=torch.bool, device=f_z.device)
    return dists[mask].mean()


# =============================================================================
# Training
# =============================================================================

def train_ginn(args):
    device = torch.device("cpu")
    mode = args.mode
    mesh_type = args.mesh
    print(f"[GINN] Training mode: {mode}, mesh: {mesh_type}")
    print(f"  latent_dim={args.latent_dim}, device={device}")

    # --- Mesh setup (identical to train.py) ---
    disp_mask = None
    bottom_lip_mask_np = None
    mesh_file_path = getattr(args, "mesh_file", None)

    if mesh_file_path:
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' not supported for OBJ mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = load_obj(
            mesh_file_path,
            anchor_bottom_z=getattr(args, "floor_anchors", False),
        )
        mesh_type = "obj"
    elif mesh_type == "grid":
        vertices_rest, faces_np, interior_mask = make_quad_grid(4, 4)
    elif mesh_type == "box":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' not supported for box mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = make_open_box(2)
    elif mesh_type == "semiTri":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' not supported for semiTri")
        vertices_rest, faces_np, interior_mask = make_semicircle_tri()
    elif mesh_type == "hemiTri":
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' not supported for hemiTri")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = make_hemisphere_tri()
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    num_verts = len(vertices_rest)
    edges_np = get_all_edges(faces_np)
    edges = torch.from_numpy(edges_np).long().to(device)
    faces = torch.from_numpy(faces_np).long().to(device)

    rest_lengths_np = np.linalg.norm(
        vertices_rest[edges_np[:, 1]] - vertices_rest[edges_np[:, 0]], axis=1
    )
    rest_lengths = torch.from_numpy(rest_lengths_np).float().to(device)

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

    if bottom_lip_mask_np is not None:
        mask_np = np.ones(num_verts * 3, dtype=np.float32)
        lip_indices = np.where(bottom_lip_mask_np)[0]
        for idx in lip_indices:
            mask_np[idx * 3 + 2] = 0.0
        disp_mask = torch.from_numpy(mask_np).to(device)
        num_frozen = int(bottom_lip_mask_np.sum())
        label = "OBJ" if mesh_type == "obj" else mesh_type.capitalize()
        print(f"  {label} mesh: {num_verts} verts, {num_frozen} bottom-lip (z frozen)")

    # --- Mode-specific setup ---
    if mode in ("free3d", "stiffFree3d", "old_diag_penalty"):
        output_dim = num_verts * 3
        q_seed = torch.from_numpy(vertices_rest.flatten()).float().to(device)
    elif mode == "anchored":
        interior_indices_np = np.where(interior_mask)[0]
        output_dim = len(interior_indices_np) * 2
        q_seed = torch.from_numpy(vertices_rest[interior_indices_np, :2].flatten()).float().to(device)
    elif mode == "free":
        output_dim = num_verts * 2
        q_seed = torch.from_numpy(vertices_rest[:, :2].flatten()).float().to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  output_dim={output_dim}")
    if mode == "stiffFree3d":
        print(f"  k_xy={args.k_xy}, k_z={args.k_z}")

    is_3d = mode in ("free3d", "stiffFree3d", "old_diag_penalty")

    # --- Model ---
    model = SubspaceDecoder(
        latent_dim=args.latent_dim, output_dim=output_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_steps, gamma=0.5
    )

    # --- GINN diversity constraint ALM dual variable ---
    lam_div = torch.tensor(0.0, device=device)
    rho_div = args.rho_div_init
    G_div = 0.0
    eps_adagrad = 1e-8

    print(f"\n[GINN] Training for {args.num_steps} steps...")
    print(f"  feasibility_eps={args.feasibility_eps}")
    print(f"  diversity_delta={args.diversity_delta}, warmup={args.diversity_warmup}")
    print(f"  ρ_div_init={args.rho_div_init}, ρ_div_max={args.rho_div_max}")
    print(f"  w_anchor={args.w_anchor}")
    print()

    log_interval = 1000
    start_time = time.time()
    losses_log = []

    for step in range(1, args.num_steps + 1):
        rho_train = min(step / args.num_steps, 1.0)

        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        f_z = seeded_forward(model, z, q_seed, rho_train, disp_mask=disp_mask)

        # --- Per-sample base energy ---
        e_pot = _base_energy_per_sample(
            f_z, mode, q_seed, edges, rest_lengths, faces, rest_areas,
            args.k_xy if mode == "stiffFree3d" else 1.0,
            args.k_z if mode == "stiffFree3d" else 1.0,
        )

        # --- Feasibility: per-sample planarity check ---
        if is_3d:
            verts_batch = f_z.view(f_z.shape[0], -1, 3)
            e_planar_batch = compute_planarity_energy_per_sample_torch(verts_batch, faces)
            feasible_mask = (e_planar_batch < args.feasibility_eps)
            feasible_frac = feasible_mask.float().mean().item()

            if feasible_mask.any():
                loss_energy = e_pot[feasible_mask].mean()
            else:
                loss_energy = e_planar_batch.mean()

            loss_planar_soft = args.w_planarity * e_planar_batch.mean()
        else:
            feasible_frac = 1.0
            e_planar_batch = torch.zeros(f_z.shape[0], device=device)
            loss_energy = e_pot.mean()
            loss_planar_soft = torch.tensor(0.0, device=device)

        # --- Diversity constraint via ALM (inequality: D ≥ δ) ---
        D = diversity_loss(f_z)
        div_active = step > args.diversity_warmup
        if div_active:
            c_div = args.diversity_delta - D
            c_div_pos = torch.clamp(c_div, min=0.0)
            loss_div_alm = lam_div * c_div + (rho_div / 2.0) * c_div_pos ** 2
        else:
            c_div = args.diversity_delta - D
            loss_div_alm = torch.tensor(0.0, device=device)

        # --- Anchor loss (fixed weight, not ALM) ---
        z_zero = torch.zeros(1, args.latent_dim, device=device)
        mlp_at_zero = model(z_zero)
        if disp_mask is not None:
            mlp_at_zero = mlp_at_zero * disp_mask.unsqueeze(0)
        loss_anchor = (mlp_at_zero ** 2).mean()

        # --- Total loss ---
        loss = (loss_energy
                + loss_planar_soft
                + loss_div_alm
                + args.w_anchor * loss_anchor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Diversity dual update (every 50 steps, after warmup) ---
        if div_active and step % 50 == 0:
            with torch.no_grad():
                cv = c_div.item()
                if cv > 0:
                    G_div += cv ** 2
                    rho_div = args.rho_div_init / (G_div ** 0.5 + eps_adagrad)
                    rho_div = min(rho_div, args.rho_div_max)
                    lam_div += rho_div * cv
                    lam_div = max(lam_div, 0.0)

        # --- Logging ---
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            log_entry = {
                "step": step, "rho_train": rho_train,
                "loss": loss.item(),
                "loss_energy": loss_energy.item(),
                "loss_anchor": loss_anchor.item(),
                "c_planarity_mean": e_planar_batch.mean().item(),
                "feasible_fraction": feasible_frac,
                "diversity_D": D.item(),
                "c_div": c_div.item(),
                "lambda_div": float(lam_div),
                "rho_div": rho_div,
                "lr": lr_now,
            }
            losses_log.append(log_entry)
            print(f"  step {step:6d}/{args.num_steps} | ρ={rho_train:.3f} | "
                  f"loss={loss.item():.6f} (E={loss_energy.item():.4f} "
                  f"A={loss_anchor.item():.4f})"
                  f" | feas={feasible_frac:.2f} D={D.item():.4f}"
                  f" | λD={float(lam_div):.3f}"
                  f" | lr={lr_now:.2e} | {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n[GINN] Training complete in {total_time:.1f}s")

    # --- Save ---
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
        "lam": 0.0,
        "w_anchor": args.w_anchor,
        "variant": "ginn",
        "ginn_feasibility_eps": args.feasibility_eps,
        "ginn_diversity_delta": args.diversity_delta,
        "ginn_lambda_div_final": float(lam_div),
        "ginn_rho_div_final": rho_div,
    }
    if mesh_file_path:
        save_dict["mesh_file"] = os.path.abspath(mesh_file_path)
    if mode == "stiffFree3d":
        save_dict["k_xy"] = args.k_xy
        save_dict["k_z"] = args.k_z
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(losses_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    if getattr(args, "plot_loss", False):
        try:
            import matplotlib.pyplot as plt
            steps = [e["step"] for e in losses_log]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(steps, [e["loss"] for e in losses_log], label="total")
            axes[0, 0].plot(steps, [e["loss_energy"] for e in losses_log], label="energy")
            axes[0, 0].set_title("Loss components")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.2)

            axes[0, 1].plot(steps, [e["feasible_fraction"] for e in losses_log], label="feasible%")
            axes[0, 1].set_ylim(-0.05, 1.05)
            axes[0, 1].set_title("Feasible fraction")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.2)

            axes[1, 0].plot(steps, [e["diversity_D"] for e in losses_log], label="D (pairwise)")
            axes[1, 0].axhline(args.diversity_delta, color="r", ls="--", label=f"δ={args.diversity_delta}")
            axes[1, 0].set_title("Diversity")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.2)

            axes[1, 1].plot(steps, [e["lambda_div"] for e in losses_log], label="λ_div")
            axes[1, 1].plot(steps, [e["rho_div"] for e in losses_log], label="ρ_div")
            axes[1, 1].set_title("Diversity ALM duals")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.2)

            for ax in axes.flat:
                ax.set_xlabel("step")
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
        description="Train neural subspace (GINN variant) for planar quad mesh"
    )

    # Same CLI surface as train.py
    parser.add_argument("--mesh", type=str, default="grid",
                        choices=["grid", "box", "semiTri", "hemiTri"])
    parser.add_argument("--mesh_file", type=str, default=None,
                        help="Path to OBJ file for generic mesh loading")
    parser.add_argument("--floor_anchors", action="store_true",
                        help="For OBJ meshes: freeze detected floor vertices (enable bottom anchors)")
    parser.add_argument("--mode", type=str, default="free3d",
                        choices=["anchored", "free", "free3d", "stiffFree3d", "old_diag_penalty"])
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_steps", type=int, default=12500)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Ignored by GINN (diversity replaces metric reg); kept for CLI compat")
    parser.add_argument("--w_anchor", type=float, default=10.0,
                        help="Anchor loss weight (fixed, not ALM-managed)")
    parser.add_argument("--w_planarity", type=float, default=10.0,
                        help="Soft planarity penalty weight (supplements feasibility gate)")
    parser.add_argument("--w_diag_planarity", type=float, default=0.0,
                        help="Kept for CLI compat")
    parser.add_argument("--k_xy", type=float, default=1.0)
    parser.add_argument("--k_z", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_loss", action="store_true")

    # GINN-specific arguments
    parser.add_argument("--feasibility_eps", type=float, default=0.01,
                        help="Max planarity violation for a sample to count as feasible")
    parser.add_argument("--diversity_delta", type=float, default=0.1,
                        help="Minimum required mean pairwise distance D")
    parser.add_argument("--rho_div_init", type=float, default=0.1,
                        help="Initial ρ for diversity constraint")
    parser.add_argument("--rho_div_max", type=float, default=50.0,
                        help="Max ρ for diversity constraint")
    parser.add_argument("--diversity_warmup", type=int, default=5000,
                        help="Steps before diversity constraint activates")

    args = parser.parse_args()

    if args.output_dir is None:
        parts = ["checkpoints", "ginn"]
        parts.append(args.mode)
        if args.mesh_file:
            mesh_part = "obj_" + os.path.splitext(os.path.basename(args.mesh_file))[0]
        else:
            mesh_part = args.mesh
        if mesh_part != "grid":
            parts.append(mesh_part)
        parts.append(f"d{args.latent_dim}")
        if args.mode == "stiffFree3d":
            parts.append(f"kxy{args.k_xy}_kz{args.k_z}")
        parts.append(f"eps{args.feasibility_eps}_delta{args.diversity_delta}")
        args.output_dir = os.path.join(*parts)
    print(f"Output directory: {args.output_dir}")

    train_ginn(args)

"""
Variant 1 — Augmented Lagrangian Method (ALM) training.

Replaces fixed penalty weights (w_planarity, w_diag_planarity, w_anchor) with
self-adapting Lagrange multipliers so that constraint satisfaction converges
without manual weight tuning.

Theory:
  Standard penalty:       L = E_pot + w · c(q)²
  ALM:                    L = E_pot + λ·c(q) + (ρ/2)·c(q)²

  After every dual_update_interval steps:
    λ ← λ + ρ · c̄
    ρ ← min(ρ · rho_scale, rho_max)

Reference: Basir & Senocak (2023) — arxiv.org/abs/2306.04904

Shares mesh infrastructure, SubspaceDecoder, and seeded_forward from
train.py / mesh.py verbatim.  Accepts the same CLI surface so it can be
swapped into benchmark runs with no flag changes (plus ALM-specific args).
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
    assemble_vertices_torch,
)

from train import SubspaceDecoder, seeded_forward, metric_preserving_loss


# =============================================================================
# Per-sample constraint helpers (return shape (B,))
# =============================================================================

def _base_energy_per_sample(f_z, mode, q_seed, edges, rest_lengths, faces,
                            rest_areas, k_xy, k_z):
    """Edge + area (+ displacement springs for stiffFree3d). Returns (B,)."""
    B = f_z.shape[0]
    energies = []
    rest_verts = q_seed.view(-1, 3) if mode == "stiffFree3d" else None
    for i in range(B):
        if mode in ("anchored", "free"):
            dim = 2
        else:
            dim = 3
        all_verts = f_z[i].view(-1, dim)
        e_edge = compute_edge_energy_torch(all_verts, edges, rest_lengths)
        e_area = compute_quad_area_energy_torch(all_verts, faces, rest_areas)
        e = 1.0 * e_edge + 0.5 * e_area
        if mode == "stiffFree3d":
            disp = all_verts - rest_verts
            e = e + k_xy * torch.sum(disp[:, :2] ** 2) + k_z * torch.sum(disp[:, 2] ** 2)
        energies.append(e)
    return torch.stack(energies)


def _planarity_per_sample(f_z, faces):
    """Volumetric planarity per sample. Returns (B,)."""
    B = f_z.shape[0]
    vals = []
    for i in range(B):
        all_verts = f_z[i].view(-1, 3)
        vals.append(compute_planarity_energy_torch(all_verts, faces))
    return torch.stack(vals)


def _diag_planarity_per_sample(f_z, faces):
    """Diagonal planarity per sample. Returns (B,)."""
    B = f_z.shape[0]
    vals = []
    for i in range(B):
        all_verts = f_z[i].view(-1, 3)
        vals.append(compute_diag_planarity_energy_torch(all_verts, faces))
    return torch.stack(vals)


# =============================================================================
# Training
# =============================================================================

def train_alm(args):
    device = torch.device("cpu")
    mode = args.mode
    mesh_type = args.mesh
    print(f"[ALM] Training mode: {mode}, mesh: {mesh_type}")
    print(f"  latent_dim={args.latent_dim}, device={device}")

    # --- Mesh setup (identical to train.py) ---
    disp_mask = None
    bottom_lip_mask_np = None
    mesh_file_path = getattr(args, "mesh_file", None)

    if mesh_file_path:
        if mode in ("anchored", "free"):
            raise ValueError(f"Mode '{mode}' not supported for OBJ mesh")
        vertices_rest, faces_np, interior_mask, bottom_lip_mask_np = load_obj(mesh_file_path)
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
        boundary_indices_np = np.where(~interior_mask)[0]
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

    # --- ALM dual variables (not optimized by Adam) ---
    lam_planarity = torch.tensor(args.lambda_planarity_init, device=device)
    lam_diag = torch.tensor(0.0, device=device)
    lam_anchor = torch.tensor(args.lambda_anchor_init, device=device)

    rho_planarity = args.rho_init
    rho_diag = args.rho_init
    rho_anchor = args.rho_init

    # AdaGrad-style accumulators for adaptive ρ per constraint
    G_planarity = 0.0
    G_diag = 0.0
    G_anchor = 0.0
    eps_adagrad = 1e-8

    has_planarity = mode in ("free3d", "stiffFree3d", "old_diag_penalty")
    has_diag = has_planarity and faces_np.shape[1] == 4

    print(f"\n[ALM] Training for {args.num_steps} steps...")
    print(f"  ρ_init={args.rho_init}, ρ_max={args.rho_max}, ρ_scale={args.rho_scale}")
    print(f"  dual_update_interval={args.dual_update_interval}")
    print(f"  λ_metric={args.lam} (fixed, not ALM)")
    print(f"  ALM constraints: planarity={has_planarity}, diag={has_diag}, anchor=True")
    print()

    log_interval = 1000
    start_time = time.time()
    losses_log = []

    for step in range(1, args.num_steps + 1):
        rho_train = min(step / args.num_steps, 1.0)
        sigma_rho = args.sigma * rho_train

        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        f_z = seeded_forward(model, z, q_seed, rho_train, disp_mask=disp_mask)

        # --- Base energy (edge + area + displacement springs) ---
        e_base = _base_energy_per_sample(
            f_z, mode, q_seed, edges, rest_lengths, faces, rest_areas,
            args.k_xy if mode == "stiffFree3d" else 1.0,
            args.k_z if mode == "stiffFree3d" else 1.0,
        )
        loss_energy = e_base.mean()

        # --- Planarity constraint via ALM ---
        loss_planar_alm = torch.tensor(0.0, device=device)
        c_planar = torch.tensor(0.0, device=device)
        if has_planarity:
            c_planar_batch = _planarity_per_sample(f_z, faces)
            c_planar = c_planar_batch.mean()
            loss_planar_alm = lam_planarity * c_planar + (rho_planarity / 2.0) * c_planar ** 2

        # --- Diagonal planarity constraint via ALM ---
        loss_diag_alm = torch.tensor(0.0, device=device)
        c_diag = torch.tensor(0.0, device=device)
        if has_diag:
            c_diag_batch = _diag_planarity_per_sample(f_z, faces)
            c_diag = c_diag_batch.mean()
            loss_diag_alm = lam_diag * c_diag + (rho_diag / 2.0) * c_diag ** 2

        # --- Anchor constraint via ALM ---
        z_zero = torch.zeros(1, args.latent_dim, device=device)
        mlp_at_zero = model(z_zero)
        if disp_mask is not None:
            mlp_at_zero = mlp_at_zero * disp_mask.unsqueeze(0)
        c_anchor = (mlp_at_zero ** 2).mean()
        loss_anchor_alm = lam_anchor * c_anchor + (rho_anchor / 2.0) * c_anchor ** 2

        # --- Metric-preserving regularizer (fixed weight, not ALM) ---
        loss_metric = metric_preserving_loss(f_z, z, sigma_rho)

        # --- Total loss ---
        loss = (loss_energy
                + loss_planar_alm
                + loss_diag_alm
                + loss_anchor_alm
                + args.lam * loss_metric)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Dual update ---
        if step % args.dual_update_interval == 0:
            with torch.no_grad():
                if has_planarity:
                    cv = c_planar.item()
                    G_planarity += cv ** 2
                    rho_planarity = args.rho_init / (G_planarity ** 0.5 + eps_adagrad)
                    rho_planarity = min(rho_planarity, args.rho_max)
                    lam_planarity += rho_planarity * cv

                if has_diag:
                    cv = c_diag.item()
                    G_diag += cv ** 2
                    rho_diag = args.rho_init / (G_diag ** 0.5 + eps_adagrad)
                    rho_diag = min(rho_diag, args.rho_max)
                    lam_diag += rho_diag * cv

                cv = c_anchor.item()
                G_anchor += cv ** 2
                rho_anchor = args.rho_init / (G_anchor ** 0.5 + eps_adagrad)
                rho_anchor = min(rho_anchor, args.rho_max)
                lam_anchor += rho_anchor * cv

        # --- Logging ---
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            log_entry = {
                "step": step, "rho_train": rho_train,
                "loss": loss.item(),
                "loss_energy": loss_energy.item(),
                "loss_metric": loss_metric.item(),
                "c_planarity": c_planar.item(),
                "c_diag": c_diag.item(),
                "c_anchor": c_anchor.item(),
                "lambda_planarity": float(lam_planarity),
                "lambda_diag": float(lam_diag),
                "lambda_anchor": float(lam_anchor),
                "rho_planarity": rho_planarity,
                "rho_diag": rho_diag,
                "rho_anchor": rho_anchor,
                "lr": lr_now,
            }
            losses_log.append(log_entry)
            print(f"  step {step:6d}/{args.num_steps} | ρ={rho_train:.3f} | "
                  f"loss={loss.item():.6f} (E={loss_energy.item():.4f} "
                  f"M={loss_metric.item():.4f} "
                  f"cP={c_planar.item():.4f} cA={c_anchor.item():.4f})"
                  f" | λP={float(lam_planarity):.3f} ρP={rho_planarity:.3f}"
                  f" | lr={lr_now:.2e} | {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n[ALM] Training complete in {total_time:.1f}s")

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
        "lam": args.lam,
        "w_anchor": 0.0,
        "variant": "alm",
        "alm_lambda_planarity": float(lam_planarity),
        "alm_lambda_diag": float(lam_diag),
        "alm_lambda_anchor": float(lam_anchor),
        "alm_rho_planarity": rho_planarity,
        "alm_rho_diag": rho_diag,
        "alm_rho_anchor": rho_anchor,
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
            axes[0, 0].plot(steps, [e["loss_metric"] for e in losses_log], label="metric")
            axes[0, 0].set_title("Loss components")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.2)

            axes[0, 1].plot(steps, [e["c_planarity"] for e in losses_log], label="planarity")
            axes[0, 1].plot(steps, [e["c_diag"] for e in losses_log], label="diag")
            axes[0, 1].plot(steps, [e["c_anchor"] for e in losses_log], label="anchor")
            axes[0, 1].set_title("Constraint violations")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.2)

            axes[1, 0].plot(steps, [e["lambda_planarity"] for e in losses_log], label="λ_plan")
            axes[1, 0].plot(steps, [e["lambda_diag"] for e in losses_log], label="λ_diag")
            axes[1, 0].plot(steps, [e["lambda_anchor"] for e in losses_log], label="λ_anchor")
            axes[1, 0].set_title("Lagrange multipliers")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.2)

            axes[1, 1].plot(steps, [e["rho_planarity"] for e in losses_log], label="ρ_plan")
            axes[1, 1].plot(steps, [e["rho_diag"] for e in losses_log], label="ρ_diag")
            axes[1, 1].plot(steps, [e["rho_anchor"] for e in losses_log], label="ρ_anchor")
            axes[1, 1].set_title("Penalty coefficients ρ")
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
        description="Train neural subspace (ALM variant) for planar quad mesh"
    )

    # Same CLI surface as train.py
    parser.add_argument("--mesh", type=str, default="grid",
                        choices=["grid", "box", "semiTri", "hemiTri"])
    parser.add_argument("--mesh_file", type=str, default=None)
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
                        help="Metric-preserving weight (kept as fixed regularizer)")
    parser.add_argument("--w_anchor", type=float, default=10.0,
                        help="Ignored by ALM (anchor is ALM-managed); kept for CLI compat")
    parser.add_argument("--w_planarity", type=float, default=10.0,
                        help="Ignored by ALM (planarity is ALM-managed); kept for CLI compat")
    parser.add_argument("--w_diag_planarity", type=float, default=0.0,
                        help="Ignored by ALM; kept for CLI compat")
    parser.add_argument("--k_xy", type=float, default=1.0)
    parser.add_argument("--k_z", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_loss", action="store_true")

    # ALM-specific arguments
    parser.add_argument("--rho_init", type=float, default=0.1,
                        help="Initial penalty coefficient ρ₀")
    parser.add_argument("--rho_max", type=float, default=100.0,
                        help="Maximum ρ (clamp to prevent ill-conditioning)")
    parser.add_argument("--rho_scale", type=float, default=1.05,
                        help="Multiplicative growth of ρ each dual update")
    parser.add_argument("--dual_update_interval", type=int, default=50,
                        help="Primal steps between each dual update")
    parser.add_argument("--lambda_planarity_init", type=float, default=0.0,
                        help="Initial λ for planarity constraint")
    parser.add_argument("--lambda_anchor_init", type=float, default=0.0,
                        help="Initial λ for anchor constraint")

    args = parser.parse_args()

    if args.output_dir is None:
        parts = ["checkpoints", "alm"]
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
        parts.append(f"rho{args.rho_init}")
        args.output_dir = os.path.join(*parts)
    print(f"Output directory: {args.output_dir}")

    train_alm(args)

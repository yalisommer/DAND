# Constraint Enforcement Strategies

This project trains an MLP decoder that maps a low-dimensional latent space to
physically-valid quad-mesh deformations. The central challenge is enforcing
**geometric constraints** (planarity, anchor, etc.) during training. We
implement three approaches that all share the same mesh infrastructure,
`SubspaceDecoder` architecture, and residual seeded formulation — only the
constraint enforcement strategy differs.

---

## 1. Baseline (`train.py`) — Fixed Penalty Weights

### How it works

Each constraint is turned into a squared penalty term with a **hand-tuned
scalar weight** added to the loss:

```
L = E_pot + w_planarity · c_plan(q)² + w_anchor · c_anchor(q)² + λ · L_metric
```

The optimizer minimizes this single combined objective. Constraint satisfaction
is approximate: the residual violation at convergence is proportional to
`1/w` — a larger weight pushes the violation closer to zero but also makes the
loss landscape stiffer and harder to optimize.

### Strengths

- **Simple**: one scalar per constraint, easy to implement and reason about.
- **Predictable**: you can directly control how much the optimizer cares about
  each term by adjusting its weight.
- **Fast per step**: no dual updates, no feasibility checks, just forward +
  backward on a single loss.

### Weaknesses

- **Requires manual tuning**: the correct `w_planarity` depends on the mesh, the
  energy scale, and the other weights. A value that works for a 25-vertex grid
  may be wildly wrong for a 500-vertex vase.
- **No convergence guarantee**: there is no mechanism that drives the constraint
  violation to exactly zero. You always trade off energy quality vs. constraint
  satisfaction.
- **Weight coupling**: changing one weight shifts the balance of all others. A
  sweep over `w_planarity` also implicitly changes the relative importance of
  the metric-preserving regularizer.

### Key CLI flags

| Flag | Role |
|------|------|
| `--w_planarity` | Planarity penalty weight |
| `--w_diag_planarity` | Diagonal planarity weight |
| `--w_anchor` | Anchor loss weight |
| `--lam` | Metric-preserving regularizer weight |
| `--k_xy`, `--k_z` | Displacement spring stiffness (stiffFree3d) |

---

## 2. ALM (`train_alm.py`) — Augmented Lagrangian Method

### How it works

Instead of fixed penalty weights, each constraint gets a **Lagrange multiplier**
`λ_i` and a **penalty coefficient** `ρ_i` that adapt during training:

```
L_ALM = E_pot + λ_plan · c_plan + (ρ_plan/2) · c_plan² + λ_anchor · c_anchor + (ρ_anchor/2) · c_anchor² + λ_metric · L_metric
```

Training alternates between:

1. **Primal steps** (standard gradient descent on `L_ALM` w.r.t. network
   weights θ).
2. **Dual updates** (every `dual_update_interval` steps, outside the
   optimizer):
   ```
   λ_i ← λ_i + ρ_i · c̄_i       (c̄ = mean violation over last batch)
   ρ_i ← adaptive via AdaGrad    (prevents any single constraint from dominating)
   ```

This is classical dual ascent: the primal minimizes `L_ALM`, the dual maximizes
it. The fixed point satisfies the KKT conditions of the constrained problem —
meaning constraints are driven to **exactly zero** at convergence, not merely
to `O(1/w)`.

### Why adaptive ρ (AdaGrad-style)?

A single global ρ can blow up if one constraint is much harder than another.
Following Basir & Senocak (2023), we maintain per-constraint accumulators:

```
G_i += c_i²
ρ_i = ρ_init / (√G_i + eps)
```

Constraints that are already well-satisfied (small `c_i`) accumulate slowly and
keep their ρ large (strong enforcement). Constraints that fluctuate
(large `c_i`) accumulate quickly and have their ρ reduced (prevents
oscillation).

### Constraints managed by ALM

| Dual variable | Constraint `c(q)` | Replaces baseline's |
|---|---|---|
| `λ_planarity` | `compute_planarity_energy_torch(verts, faces)` | `w_planarity` |
| `λ_diag` | `compute_diag_planarity_energy_torch(verts, faces)` | `w_diag_planarity` |
| `λ_anchor` | `mean(MLP(0)²)` | `w_anchor` |

The metric-preserving regularizer (`L_metric`) is **not** converted to ALM. It
is an intrinsic regularizer on the latent map's geometry, not a geometric
constraint with a target violation of zero. It stays as `lam * L_metric`.

### Expected behavior vs. baseline

- **Early training**: λ is small → model is free to explore, same as baseline
  with low weights.
- **Late training**: λ grows until `c_plan ≈ 0` is enforced tightly.
- **Loss curve for planarity**: initial growth then collapse toward zero. In the
  baseline, planarity plateaus at a nonzero residual proportional to
  `1/w_planarity`.
- **No weight tuning**: you set `ρ_init` and `ρ_max` once and the method adapts.
  The same settings should work across different meshes and energy scales.

### Key CLI flags

| Flag | Default | Role |
|------|---------|------|
| `--rho_init` | 0.1 | Initial penalty coefficient ρ₀ |
| `--rho_max` | 100.0 | Maximum ρ (prevents ill-conditioning) |
| `--rho_scale` | 1.05 | Multiplicative growth of ρ each dual update |
| `--dual_update_interval` | 50 | Primal steps between dual updates |
| `--lambda_planarity_init` | 0.0 | Initial λ for planarity |
| `--lambda_anchor_init` | 0.0 | Initial λ for anchor |

### Reference

Basir & Senocak (2023), *"An adaptive augmented Lagrangian method for training
physics and equality constrained ANNs"* — [arxiv.org/abs/2306.04904](https://arxiv.org/abs/2306.04904)

### Risks

- **ρ blow-up**: if `rho_scale` is too aggressive, ρ grows exponentially and
  makes the landscape numerically unstable. The AdaGrad-style adaptation and
  `rho_max` clamp mitigate this.
- **Anchor double-counting**: do not also set `--w_anchor > 0` in ALM mode —
  the anchor constraint is already ALM-managed.

---

## 3. GINN (`train_ginn.py`) — Geometry-Informed Neural Network

### How it works

GINN decomposes the training problem into three explicit roles:

```
min_θ   Objective(f_θ)              — potential energy (minimize)
s.t.    Feasibility(f_θ(z)) ≤ ε    — planarity must be small (hard gate)
        Diversity(f_θ)     ≥ δ      — latent samples must spread out (lower bound)
```

This is structurally different from the baseline and ALM: constraints are not
just penalty terms — they have distinct **roles** in the optimization.

#### Feasibility gate

For each sample in the batch, we check if its planarity violation is below
`feasibility_eps`. Only **feasible** samples contribute to the energy
objective:

```python
feasible_mask = (e_planar_per_sample < feasibility_eps)
if feasible_mask.any():
    loss_energy = e_pot[feasible_mask].mean()
else:
    loss_energy = e_planar_batch.mean()   # fallback: push toward feasibility
```

This means the optimizer only learns to reduce energy on configurations that
already satisfy the constraint. When no configuration is feasible yet (early
training), it falls back to minimizing planarity directly.

A soft planarity penalty (`--w_planarity`) supplements the gate to provide
gradient signal even when all samples are infeasible.

#### Diversity constraint (replaces metric-preserving loss)

The baseline uses a metric-preserving regularizer to prevent the latent space
from collapsing. GINN replaces this with a **formal diversity constraint**:

```
D = mean pairwise L2 distance in output space
Constraint: D ≥ δ (diversity_delta)
```

This is an **inequality** constraint enforced via its own ALM dual variable
`λ_div`:

```
c_div = δ - D             (positive when violated: D < δ)
L_div = λ_div · c_div + (ρ_div/2) · max(0, c_div)²
```

The `max(0, ...)` is key: we only penalize when diversity is **below** the
threshold, not when it exceeds it.

#### Diversity warmup

For the first `--diversity_warmup` steps, the diversity constraint is inactive
(`λ_div = 0`, `ρ_div = 0`). This lets the model first learn to produce
feasible (planar) configurations before being pushed to spread out. Without
warmup, diversity pressure can force the model into non-planar regions before
it knows how to be planar at all.

### Expected behavior vs. baseline and ALM

- **Feasible fraction** should rise from ~0 early in training to ~1.0 at
  convergence. This is the key diagnostic — if it stays at 0, lower
  `feasibility_eps` or increase warmup.
- **Diversity** is a formal guarantee, not a soft regularizer. The optimizer
  cannot sacrifice diversity to reduce energy (unlike the baseline where
  `L_metric` competes with `E_pot` for gradient bandwidth).
- **Energy quality**: only feasible samples count toward energy, so the model
  focuses its capacity on the physically-valid region of configuration space.

### Key CLI flags

| Flag | Default | Role |
|------|---------|------|
| `--feasibility_eps` | 0.01 | Max planarity for a sample to be "feasible" |
| `--diversity_delta` | 0.1 | Minimum required mean pairwise distance |
| `--rho_div_init` | 0.1 | Initial ρ for diversity ALM |
| `--rho_div_max` | 50.0 | Max ρ for diversity ALM |
| `--diversity_warmup` | 5000 | Steps before diversity constraint activates |
| `--w_planarity` | 10.0 | Soft planarity penalty (supplements gate) |
| `--w_anchor` | 10.0 | Anchor loss (fixed weight, not ALM) |

### Reference

Knöbelreiter et al. (2024), *"Geometry-Informed Neural Networks"* —
[arxiv.org/abs/2402.14009](https://arxiv.org/abs/2402.14009)

### Risks

- **Diversity–feasibility conflict**: a high `diversity_delta` can force the
  model to spread into non-planar regions. Keep warmup long enough that the
  model has a well-formed feasible region first. If `feasible_fraction` drops
  after diversity activates, reduce `diversity_delta`.
- **feasibility_eps scale**: this threshold depends on the mesh and the
  absolute scale of the planarity energy. For a 25-vertex grid, 0.01 is
  reasonable; for a 500-vertex mesh with larger faces, you may need 0.1 or
  more.

---

## Side-by-side comparison

| Aspect | Baseline | ALM | GINN |
|--------|----------|-----|------|
| **Constraint enforcement** | Fixed penalty weights | Adaptive Lagrange multipliers | Feasibility gate + ALM diversity |
| **Planarity** | `w · c²` (soft penalty) | `λ·c + (ρ/2)·c²` (dual ascent) | Hard gate (`c < ε`) + soft penalty |
| **Diversity / metric** | `lam · L_metric` (soft reg) | `lam · L_metric` (same as baseline) | ALM inequality (`D ≥ δ`) |
| **Anchor** | `w_anchor · c²` | ALM-managed | Fixed weight |
| **Convergence to c=0** | No (residual ∝ 1/w) | Yes (KKT) | Yes (gating ensures only feasible samples count) |
| **Tuning burden** | High (one weight per constraint) | Low (`ρ_init`, `ρ_max`) | Medium (`ε`, `δ`, warmup) |
| **Per-step cost** | Lowest | +dual update overhead | +per-sample feasibility check |
| **Best for** | Quick experiments, known-good weights | Hands-off constraint satisfaction | Strict feasibility with diversity guarantee |

---

## Benchmark commands (vase mesh, stiffFree3d)

```bash
# Baseline
python train.py \
    --mesh_file Meshes/normalized_vase2_fat.obj \
    --mode stiffFree3d --latent_dim 6 \
    --k_xy 1.0 --k_z 0.5 \
    --num_steps 50000 \
    --output_dir results/baseline_vase --plot_loss

# ALM
python train_alm.py \
    --mesh_file Meshes/normalized_vase2_fat.obj \
    --mode stiffFree3d --latent_dim 6 \
    --k_xy 1.0 --k_z 0.5 \
    --rho_init 0.1 --rho_max 100 --dual_update_interval 50 \
    --num_steps 50000 \
    --output_dir results/alm_vase --plot_loss

# GINN
python train_ginn.py \
    --mesh_file Meshes/normalized_vase2_fat.obj \
    --mode stiffFree3d --latent_dim 6 \
    --k_xy 1.0 --k_z 0.5 \
    --feasibility_eps 0.01 --diversity_delta 0.1 \
    --diversity_warmup 5000 \
    --num_steps 50000 \
    --output_dir results/ginn_vase --plot_loss
```

### What to compare in the results

| Metric | Where | What to look for |
|--------|-------|------------------|
| Final planarity (mean c_plan) | `train_log.json` | ALM/GINN should be lower than baseline |
| Feasible fraction | GINN `train_log.json` | Should reach ≥ 0.95 at convergence |
| Diversity (mean pairwise dist) | GINN `train_log.json` | Should stay above `δ` |
| Sensitivity to weight choice | Sweep `w_planarity` for baseline | ALM/GINN should be robust; baseline varies wildly |
| Training time | `train_log.json` timestamps | ALM has small overhead; GINN has per-sample planarity cost |
| Loss plots | `loss_plot.png` in each output dir | Visual comparison of convergence curves |

# Training Explained

## Overview

This project trains a neural network to learn a **low-dimensional latent space** for deforming a 4×4 quad mesh (25 vertices, 16 quad faces). The approach is **data-free**: there is no dataset of pre-computed deformations. Instead, the network learns by sampling random latent vectors and minimizing a physics-inspired loss function.

Inspired by: *"Data-Free Learning of Reduced-Order Kinematics"*

---

## 1. The Mesh

The mesh is a regular 4×4 grid of quadrilaterals:

```
20──21──22──23──24
│   │   │   │   │
15──16──17──18──19
│   │   │   │   │
10──11──12──13──14
│   │   │   │   │
 5── 6── 7── 8── 9
│   │   │   │   │
 0── 1── 2── 3── 4
```

- **25 vertices** on integer coordinates (0,0) to (4,4), z=0 at rest
- **16 quad faces**
- **9 interior vertices** (6, 7, 8, 11, 12, 13, 16, 17, 18) — these are free in "anchored" mode
- **16 boundary vertices** — fixed in "anchored" mode, free in all other modes
- **56 edges** (40 grid edges + 16 shared internal edges)

This flat grid is the **rest state** (`q_seed`), the configuration the network starts from and the configuration that `z=0` should map to.

---

## 2. Data-Free Training (No Dataset!)

Unlike typical neural networks, there is **no training data, no validation set, and no epochs**. Instead:

1. Each training **step**, we sample a batch of random latent vectors `z ~ N(0, I)` from a standard normal distribution.
2. The decoder network maps each `z` to a mesh configuration (vertex positions).
3. We compute a loss that measures how "good" each configuration is.
4. We backpropagate and update the network weights.

This is repeated for `num_steps` iterations (default: 50,000). Every step uses fresh random samples — the network never sees the same `z` twice. There are no epochs because there is no finite dataset to iterate over.

**The key insight**: the loss function encodes the physics. The network discovers low-energy deformations by exploring the space of possible configurations, guided by the energy landscape.

---

## 3. The Decoder (MLP)

The decoder is a Multi-Layer Perceptron (MLP) called `SubspaceDecoder`:

```
z ∈ R^d  →  [Linear → ELU] × num_layers  →  Linear  →  displacement ∈ R^output_dim
```

- **Input**: latent vector `z` (dimension `d`, typically 3 or 6)
- **Output**: vertex displacement from the rest state
- **Architecture**: 5 hidden layers, 64 units each, ELU activations (following the paper)
- **Same architecture for all modes** — only the output dimension changes

The final vertex positions are computed via the **residual formulation**:

```
f(z) = q_seed + MLP(z)
```

where `q_seed` is the rest configuration (flat grid). The MLP learns *displacements* from rest, not absolute positions. This is critical: it means `MLP(0) ≈ 0` maps to the rest state (enforced by the anchor loss, see below).

---

## 4. Seeded Training Schedule (ρ ramp)

To prevent the network from outputting wild, high-energy configurations at the start, we use a **seeded schedule**:

```
f(z) = q_seed + ρ · MLP(z)
```

- `ρ` linearly increases from **0 to 1** over the course of training (`ρ = step / num_steps`)
- At `ρ=0` (step 0): output is always `q_seed` regardless of `z`
- At `ρ=1` (final step): output is `q_seed + MLP(z)` (full decoder)

This gradually "grows" the deformation space outward from the rest state, ensuring the network always explores near low-energy configurations rather than jumping to chaotic high-energy states.

The effective scale `σ_eff = σ · ρ` used in the metric regularizer also ramps, so that regularizer starts gentle and tightens as the subspace expands.

---

## 5. Loss Function

The total loss at each step is:

```
L = L_energy + λ · L_metric + w_anchor · L_anchor
```

Each component is described below.

### 5.1. Potential Energy (`L_energy`)

Measures how physically plausible each decoded configuration is. Averaged over the batch:

```
L_energy = mean over batch of E_pot(f(z))
```

The specific terms in `E_pot` depend on the training mode (see Section 6), but the building blocks are:

| Term | Formula | What it penalizes | Default weight |
|------|---------|-------------------|----------------|
| **Edge energy** | `Σ (‖vᵢ - vⱼ‖ - L₀)²` | Edges that stretch or compress vs. rest lengths | `w_edge = 1.0` |
| **Area energy** | `Σ (A_quad - A₀)²` | Quads whose area deviates from rest area (1.0) | `w_area = 0.5` |
| **Planarity** | `Σ [vol(v₀,v₁,v₂,v₃)]²` | Non-coplanar quad faces (bending) | `w_planarity` (varies) |
| **XY displacement** | `k_xy · Σ(Δxᵢ² + Δyᵢ²)` | In-plane movement from rest position | `k_xy` (varies) |
| **Z displacement** | `k_z · Σ zᵢ²` | Out-of-plane movement (z-coordinate) | `k_z` (varies) |

**Edge energy** acts like springs on every mesh edge — it resists stretching and compression. This is the primary structural integrity term.

**Area energy** penalizes quad faces that change area, preventing severe shearing or collapse even if edge lengths are preserved.

**Planarity energy** measures the signed volume of the tetrahedron formed by each quad's 4 vertices: `vol = (v₃-v₀) · ((v₁-v₀) × (v₂-v₀))`. For a perfectly flat quad, this volume is zero. Squaring penalizes any out-of-plane bending. Used only in `free3d` mode.

**Displacement stiffness** (`k_xy`, `k_z`) acts like springs tying each vertex to its rest position. Used only in `stiffFree3d` mode. By using different stiffness values for in-plane vs. out-of-plane, the energy landscape can be shaped to prefer certain types of deformation.

### 5.2. Metric-Preserving Regularizer (`L_metric`)

Ensures the latent space is well-structured — prevents collapse and maintains consistent spacing:

```
L_metric = mean over pairs of [ log( ‖f(z) - f(z')‖ / (σ_eff · ‖z - z'‖) ) ]²
```

This pushes the ratio `‖f(z) - f(z')‖ / (σ · ‖z - z'‖)` toward 1 for all pairs of latent samples in the batch. In other words, the decoder should be an approximate **isometry** (distance-preserving map) from latent space to configuration space.

- If the ratio < 1: the decoder is *collapsing* (different z's map to similar configs) → penalized
- If the ratio > 1: the decoder is *stretching* (small z changes cause huge deformations) → penalized
- The **squared log** ensures both directions are equally penalized (log makes the penalty symmetric around ratio=1)

Without this term, the network could satisfy the energy loss by mapping all `z` values to the same low-energy configuration (collapse) or by only using a tiny corner of the latent space.

**Parameters**:
- `σ` (`--sigma`, default 0.5): target Lipschitz constant — how much distance in configuration space corresponds to unit distance in latent space. Larger σ → larger deformations for the same latent vector magnitude.
- `λ` (`--lam`, default 1.0): weight of metric loss relative to energy loss

### 5.3. Anchor Loss (`L_anchor`)

Pins the origin of latent space to the rest configuration:

```
L_anchor = mean( MLP(0)² )
```

Without this, `MLP(0)` could drift to any value, so `z=0` wouldn't map to the flat grid. The anchor loss forces `MLP(0) ≈ 0`, ensuring `f(0) = q_seed + 0 = q_seed`.

This is evaluated every step with a dedicated forward pass of `z = [0, 0, ..., 0]`.

- `w_anchor` (`--w_anchor`, default 10.0): how strongly the origin is pinned. A value of 10.0 means the anchor is weighted 10× compared to the energy loss, ensuring the origin stays firmly at rest.

---

## 6. Training Modes

Each mode controls which vertices are free, how many dimensions the output has, and which energy terms are used.

### 6.1. `anchored` — Boundary Fixed, 2D

| Property | Value |
|----------|-------|
| Free vertices | 9 interior only |
| Output dim | 18 (9 vertices × 2 coordinates) |
| DOF | 18 |
| Energy terms | Edge + Area |

The 16 boundary vertices are locked in place. The MLP only predicts (x,y) displacements for the 9 interior vertices. This is the simplest mode and a good baseline — 6 latent dims captures most of the 18 DOF.

### 6.2. `free` — All Vertices, 2D

| Property | Value |
|----------|-------|
| Free vertices | All 25 |
| Output dim | 50 (25 vertices × 2 coordinates) |
| DOF | 50 |
| Energy terms | Edge + Area |

All vertices can move in the xy-plane. More expressive than anchored but harder to compress into few latent dims since the full DOF is 50.

### 6.3. `free3d` — All Vertices, 3D with Planarity Penalty

| Property | Value |
|----------|-------|
| Free vertices | All 25 |
| Output dim | 75 (25 vertices × 3 coordinates) |
| DOF | 75 |
| Energy terms | Edge + Area + **Planarity** |

Vertices can move in all 3 dimensions. The **planarity penalty** (`w_planarity`) controls how much bending is allowed:

- `w_planarity = 10.0` → very flat mesh (little bending allowed)
- `w_planarity = 1.0` → moderate bending
- `w_planarity = 0.2` → lots of bending freedom

Checkpoint path: `checkpoints/free3d/d{dim}/wp{w_planarity}/model.pt`

### 6.4. `stiffFree3d` — All Vertices, 3D with Anisotropic Stiffness

| Property | Value |
|----------|-------|
| Free vertices | All 25 |
| Output dim | 75 (25 vertices × 3 coordinates) |
| DOF | 75 |
| Energy terms | Edge + Area + **k_xy · Σ(Δxy²)** + **k_z · Σ(z²)** |

This mode replaces the volumetric planarity penalty with **displacement springs** that tie each vertex to its rest position. Critically, in-plane (xy) and out-of-plane (z) springs have **separate stiffness values**:

```
E_displacement = k_xy · Σ(Δxᵢ² + Δyᵢ²) + k_z · Σ(zᵢ²)
```

The **ratio `k_z / k_xy`** determines bending preference:
- `k_z < k_xy` (e.g., k_xy=1.0, k_z=0.1) → z-displacement is cheap relative to xy → model discovers bending modes
- `k_z = k_xy` (e.g., both 1.0) → isotropic, no preference
- `k_z > k_xy` (e.g., k_xy=1.0, k_z=5.0) → z-displacement is expensive → model stays flat

**Why both xy and z are penalized**: If only z were penalized, the metric loss would always prefer xy-motion (which costs 0 displacement penalty) over z-motion (which costs k_z). By also penalizing xy, both directions have a cost, and the *relative* cost (the ratio) determines which the model prefers.

Checkpoint path: `checkpoints/stiffFree3d/d{dim}/kxy{k_xy}_kz{k_z}/model.pt`

---

## 7. Optimizer and Learning Rate Schedule

- **Optimizer**: Adam with learning rate `1e-4`
- **LR schedule**: Step decay — LR is halved every 12,500 steps
- Over 50,000 steps, the LR goes through 4 halvings: `1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6`

---

## 8. Hyperparameters Summary

### Global (all modes)

| Parameter | CLI flag | Default | Role |
|-----------|----------|---------|------|
| Latent dim | `--latent_dim` | 6 | Size of the latent space |
| Hidden dim | `--hidden_dim` | 64 | Width of MLP hidden layers |
| Num layers | `--num_layers` | 5 | Depth of MLP |
| Num steps | `--num_steps` | 50000 | Total training iterations |
| Batch size | `--batch_size` | 32 | Latent samples per step |
| Learning rate | `--lr` | 1e-4 | Adam optimizer LR |
| LR decay steps | `--lr_decay_steps` | 12500 | Halve LR every N steps |
| σ | `--sigma` | 0.5 | Target subspace scale (Lipschitz constant) |
| λ | `--lam` | 1.0 | Metric loss weight |
| w_anchor | `--w_anchor` | 10.0 | Anchor loss weight |

### Mode-specific

| Parameter | CLI flag | Default | Applies to |
|-----------|----------|---------|------------|
| w_planarity | `--w_planarity` | 10.0 | `free3d` only |
| k_xy | `--k_xy` | 1.0 | `stiffFree3d` only |
| k_z | `--k_z` | 1.0 | `stiffFree3d` only |

---

## 9. Checkpoint Structure

Models auto-save to organized directories:

```
checkpoints/
├── anchored/
│   └── d6/
│       └── model.pt
├── free/
│   └── d6/
│       └── model.pt
├── free3d/
│   ├── d3/
│   │   └── wp0.2/
│   │       └── model.pt
│   └── d6/
│       ├── wp0.2/
│       ├── wp1.0/
│       └── wp10.0/
│           └── model.pt
└── stiffFree3d/
    └── d6/
        ├── kxy1.0_kz0.1/
        ├── kxy1.0_kz1.0/
        └── kxy1.0_kz5.0/
            └── model.pt
```

Each checkpoint (`model.pt`) contains:
- `model_state_dict` — network weights
- `q_seed` — rest state vertex positions (for residual decoding)
- `latent_dim`, `output_dim`, `hidden_dim`, `num_layers` — architecture
- `mode`, `sigma`, `lam`, `w_anchor` — training config
- `w_planarity` (free3d) or `k_xy`, `k_z` (stiffFree3d) — mode-specific params

A `train_log.json` is also saved alongside each model with per-step loss breakdowns.

---

## 10. Example Commands

```bash
# Train anchored mesh (boundary fixed, 18 DOF → 6 latent dims)
python3 train.py --mode anchored --latent_dim 6

# Train free 2D mesh (all verts, 50 DOF → 6 latent dims)
python3 train.py --mode free --latent_dim 6

# Train free 3D mesh with moderate bending
python3 train.py --mode free3d --latent_dim 6 --w_planarity 1.0

# Train stiff 3D mesh — bendy (low k_z/k_xy ratio)
python3 train.py --mode stiffFree3d --latent_dim 6 --k_xy 1.0 --k_z 0.1

# Train stiff 3D mesh — rigid (high k_z/k_xy ratio)
python3 train.py --mode stiffFree3d --latent_dim 6 --k_xy 1.0 --k_z 5.0

# Retrain all 9 models (3 batches of 3 in parallel)
bash retrain_all.sh

# Visualize a trained model
python3 latent_viz.py --model checkpoints/stiffFree3d/d6/kxy1.0_kz0.1/model.pt
```

---

## 11. Intuition: Why This Works

The three loss terms create a balanced tension:

1. **Energy loss** says: "find configurations that don't violate physics" (low edge stretch, area preservation, stiffness constraints)
2. **Metric loss** says: "use the full latent space — don't collapse everything to one point, and don't have tiny z-changes cause huge jumps"
3. **Anchor loss** says: "the origin of latent space should be the flat rest mesh"

The network resolves this tension by learning an organized latent space where:
- Each latent direction corresponds to a smooth, physically plausible deformation mode
- Moving along one latent axis produces consistent, predictable changes
- The magnitude of the latent vector roughly corresponds to the magnitude of deformation
- The origin is always the undeformed mesh

The ρ ramp ensures stable convergence by starting from the known-good rest state and gradually expanding the deformation space, rather than trying to learn everything at once.

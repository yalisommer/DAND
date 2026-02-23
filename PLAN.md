# Planar Quad Toy — Project Plan

A toy implementation inspired by *Data-Free Learning of Reduced-Order Kinematics* (Sharp et al.), applied to a simple 2D planar quadrilateral mesh.

---

## 1. Problem Setup

### 1.1 Mesh Definition
- **Topology**: A 4×4 grid of planar quadrilaterals → 5×5 = **25 vertices**, 16 quad faces.
- **Initial (seed) configuration**: A flat, regular grid of unit squares. Vertices at integer coordinates `(i, j)` for `i, j ∈ {0, 1, 2, 3, 4}`.
- **Boundary conditions**: All boundary (exterior) vertices are **fixed**. This leaves a 3×3 = **9 interior vertices** free to move, for a total of **18 degrees of freedom** (9 vertices × 2D coordinates).
- **Planar constraint**: All motion is strictly 2D (the z-coordinate is always 0). In a later extension, this will be relaxed to near-planar (small ε bending into 3D).

### 1.2 Vertex & Face Indexing
```
Vertices (5×5 grid):          Faces (4×4 grid of quads):
20 - 21 - 22 - 23 - 24        Q12  Q13  Q14  Q15
|    |    |    |    |          
15 - 16 - 17 - 18 - 19        Q8   Q9   Q10  Q11
|    |    |    |    |          
10 - 11 - 12 - 13 - 14        Q4   Q5   Q6   Q7
|    |    |    |    |          
 5 -  6 -  7 -  8 -  9        Q0   Q1   Q2   Q3
|    |    |    |    |          
 0 -  1 -  2 -  3 -  4
```

**Interior (free) vertices**: {6, 7, 8, 11, 12, 13, 16, 17, 18} — 9 vertices, 18 DOF.

**Boundary (fixed) vertices**: {0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24} — 16 vertices.

---

## 2. Energy Function

Following the paper's philosophy, we need a differentiable potential energy `E_pot(q)` where `q ∈ R^18` contains the 2D positions of the 9 interior vertices (boundary vertices are constant and stitched in before evaluation).

### 2.1 Energy Terms (Candidate)
We combine several terms to define physically meaningful low-energy configurations:

1. **Edge-length preservation (spring energy)**:
   ```
   E_edge = Σ_edges (||v_i - v_j|| - L_0)^2
   ```
   Where `L_0 = 1.0` is the rest length (unit squares). Summed over all edges (both interior and boundary-to-interior). This penalizes stretching/compression.

2. **Quad area preservation** *(optional)*:
   ```
   E_area = Σ_quads (A_q - A_0)^2
   ```
   Where `A_0 = 1.0` is the rest area. Prevents quads from collapsing to zero area or inverting.

3. **Diagonal (shear) springs** *(optional)*:
   ```
   E_diag = Σ_quads (||d1|| - D_0)^2 + (||d2|| - D_0)^2
   ```
   Where `d1, d2` are the two diagonals of each quad and `D_0 = √2` is the rest diagonal. Resists shearing.

### 2.2 Energy Weighting
The total energy is a weighted sum:
```
E_pot(q) = w_edge * E_edge + w_area * E_area + w_diag * E_diag
```
Weights to be tuned; start with `w_edge = 1.0` as the primary term. The other terms can be added as needed to get good behavior.

### 2.3 Key Properties
- Fully differentiable via PyTorch autograd.
- The seed configuration (regular grid) is the global minimum of this energy (E_pot = 0).
- The energy landscape should be smooth enough for gradient-based training.

---

## 3. Neural Subspace Model

### 3.1 Architecture
Following the paper: a simple **MLP (Multi-Layer Perceptron)** decoder:

```
f_θ : R^d → R^18
```

Maps a latent vector `z ∈ R^d` to the 2D positions of the 9 interior vertices (the 18 DOF).

- **Latent dimension (d)**: Start with `d = 18` (matching the full DOF count) to allow maximum expressivity. Later, reduce to find the effective dimensionality (e.g., d = 9, 4, 2).
- **Hidden layers**: 5 layers, width 64 (our system is small; the paper uses 64–128).
- **Activations**: ELU (following the paper).
- **Output**: Raw 2D coordinates for interior vertices. Boundary vertices are appended as constants before energy evaluation.

### 3.2 Seeded Training (from the paper, Section 4.1)
The critical training trick — the **seed blending schedule**:

```
f_θ(z) = ρ · MLP_θ(z) + (1 - ρ) · q_seed
```

- `q_seed` = the flat regular grid positions (the rest state).
- `ρ` **linearly increases from 0 → 1** over training.
  - At ρ = 0: output is always `q_seed`, regardless of z. The MLP has no influence.
  - At ρ = 1: output is purely the MLP. The seed is fully removed.
- **Intuition**: The subspace "grows outward" from the known-good seed state. Early training stays near the seed, late training explores freely.

### 3.3 Loss Function (Equation 4 from the paper)
```
L(θ) = E_{z,z' ~ N(0,I)} [ E_pot(f_θ(z)) + λ · log( |f_θ(z) - f_θ(z')|_M / (ρσ |z - z'|) ) ]
```

Two terms:
1. **Potential energy** `E_pot(f_θ(z))`: Encourages decoded configurations to be low-energy.
2. **Metric-preserving regularizer**: Prevents the map from collapsing (different z's should produce different configurations). Without this, the network would learn to always output `q_seed`.

**Hyperparameters**:
- `λ`: Regularizer weight. Controls trade-off between low energy and subspace diversity. Start with `λ = 1.0`.
- `σ`: Subspace scale. Controls how far from the seed the subspace can reach. Start large, tune down per paper's guidance. Also scaled by ρ during training.
- `M`: Mass matrix. For simplicity, can use the identity matrix (uniform mass).

### 3.4 Training Details
- **Optimizer**: Adam, lr = 1e-4.
- **Batch size**: 32 latent samples per step.
- **Training steps**: Start with ~100k (our system is tiny vs. the paper's 1M).
- **LR decay**: Halve every 25k steps (scaled from paper's 250k/1M schedule).
- **ρ schedule**: Linear 0 → 1 over full training.
- **Sampling**: `z ~ N(0, I)` for each batch; pairwise metric term computed over all pairs in the batch.

---

## 4. Visualization (Polyscope)

### 4.1 Real-Time Interactive Viewer
Using **Polyscope** (Python bindings) for mesh visualization with interactive controls:

- **Mesh display**: Register the 4×4 quad mesh. Update vertex positions in real-time.
- **Latent sliders**: One slider per latent dimension (d sliders). Moving a slider changes the corresponding component of `z`, runs the MLP forward pass, and updates the mesh.
- **Color coding**: Optionally color quads by local energy or strain to show which regions are stressed.
- **Boundary highlighting**: Visually distinguish fixed boundary vertices from free interior ones.

### 4.2 Implementation
Polyscope's imgui integration supports custom sliders/callbacks:
```python
import polyscope as ps
ps.init()
mesh = ps.register_surface_mesh("quad_grid", vertices, faces)
# In callback: read slider values → decode z → update mesh positions
ps.set_user_callback(callback_fn)
ps.show()
```

---

## 5. Implementation Phases

### Phase 1: Mesh & Energy Foundation
- [ ] Define the 4×4 quad mesh: vertices, faces, boundary/interior masks.
- [ ] Implement the energy function(s) in PyTorch (differentiable).
- [ ] Verify energy is 0 at seed config, positive elsewhere.
- [ ] Basic Polyscope visualization of the mesh.

### Phase 2: Neural Subspace Model
- [ ] Implement the MLP decoder (z → interior vertex positions).
- [ ] Implement seeded training with ρ schedule.
- [ ] Implement the full loss (energy + metric-preserving regularizer).
- [ ] Training loop with Adam optimizer.

### Phase 3: Interactive Visualization
- [ ] Polyscope viewer with latent sliders.
- [ ] Real-time decode & mesh update.
- [ ] Energy display (total energy readout, per-quad coloring).
- [ ] Controls for latent dimension count (retrain with different d).

### Phase 4: Experiments & Extensions
- [ ] Sweep latent dimensions: d = 18, 9, 4, 2 — observe quality.
- [ ] Tune hyperparameters (λ, σ) for good subspace coverage.
- [ ] Relax planarity: allow small ε out-of-plane bending (2D → 3D).
- [ ] Explore conditional subspaces (e.g., condition on material stiffness).

---

## 6. Tech Stack
| Component         | Choice          |
|-------------------|-----------------|
| Language          | Python          |
| ML Framework      | PyTorch         |
| Visualization     | Polyscope       |
| Architecture      | MLP (5×64, ELU) |
| Optimizer         | Adam            |

---

## 7. Open Questions / Decisions to Revisit
1. **Energy function details**: Start with edge-length springs. Add area/shear terms if needed for better behavior. Once the paper's approach is working, we can refine.
2. **Latent dim strategy**: Start at d = 18 (full DOF). The interesting question is how small d can get while still producing diverse, meaningful deformations.
3. **Mass matrix M**: Use identity for simplicity, or a proper lumped mass matrix? The paper uses a proper mass matrix; for our toy example, identity is likely fine.
4. **σ tuning**: The paper recommends starting large, observing when configs look good during training, and scaling down. We'll do the same.
5. **Planarity relaxation (future)**: When moving to 3D, the DOF jumps to 27 (9 vertices × 3D). The planarity constraint becomes a penalty term rather than a hard constraint.

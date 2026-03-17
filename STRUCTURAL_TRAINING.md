### Structural3D Training Loss

This document describes the loss used when `--mode structural3d` in `train.py`.

The decoder \(f_\theta(z)\) produces full 3D deformations of a mesh:

- **Latent**: \(z \in \mathbb{R}^d\)
- **Output**: \(f(z) \in \mathbb{R}^{3N}\) (positions of \(N\) vertices)
- **Seeded form** (internally): \(f(z) = q_{\text{seed}} + \rho \cdot \text{MLP}(z)\) with a scalar schedule \(\rho \in [0,1]\) that ramps from 0→1 during training.

For `structural3d` the total loss is:

\[
L = E_{\text{pot}} 
  + L_{\text{trivial}}
  + L_{\text{diversity}}
  + \lambda \, L_{\text{metric}}
  + w_{\text{anchor}} \, L_{\text{anchor}}
\]

where each term is:

---

### 1. Structural Potential Energy \(E_{\text{pot}}\)

Implemented by `potential_energy_structural3d` and logged as `loss_energy`:

- **Inputs**: current verts \(f(z)\), rest verts, topology (edges, faces, face adjacency), and hyperparameters:
  - `w_planarity`
  - `w_anticollapse`, `anticollapse_eps`
  - `w_dirichlet`
  - `w_edge`
  - `w_area`

- **Components**:

  - **Quad planarity** (`w_planarity`):
    - Same volumetric planarity energy as in the baseline model.
    - Penalizes warped quads: each face is close to planar.

  - **Anti-collapse** (`w_anticollapse`, `anticollapse_eps`):
    - Area-based term to discourage vanishing / inverted elements.
    - `anticollapse_eps` is an area threshold; faces whose area drops below this are penalized strongly.
    - If `anticollapse_eps <= 0`, it is **auto-set** to 15% of the mean rest face area (printed at startup).

  - **Dirichlet / smoothness** (`w_dirichlet`):
    - Uses `get_face_adjacency` to penalize large differences between neighboring face transformations.
    - Encourages smooth bending and avoids “tearing” between faces.
    - Kept relatively weak by default so folds and localized features are still allowed.

  - **Edge + area anchors** (`w_edge`, `w_area`):
    - Optional, typically small or zero.
    - Edge term: penalizes changes in rest edge lengths.
    - Area term: penalizes changes in rest face areas.
    - Behave like weak “material anchors” that resist large metric distortion without dominating.

Overall, \(E_{\text{pot}}\) is a physics-inspired structural energy: it prefers deformations that respect planarity, avoid collapse, and remain roughly metric-preserving, while still allowing non-trivial bending.

---

### 2. Anti-triviality \(L_{\text{trivial}}\)

Function: `anti_triviality_loss(f_z, q_seed, z, rho, eps=trivial_eps)`, weighted by `w_trivial` and logged as `loss_trivial`.

**Goal**: prevent the network from mapping large \(\|z\|\) vectors to near-rest configurations. Large latent norms should correspond to meaningful shape changes.

**Computation**:

1. Reshape \(f(z)\) and \(q_{\text{seed}}\) to \((B, N, 3)\) and \((N, 3)\).
2. Remove per-sample translation by subtracting centroids.
3. Compute a rest-state RMS scale from `q_seed` (centroid-removed).
4. Form a translation- and scale-invariant displacement:
   \[
   d_{\text{shape}} = f_{\text{centered}} - q_{\text{centered}}
   \]
5. Compare \(\|z\|^2\) against the normalised displacement magnitude; penalize when \(\|z\|\) is large but the shape displacement is still small.

This term is **gated by \(\rho^2\)** (the same schedule used in the seeded forward), so it is effectively inactive during warmup and only turns on once the model has started learning non-trivial deformations.

---

### 3. Axis Diversity \(L_{\text{diversity}}\)

Function: `axis_diversity_loss(model, q_seed, rho, latent_dim, alpha, disp_mask)`, weighted by `w_diversity` and logged as `loss_diversity`.

**Goal**: ensure that different latent axes produce **different** deformations, not the same motion reparameterized.

**How it works**:

- For each latent axis \(k \in \{0, \dots, d-1\}\), probe:
  \[
  z_k = \alpha e_k,\quad f_k = f(z_k)
  \]
- As in anti-triviality, remove translation and scale from \(f_k\) and `q_seed` to get a shape displacement vector \(d_k\).
- Normalize each \(d_k\) and compute cosine similarities \(\cos(d_i, d_j)\) for all \(i \ne j\).
- Penalize squared cosine similarities:
  \[
  L_{\text{diversity}} \propto \sum_{i \ne j} \cos^2(d_i, d_j)
  \]

Interpretation:

- If two axes produce nearly the same deformation, \(\cos(d_i, d_j) \approx 1\) ⇒ high penalty.
- If each axis produces an orthogonal “direction of shape change”, cosines are near 0 ⇒ low penalty.

`diversity_alpha` controls the probe magnitude \(\alpha\). It should be on the order of the sampling σ so probes land in the part of latent space where the model actually deforms the mesh.

On structural3d runs, this term is **visible in the plots** as the “axis diversity” curve and in `train_log.json` as `loss_diversity`.

---

### 4. Metric-preserving Regularizer \(L_{\text{metric}}\)

Function: `metric_preserving_loss(f_z, z, sigma_rho)`, weighted by `lam` and logged as `loss_metric`.

**Goal**: encourage a well-behaved latent geometry, where distances in latent space correspond (up to scaling) to distances in deformation space.

**Idea**:

- For all pairs \((z_i, z_j)\) in the batch, compare:
  - Latent distance \(\|z_i - z_j\|\)
  - Output distance \(\|f(z_i) - f(z_j)\|\)
- Penalize the squared log-ratio of these distances after scaling by `sigma_rho`:
  \[
  \log\left( \frac{\|f(z_i) - f(z_j)\|}{\sigma_\rho \|z_i - z_j\|} \right)^2
  \]

This discourages both collapsing directions (small changes in \(z\) leading to huge changes in shape) and dead directions (large latent distances with tiny output change).

---

### 5. Anchor Loss \(L_{\text{anchor}}\)

Weighted by `w_anchor` and logged as `loss_anchor`.

**Goal**: keep the rest configuration close to `z = 0`.

- Evaluate the MLP at \(z = 0\): `mlp_at_zero = model(0)`.
- Optionally apply `disp_mask` to zero-out frozen DOFs (e.g. box bottom-lip z).
- Penalize \(\| \text{MLP}(0) \|^2\).

This simple quadratic term ensures that the learned manifold is centered around the rest shape, so sampling near the origin produces small deformations.

---

### Logging and Plots

For `structural3d` runs with `--plot_loss`, `train.py` logs the following per `log_interval`:

- `loss`              — total loss
- `loss_energy`       — structural potential energy \(E_{\text{pot}}\)
- `loss_trivial`      — anti-triviality term
- `loss_diversity`    — axis diversity term
- `loss_metric`       — metric-preserving term
- `loss_anchor`       — anchor loss at \(z = 0\)

At the end of training, it writes:

- `train_log.json` containing these fields over time
- `loss_plot.png` showing **all six** curves on one figure:
  - total loss
  - energy
  - anti-triviality
  - axis diversity
  - metric
  - anchor

This makes it easy to diagnose:

- Whether the structural energy is converging.
- Whether anti-triviality and axis diversity are actually active.
- How strong the metric and anchor terms are relative to the rest.

---

### Recommended Structural3D Hyperparameters

Empirically reasonable starting point (matching recent experiments on `shell_m2`):

- `--w_planarity 50`
- `--w_anticollapse 10`
- `--w_dirichlet 0.05`
- `--w_trivial 1.0`
- `--w_diversity 1.0`
- `--lam 1.0`
- `--w_anchor 10.0`
- `--anticollapse_eps -1` (auto-set to 15% of mean rest area)
- Optionally small `--w_edge` / `--w_area` (e.g. 0.05) if you want weak extra metric anchoring.

These give:

- **E_pot**: structural fidelity (no collapsed faces, reasonably planar quads, smooth bending).
- **L_trivial**: large \(\|z\|\) always lead to non-trivial deformations.
- **L_diversity**: each latent axis controls a distinct, interpretable direction.
- **L_metric**: latent geometry roughly reflects deformation geometry.
- **L_anchor**: rest configuration pinned near `z = 0`.


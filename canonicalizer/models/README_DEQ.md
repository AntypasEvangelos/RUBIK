# Deep Equilibrium Model (DEQ) Canonicalizers

This directory contains two DEQ-based canonicalizer implementations that extend the original frozen energy approach.

## Overview

| Model | Scope | Parameters | Use Case |
|-------|-------|------------|----------|
| **FrozenEnergyCanonicalizer** | Global Sim(2) | 4 DOF (ω, σ, vx, vy) | Baseline, planar scenes |
| **DEQCanonicalizer** | Global Sim(2) | 4 DOF (learned energy) | Adaptive computation, learned features |
| **LocalDEQCanonicalizer** | Local Sim(2) field | 4×H×W DOF | Non-planar, depth variation |

---

## 1. Global DEQ Canonicalizer (`deq_canonicalizer.py`)

### What is DEQ?

Deep Equilibrium Models (Bai et al., NeurIPS 2019) solve for a **fixed point** instead of unrolling iterations:

```python
# Standard approach (frozen energy):
for i in range(K):  # Fixed K iterations
    g = g · exp(-α·∇E)

# DEQ approach:
while ||h_θ(g)|| > ε:  # Iterate until convergence
    g = g · exp(h_θ(g))
```

At the fixed point: `h_θ(g*) = 0` (no more updates needed).

### Key Differences from Frozen Energy

| Aspect | Frozen Energy | DEQ |
|--------|---------------|-----|
| **Energy landscape** | Fixed (DINO features) | Learned (update network) |
| **Iterations** | Fixed (e.g., 5) | Adaptive (until convergence) |
| **Memory** | O(K) | O(1) via implicit grad |
| **Convergence** | Not guaranteed | Guaranteed (with proper design) |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEQ CANONICALIZER                        │
│                                                             │
│  1. FEATURE EXTRACTION                                      │
│     F_T = DINO(I_T)  [can be frozen or trainable]         │
│                                                             │
│  2. FIXED POINT ITERATION                                   │
│     Initialize: g_0 = Identity                             │
│                                                             │
│     Repeat until convergence:                              │
│       I_S^(k) = warp(I_S, g_k)                             │
│       F_S^(k) = DINO(I_S^(k))                              │
│       F_cond = CrossAttention(F_S^(k), F_T)                │
│       Δξ = UpdateNetwork(F_cond)  ← LEARNED                │
│       g_{k+1} = g_k · exp(Δξ)                              │
│                                                             │
│       if ||Δξ|| < ε: break  ← ADAPTIVE                     │
│                                                             │
│  3. IMPLICIT DIFFERENTIATION (optional)                     │
│     Backprop via: ∂L/∂θ = -(∂L/∂g*)(I - ∂f/∂g*)^{-1}∂f/∂θ │
│                                                             │
│  Output: g* ∈ Sim(2)                                       │
└─────────────────────────────────────────────────────────────┘
```

### Usage

```python
from canonicalizer.models import DEQCanonicalizer

# Initialize
model = DEQCanonicalizer(
    dino_model='dinov2_vitb14',
    freeze_dino=True,  # or False for end-to-end training
    max_iterations=20,
    convergence_threshold=1e-3
)

# Forward
source, target = load_images()
warped, g_star, trajectory = model(
    source, target,
    return_trajectory=True
)

print(f"Converged in {len(trajectory)} iterations")
```

### Advantages

✅ **Adaptive computation** - Harder cases get more iterations
✅ **Learnable energy** - Can optimize for task-specific objectives
✅ **Memory efficient** - O(1) with implicit differentiation
✅ **Convergence guarantees** - With proper Lipschitz constraints

### Disadvantages

❌ **More complex** - Harder to train than frozen energy
❌ **Slower** - Variable iterations make batching harder
❌ **Requires tuning** - Convergence threshold, max iterations

---

## 2. Local DEQ Canonicalizer (`local_deq_canonicalizer.py`)

### Motivation

The global Sim(2) assumption breaks down for:
- **Non-planar scenes** (buildings, terrain)
- **Depth variation** (foreground/background at different scales)
- **Camera rotation + translation** (different parts undergo different transformations)

### Solution: Spatially-Varying Transformations

Instead of a single `g ∈ Sim(2)`, predict a **field** of transformations:

```
g_field: ℝ² → Sim(2)
g_field(x, y) = local transformation at pixel (x, y)
```

Discretized as a grid: `g_field[i, j] ∈ Sim(2)` for patch (i, j).

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              LOCAL DEQ CANONICALIZER                        │
│                                                             │
│  Input: Source I_S, Target I_T                             │
│                                                             │
│  1. PATCH-BASED FEATURES                                    │
│     F_T = DINO(I_T)  ∈ ℝ^(H_patch × W_patch × D)          │
│                                                             │
│  2. LOCAL UPDATE NETWORK                                    │
│     F_cond = CrossAttention(F_S, F_T)                      │
│     F_spatial = Conv2D(F_cond)  # Spatially process        │
│     Δξ_field = Conv2D(F_spatial)  ∈ ℝ^(4 × H_grid × W_grid)│
│                                                             │
│  3. APPLY LOCAL TRANSFORMATIONS                             │
│     For each grid cell (i, j):                             │
│       g_{i,j}^{k+1} = g_{i,j}^k · exp(Δξ_{i,j})           │
│                                                             │
│  4. CREATE DENSE WARP FIELD                                 │
│     For each pixel (x, y):                                 │
│       - Find which grid cell it belongs to                  │
│       - Apply corresponding g_{i,j}                        │
│       - Optionally interpolate between neighbors           │
│                                                             │
│  Output: g_field ∈ Sim(2)^(H_grid × W_grid)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Choices

**1. Patch Resolution** (e.g., 8×8 grid)
- **Larger grid** → More flexibility, but harder to train
- **Smaller grid** → More constrained, smoother

**2. Interpolation**
- **Nearest neighbor** - Fast, discontinuous
- **Bilinear** - Smooth, but complex with Lie groups
- **Lie algebra interpolation** - Mathematically correct

**3. Regularization**
- **Smoothness** - Encourage neighboring patches to have similar transforms
- **Sparsity** - Prefer identity for most patches
- **Total variation** - Penalize large gradients in transformation field

### Usage

```python
from canonicalizer.models import LocalDEQCanonicalizer

# Initialize
model = LocalDEQCanonicalizer(
    dino_model='dinov2_vitb14',
    patch_resolution=8,  # 8×8 grid of transformations
    max_iterations=20
)

# Forward
warped, g_field = model(source, target)
# g_field: (B, 3, 3, 8, 8) - transformation for each patch

# Extract global approximation
g_global = model.get_global_transformation(g_field)
```

### Visualization

```python
import matplotlib.pyplot as plt

# Visualize transformation field
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Rotation field
rotation_field = extract_rotation(g_field)  # (H_grid, W_grid)
axes[0, 0].imshow(rotation_field, cmap='twilight')
axes[0, 0].set_title('Rotation (degrees)')

# 2. Scale field
scale_field = extract_scale(g_field)
axes[0, 1].imshow(scale_field, cmap='viridis')
axes[0, 1].set_title('Scale')

# 3. Translation X
tx_field = extract_translation_x(g_field)
axes[1, 0].imshow(tx_field, cmap='RdBu')
axes[1, 0].set_title('Translation X')

# 4. Translation Y
ty_field = extract_translation_y(g_field)
axes[1, 1].imshow(ty_field, cmap='RdBu')
axes[1, 1].set_title('Translation Y')

plt.tight_layout()
plt.savefig('transformation_field.png')
```

### When to Use

Use **LocalDEQCanonicalizer** when:
- ✅ Scenes have significant depth variation
- ✅ Camera undergoes complex motion (rotation + translation)
- ✅ Objects at different depths need different alignments
- ✅ Non-planar scenes (buildings, terrain)

Use **DEQCanonicalizer** (global) when:
- ✅ Approximately planar scenes
- ✅ Pure rotation or translation
- ✅ Fast inference needed
- ✅ Limited training data

---

## Training

### Loss Function

Both models use the same **correspondence loss**:

```python
# Generate GT correspondences from depth + pose
pts0, pts1 = generate_correspondences(depth, K, pose)

# Forward
warped, g = model(source, target)

# For global: g is (B, 3, 3)
# For local: need to sample g_field at correspondence locations
loss = ||g · pts0 - pts1||²
```

For local model, sample transformations at keypoint locations:

```python
# For each correspondence (x, y):
#   1. Find grid cell (i, j) containing (x, y)
#   2. Get g_local = g_field[:, :, :, i, j]
#   3. Transform: pt_warped = g_local @ pt
```

### Regularization for Local Model

```python
# Smoothness: Encourage neighboring patches to be similar
def smoothness_loss(g_field):
    # Compute differences between adjacent patches
    diff_h = g_field[:, :, :, 1:, :] - g_field[:, :, :, :-1, :]
    diff_w = g_field[:, :, :, :, 1:] - g_field[:, :, :, :, :-1]

    # In Lie algebra space (more principled)
    xi_field = Sim2.log_field(g_field)  # (B, 4, H, W)
    tv = torch.abs(xi_field[:, :, 1:, :] - xi_field[:, :, :-1, :]).sum()
    tv += torch.abs(xi_field[:, :, :, 1:] - xi_field[:, :, :, :-1]).sum()

    return tv

# Total loss
loss = correspondence_loss + λ_smooth * smoothness_loss(g_field)
```

---

## Performance Considerations

### Computational Cost

| Model | Feature Extraction | Optimization | Total |
|-------|-------------------|--------------|-------|
| Frozen Energy | 1× DINO | K iterations | O(K) |
| DEQ Global | K× DINO | Until convergence | O(K_adaptive) |
| DEQ Local | K× DINO | K × (H_grid × W_grid) updates | O(K × H × W) |

**Trade-off**: Local model is more expensive but handles complex scenes.

### Memory

| Model | Forward | Backward (standard) | Backward (implicit) |
|-------|---------|---------------------|---------------------|
| Frozen Energy | O(K) | O(K) | N/A |
| DEQ Global | O(K) | O(K) | **O(1)** |
| DEQ Local | O(K × H × W) | O(K × H × W) | **O(H × W)** |

**Key advantage of DEQ**: Implicit differentiation gives O(1) memory regardless of iterations.

---

## Future Extensions

### 1. Hierarchical Local Transformations
- Coarse-to-fine refinement
- Global transform → Regional transforms → Local transforms

### 2. Differentiable Sinkhorn for Patch Matching
- Instead of cross-attention, use optimal transport
- Each source patch matches to target patch
- Transformation aligns matched patches

### 3. Neural Descriptor Fields
- Instead of DINO, learn a custom feature extractor
- Train end-to-end for canonicalization task

### 4. Uncertainty Estimation
- Predict confidence for each patch transformation
- Weight correspondences by confidence
- Reject unreliable patches

---

## References

1. **Deep Equilibrium Models**
   Bai et al., "Deep Equilibrium Models", NeurIPS 2019

2. **Implicit Differentiation**
   Gould et al., "Deep Declarative Networks", TPAMI 2021

3. **Lie Group Networks**
   Cohen & Welling, "Group Equivariant CNNs", ICML 2016

4. **Spatial Transformer Networks**
   Jaderberg et al., "Spatial Transformer Networks", NIPS 2015

5. **Deformable Registration**
   Balakrishnan et al., "VoxelMorph", CVPR 2019 (for inspiration on local transformations)

---

## Testing

Run the test suite:

```bash
cd /home/user/RUBIK
python tests/test_deq_models.py
```

This will test:
- Global DEQ forward/backward
- Local DEQ forward/backward
- Comparison with frozen energy
- Convergence behavior

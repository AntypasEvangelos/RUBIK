
# Implementation Details: Frozen Energy Canonicalizer

This document explains the technical implementation of the Conditional Multiview Canonicalizer for the RUBIK benchmark.

## 1. Problem Statement

Standard dense image matchers (like LoFTR, RoMa) struggle with large geometric transformations (viewpoint changes, scale differences). 

**Our Solution:** Pre-process the image pair $(I_S, I_T)$ by finding a transformation $g^* \in \text{Sim}(2)$ that aligns the source image $I_S$ to the target $I_T$ *before* matching.

## 2. Theoretical Foundation

### The Transformation Group: Sim(2)
We model the geometric variation using the 2D Similarity group, which handles:
- **Rotation** ($\theta$)
- **Scale** ($s$)
- **Translation** ($t_x, t_y$)

This is implemented in `core/lie_group.py` using Lie Algebra $\mathfrak{sim}(2)$ for numerical stability. We optimize in the tangent space (Lie algebra) and map to the group via the exponential map:
$$ g = \exp(\xi), \quad \xi \in \mathbb{R}^4 $$

### Energy-Based Formulation
We define the "best" alignment as one that minimizes the distance between feature representations:
$$ E(g) = d(\phi(g \cdot I_S), \phi(I_T)) $$

where $\phi$ is a feature extractor (DINOv2) and $d$ is a distance metric (Cosine/L2).

## 3. Implementation: "Frozen Energy" Approach

We implemented **Approach 1**, which treats the energy landscape as fixed (frozen) and learns to navigate it.

### Architecture (`models/frozen_energy.py`)

1.  **Frozen Backbone (DINOv2):** 
    We use a pre-trained DINOv2 model to extract semantic features. Why DINOv2? It is robust to local deformations and provides high-level semantic correspondence useful for coarse alignment.
    *   *Implementation:* Wrapped in `models/feature_extractor.py`.

2.  **Learned Initialization (Cross-Attention):**
    Gradient descent on complex energy landscapes requires a good starting point. We train a lightweight network to predict an initial transformation $\xi_0$.
    *   *Mechanism:* Source and Target features interact via a Cross-Attention block (`models/cross_attention.py`), followed by an MLP head.
    *   *Why:* This jumps over local minima to a basin of attraction near the correct solution.

3.  **Iterative Refinement (Optimization):**
    From the initial $g_0$, we refine the transformation by taking gradient steps on the feature energy.
    *   *Loop:* Warping $\rightarrow$ Feature Extraction $\rightarrow$ Energy Calc $\rightarrow$ Backward $\rightarrow$ Update.
    *   *Trick:* We use **FOMAML** (First-Order Model-Agnostic Meta-Learning) approximation by detaching gradients in the inner loop to avoid expensive and unstable second-order derivatives through the `grid_sample` operation.

## 4. Training Infrastructure

To train the Initialization Network:
*   **Losses (`training/losses.py`):** We use a supervised **Correspondence Loss**. We warp known source keypoints using predicted $g^*$ and measure distance to target keypoints.
*   **Data (`training/datasets.py`):** Loads benchmarking pairs from `rubik.json`.
*   **Visualization (`training/vis.py`):** Provides real-time visual feedback (Source | Warped | Target) to debug alignment.

## 5. Integration

We wrap the entire system in `wrappers/roma_wrapper.py`. This acts as an adapter, allowing the canonicalizer to sit transparently in front of the RoMa matcher. 

User Flow: `Input Pair` $\rightarrow$ `Canonicalize` $\rightarrow$ `Warp Source` $\rightarrow$ `RoMa Match` $\rightarrow$ `Unwarp Keypoints` $\rightarrow$ `Output matches inside original coords`.

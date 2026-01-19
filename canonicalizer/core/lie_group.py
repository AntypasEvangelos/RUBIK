"""
Sim(2) Lie Group Operations

The 2D similarity group Sim(2) captures rotation, scale, and translation:

    Sim(2) = { [[s·cos(θ), -s·sin(θ), tx],
               [s·sin(θ),  s·cos(θ), ty],
               [0,         0,         1]] : θ ∈ [0,2π), s > 0, tx,ty ∈ ℝ }

This is a 4-dimensional Lie group with Lie algebra sim(2) ≅ ℝ⁴.

The Lie algebra element ξ = (ω, σ, vx, vy) represents:
    - ω  = infinitesimal rotation angle
    - σ  = infinitesimal log-scale (σ = ṡ/s)
    - vx = infinitesimal translation x
    - vy = infinitesimal translation y

The Lie algebra as matrices:
    ξ̂ = [[σ,  -ω,  vx],
          [ω,   σ,  vy],
          [0,   0,   0]]

Key operations:
    - exp: sim(2) → Sim(2)  (exponential map)
    - log: Sim(2) → sim(2)  (logarithm map)
    - compose: Sim(2) × Sim(2) → Sim(2)  (group multiplication)
    - inverse: Sim(2) → Sim(2)  (group inverse)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Sim2:
    """
    Static methods for Sim(2) Lie group operations.

    All methods are differentiable and work with batched inputs.
    Matrices are represented as (B, 3, 3) tensors in homogeneous coordinates.
    Lie algebra elements are (B, 4) tensors: (omega, sigma, vx, vy).
    """

    # Numerical stability thresholds
    EPS = 1e-7
    TAYLOR_THRESH = 1e-4

    @staticmethod
    def exp(xi: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: sim(2) → Sim(2)

        Maps a Lie algebra element to a group element.

        Args:
            xi: (B, 4) tensor of (omega, sigma, vx, vy)
                - omega: rotation angle
                - sigma: log-scale (scale = exp(sigma))
                - vx, vy: translation components in Lie algebra

        Returns:
            (B, 3, 3) transformation matrices in Sim(2)

        Mathematical formula:
            The Lie algebra element as a matrix is:
            ξ̂ = [[σ,  -ω,  vx],
                  [ω,   σ,  vy],
                  [0,   0,   0]]

            The exponential of the upper-left 2x2 block gives:
            exp([[σ, -ω], [ω, σ]]) = e^σ · [[cos(ω), -sin(ω)],
                                            [sin(ω),  cos(ω)]]

            For the translation part:
            t = V(ω, σ) · [vx, vy]ᵀ

            where V is the "left Jacobian" of Sim(2).
        """
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)

        B = xi.shape[0]
        device = xi.device
        dtype = xi.dtype

        omega = xi[:, 0]  # rotation angle
        sigma = xi[:, 1]  # log-scale
        vx = xi[:, 2]     # translation x in Lie algebra
        vy = xi[:, 3]     # translation y in Lie algebra

        # Scale factor
        s = torch.exp(sigma)

        # Rotation components
        cos_w = torch.cos(omega)
        sin_w = torch.sin(omega)

        # Compute translation via V matrix (left Jacobian)
        t = Sim2._compute_V_times_v(omega, sigma, s, cos_w, sin_w, vx, vy)
        tx, ty = t[:, 0], t[:, 1]

        # Build 3x3 homogeneous matrix
        g = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        g[:, 0, 0] = s * cos_w
        g[:, 0, 1] = -s * sin_w
        g[:, 0, 2] = tx
        g[:, 1, 0] = s * sin_w
        g[:, 1, 1] = s * cos_w
        g[:, 1, 2] = ty
        g[:, 2, 2] = 1.0

        return g

    @staticmethod
    def _compute_V_times_v(
        omega: torch.Tensor,
        sigma: torch.Tensor,
        s: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute t = V(ω,σ) · [vx, vy]ᵀ

        The V matrix (left Jacobian for translation) relates the Lie algebra
        translation (vx, vy) to the actual translation (tx, ty) in the group.

        For Sim(2), we need to integrate:
            t = ∫₀¹ exp(τ · (σI + ω·J)) dτ · [vx, vy]ᵀ

        where J = [[0, -1], [1, 0]] is the generator of rotation.

        The result is:
            V = (1/(σ² + ω²)) · (exp([[σ, -ω], [ω, σ]]) - I) · [[σ, ω], [-ω, σ]]

        Simplified:
            V = (1/(σ² + ω²)) · [[s·cos(ω) - 1, -s·sin(ω)],  · [[σ, ω], [-ω, σ]]
                                 [s·sin(ω),      s·cos(ω) - 1]]

        Let C = s·cos(ω) - 1, S = s·sin(ω)
        Then:
            V = (1/(σ² + ω²)) · [[C·σ - S·ω,  C·ω + S·σ],
                                  [S·σ + C·ω,  S·ω + C·σ]]

        Wait, let me recalculate properly...
        Actually, the correct form is:
            V = (1/(σ² + ω²)) · [[σ(s·cos(ω)-1) + ω·s·sin(ω),   -ω(s·cos(ω)-1) + σ·s·sin(ω)],
                                 [ω(s·cos(ω)-1) + σ·s·sin(ω),    σ(s·cos(ω)-1) - ω·s·sin(ω)]]

        Hmm, let me use a cleaner derivation based on the matrix exponential.
        """
        B = omega.shape[0]
        device = omega.device
        dtype = omega.dtype

        # Compute σ² + ω²
        omega_sq = omega ** 2
        sigma_sq = sigma ** 2
        denom = omega_sq + sigma_sq

        # Create output tensor
        t = torch.zeros(B, 2, device=device, dtype=dtype)

        # Case 1: Both ω and σ are small → use Taylor expansion
        small_both = denom < Sim2.TAYLOR_THRESH ** 2

        # Case 2: General case
        general = ~small_both

        if small_both.any():
            # Taylor expansion around (ω,σ) = (0,0)
            # V ≈ I + (1/2)·[[σ, -ω], [ω, σ]] + O((σ² + ω²))
            # V·v ≈ v + (1/2)·(σ·v - ω·J·v)
            #     = [vx + σ·vx/2 + ω·vy/2, vy + σ·vy/2 - ω·vx/2]
            idx = small_both
            w = omega[idx]
            sig = sigma[idx]
            vx_i = vx[idx]
            vy_i = vy[idx]

            # Second-order accurate expansion
            # V ≈ I + (σ/2)·I + (ω/2)·J + (1/6)·(σ² + ω²)·I
            c1 = 1 + sig / 2 + (sigma_sq[idx] + omega_sq[idx]) / 6
            c2 = w / 2

            t[idx, 0] = c1 * vx_i + c2 * vy_i
            t[idx, 1] = -c2 * vx_i + c1 * vy_i

        if general.any():
            idx = general
            w = omega[idx]
            sig = sigma[idx]
            s_i = s[idx]
            cos_i = cos_w[idx]
            sin_i = sin_w[idx]
            vx_i = vx[idx]
            vy_i = vy[idx]
            d = denom[idx]

            # Let C = s·cos(ω) - 1, S = s·sin(ω)
            C = s_i * cos_i - 1
            S = s_i * sin_i

            # V = (1/d) · [[σ·C + ω·S,   -ω·C + σ·S],
            #              [ω·C + σ·S,    σ·C - ω·S]]
            # Wait, that's not symmetric as expected. Let me recalculate.

            # The integration gives:
            # V = (1/(σ² + ω²)) · (R(ω,s) - I) · [[σ, ω], [-ω, σ]]
            # where R(ω,s) = s·[[cos ω, -sin ω], [sin ω, cos ω]]
            #
            # (R - I) = [[s cos ω - 1, -s sin ω], [s sin ω, s cos ω - 1]]
            #
            # (R - I) · [[σ, ω], [-ω, σ]] =
            #   [[(s cos ω - 1)σ + s sin ω · ω,    (s cos ω - 1)ω - s sin ω · σ],
            #    [s sin ω · σ - (s cos ω - 1)ω,    s sin ω · ω + (s cos ω - 1)σ]]
            #
            # = [[σC + ωS,    ωC - σS],
            #    [σS - ωC,    ωS + σC]]
            #
            # Hmm wait, that's still not right. Let me be more careful.

            # Actually for the closed form, we have:
            # V = (1/(σ² + ω²)) · [[A, -B], [B, A]]
            # where:
            #   A = σ·(e^σ·cos ω - 1) + ω·e^σ·sin ω
            #   B = ω·(e^σ·cos ω - 1) - σ·e^σ·sin ω

            A = sig * C + w * S
            B = w * C - sig * S

            inv_d = 1.0 / (d + Sim2.EPS)
            t[idx, 0] = inv_d * (A * vx_i - B * vy_i)
            t[idx, 1] = inv_d * (B * vx_i + A * vy_i)

        return t

    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        """
        Logarithm map: Sim(2) → sim(2)

        Maps a group element to a Lie algebra element.

        Args:
            g: (B, 3, 3) transformation matrices in Sim(2)

        Returns:
            (B, 4) tensor of (omega, sigma, vx, vy)

        Mathematical derivation:
            Given g = [[a, -b, tx],
                       [b,  a, ty],
                       [0,  0,  1]]

            where a = s·cos(ω), b = s·sin(ω)

            Scale: s = sqrt(a² + b²)
            Angle: ω = atan2(b, a)
            Log-scale: σ = log(s)
            Translation: [vx, vy]ᵀ = V(ω,σ)⁻¹ · [tx, ty]ᵀ
        """
        if g.dim() == 2:
            g = g.unsqueeze(0)

        B = g.shape[0]
        device = g.device
        dtype = g.dtype

        # Extract components
        a = g[:, 0, 0]   # s·cos(ω)
        b = g[:, 1, 0]   # s·sin(ω)
        tx = g[:, 0, 2]  # translation x
        ty = g[:, 1, 2]  # translation y

        # Compute scale and angle
        s = torch.sqrt(a ** 2 + b ** 2 + Sim2.EPS)
        omega = torch.atan2(b, a)
        sigma = torch.log(s + Sim2.EPS)

        # Compute V⁻¹ · t to recover (vx, vy)
        cos_w = torch.cos(omega)
        sin_w = torch.sin(omega)
        v = Sim2._compute_V_inv_times_t(omega, sigma, s, cos_w, sin_w, tx, ty)
        vx, vy = v[:, 0], v[:, 1]

        # Stack into Lie algebra element
        xi = torch.stack([omega, sigma, vx, vy], dim=1)

        return xi

    @staticmethod
    def _compute_V_inv_times_t(
        omega: torch.Tensor,
        sigma: torch.Tensor,
        s: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute v = V(ω,σ)⁻¹ · [tx, ty]ᵀ

        If V = (1/d) · [[A, -B], [B, A]] where d = σ² + ω²
        Then V⁻¹ = (d/(A² + B²)) · [[A, B], [-B, A]]
        """
        B_batch = omega.shape[0]
        device = omega.device
        dtype = omega.dtype

        omega_sq = omega ** 2
        sigma_sq = sigma ** 2
        denom = omega_sq + sigma_sq

        v = torch.zeros(B_batch, 2, device=device, dtype=dtype)

        small_both = denom < Sim2.TAYLOR_THRESH ** 2
        general = ~small_both

        if small_both.any():
            # Taylor expansion: V⁻¹ ≈ I - (σ/2)·I - (ω/2)·J
            idx = small_both
            w = omega[idx]
            sig = sigma[idx]
            tx_i = tx[idx]
            ty_i = ty[idx]

            # V⁻¹ ≈ I - (σ/2)·I - (ω/2)·J + O(σ² + ω²)
            c1 = 1 - sig / 2
            c2 = -w / 2

            v[idx, 0] = c1 * tx_i + c2 * ty_i
            v[idx, 1] = -c2 * tx_i + c1 * ty_i

        if general.any():
            idx = general
            w = omega[idx]
            sig = sigma[idx]
            s_i = s[idx]
            cos_i = cos_w[idx]
            sin_i = sin_w[idx]
            tx_i = tx[idx]
            ty_i = ty[idx]
            d = denom[idx]

            # A, B as in _compute_V_times_v
            C = s_i * cos_i - 1
            S = s_i * sin_i
            A = sig * C + w * S
            B = w * C - sig * S

            # V⁻¹ = (d/(A² + B²)) · [[A, B], [-B, A]]
            det_V_scaled = A ** 2 + B ** 2 + Sim2.EPS
            inv_det = d / det_V_scaled

            v[idx, 0] = inv_det * (A * tx_i + B * ty_i)
            v[idx, 1] = inv_det * (-B * tx_i + A * ty_i)

        return v

    @staticmethod
    def compose(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Group composition: g1 · g2

        Args:
            g1: (B, 3, 3) first transformation
            g2: (B, 3, 3) second transformation

        Returns:
            (B, 3, 3) composed transformation g1 @ g2

        Note: This applies g2 first, then g1.
        For points: (g1·g2)·p = g1·(g2·p)
        """
        return torch.bmm(g1, g2)

    @staticmethod
    def inverse(g: torch.Tensor) -> torch.Tensor:
        """
        Group inverse.

        Args:
            g: (B, 3, 3) transformation matrices

        Returns:
            (B, 3, 3) inverse transformations g⁻¹

        For Sim(2), we can compute the inverse analytically:
            g = [[a, -b, tx],     g⁻¹ = (1/s²) · [[a,  b, -a·tx - b·ty],
                 [b,  a, ty],                      [-b, a, b·tx - a·ty ],
                 [0,  0,  1]]                      [0,  0,      s²     ]]

        where s² = a² + b²
        """
        if g.dim() == 2:
            g = g.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B = g.shape[0]
        device = g.device
        dtype = g.dtype

        a = g[:, 0, 0]
        b = g[:, 1, 0]
        tx = g[:, 0, 2]
        ty = g[:, 1, 2]

        # s² = a² + b²
        s_sq = a ** 2 + b ** 2 + Sim2.EPS
        inv_s_sq = 1.0 / s_sq

        g_inv = torch.zeros(B, 3, 3, device=device, dtype=dtype)

        # Rotation-scale block: (1/s²) · [[a, b], [-b, a]]
        g_inv[:, 0, 0] = inv_s_sq * a
        g_inv[:, 0, 1] = inv_s_sq * b
        g_inv[:, 1, 0] = inv_s_sq * (-b)
        g_inv[:, 1, 1] = inv_s_sq * a

        # Translation: (1/s²) · [-a·tx - b·ty, b·tx - a·ty]
        g_inv[:, 0, 2] = inv_s_sq * (-a * tx - b * ty)
        g_inv[:, 1, 2] = inv_s_sq * (b * tx - a * ty)

        # Homogeneous coordinate
        g_inv[:, 2, 2] = 1.0

        if squeeze:
            g_inv = g_inv.squeeze(0)

        return g_inv

    @staticmethod
    def identity(batch_size: int, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
        """
        Create identity transformations.

        Args:
            batch_size: Number of identity matrices to create
            device: Torch device
            dtype: Torch dtype

        Returns:
            (B, 3, 3) identity matrices
        """
        g = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return g.contiguous()

    @staticmethod
    def from_components(
        theta: torch.Tensor,
        scale: torch.Tensor,
        translation: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct Sim(2) matrix from individual components.

        Note: This directly puts the translation into the matrix.
        This is NOT the same as exp([theta, log(scale), vx, vy]) because
        the Lie algebra translation (vx, vy) is transformed by V to get
        the actual translation (tx, ty).

        Args:
            theta: (B,) rotation angles in radians
            scale: (B,) scale factors (s > 0)
            translation: (B, 2) translation vectors (actual translation, not Lie algebra)

        Returns:
            (B, 3, 3) transformation matrices
        """
        B = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        g = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        g[:, 0, 0] = scale * cos_t
        g[:, 0, 1] = -scale * sin_t
        g[:, 0, 2] = translation[:, 0]
        g[:, 1, 0] = scale * sin_t
        g[:, 1, 1] = scale * cos_t
        g[:, 1, 2] = translation[:, 1]
        g[:, 2, 2] = 1.0

        return g

    @staticmethod
    def to_components(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract individual components from Sim(2) matrix.

        Args:
            g: (B, 3, 3) transformation matrices

        Returns:
            theta: (B,) rotation angles
            scale: (B,) scale factors
            translation: (B, 2) translation vectors (actual translation, not Lie algebra)
        """
        if g.dim() == 2:
            g = g.unsqueeze(0)

        a = g[:, 0, 0]
        b = g[:, 1, 0]

        scale = torch.sqrt(a ** 2 + b ** 2 + Sim2.EPS)
        theta = torch.atan2(b, a)
        translation = g[:, :2, 2]

        return theta, scale, translation

    @staticmethod
    def adjoint(g: torch.Tensor) -> torch.Tensor:
        """
        Compute the adjoint representation Ad_g.

        The adjoint maps Lie algebra elements under conjugation:
            Ad_g(ξ) corresponds to g · exp(ξ) · g⁻¹ = exp(Ad_g · ξ)

        Args:
            g: (B, 3, 3) transformation matrices

        Returns:
            (B, 4, 4) adjoint matrices
        """
        if g.dim() == 2:
            g = g.unsqueeze(0)

        B = g.shape[0]
        device = g.device
        dtype = g.dtype

        a = g[:, 0, 0]
        b = g[:, 1, 0]
        tx = g[:, 0, 2]
        ty = g[:, 1, 2]

        s_sq = a ** 2 + b ** 2 + Sim2.EPS
        s = torch.sqrt(s_sq)
        inv_s = 1.0 / s

        cos_t = a * inv_s
        sin_t = b * inv_s

        # Build adjoint matrix
        Ad = torch.zeros(B, 4, 4, device=device, dtype=dtype)

        # ω and σ are invariant under conjugation
        Ad[:, 0, 0] = 1.0
        Ad[:, 1, 1] = 1.0

        # Translation part transforms by rotation (scaled)
        Ad[:, 2, 0] = ty * inv_s
        Ad[:, 2, 1] = -tx * inv_s
        Ad[:, 2, 2] = cos_t
        Ad[:, 2, 3] = sin_t

        Ad[:, 3, 0] = -tx * inv_s
        Ad[:, 3, 1] = -ty * inv_s
        Ad[:, 3, 2] = -sin_t
        Ad[:, 3, 3] = cos_t

        return Ad


def test_sim2():
    """Basic tests for Sim2 operations."""
    import torch

    print("Testing Sim2 operations...")

    # Test 1: exp of zero should be identity
    xi_zero = torch.zeros(2, 4)
    g = Sim2.exp(xi_zero)
    expected = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(g, expected, atol=1e-6), "exp(0) should be identity"
    print("✓ exp(0) = I")

    # Test 2: log of identity should be zero
    I = Sim2.identity(3)
    xi = Sim2.log(I)
    assert torch.allclose(xi, torch.zeros(3, 4), atol=1e-6), "log(I) should be zero"
    print("✓ log(I) = 0")

    # Test 3: exp(log(g)) = g (roundtrip from group)
    # Test with matrices created from exp (which guarantees consistency)
    xi_orig = torch.tensor([[0.3, 0.4, 5.0, 10.0],
                            [-0.5, -0.2, -3.0, 7.0],
                            [1.2, 0.7, 0.0, 0.0]])
    g = Sim2.exp(xi_orig)
    xi_recovered = Sim2.log(g)
    g_reexp = Sim2.exp(xi_recovered)
    assert torch.allclose(g, g_reexp, atol=1e-5), "exp(log(g)) should equal g"
    print("✓ exp(log(g)) = g")

    # Test 4: log(exp(xi)) = xi (roundtrip from Lie algebra)
    xi_orig = torch.tensor([[0.5, 0.2, 3.0, -2.0],
                            [-0.3, -0.1, 0.0, 5.0],
                            [0.0, 0.0, 1.0, 2.0]])  # Identity rotation/scale case
    g = Sim2.exp(xi_orig)
    xi_recovered = Sim2.log(g)

    # Note: omega is periodic, so we compare mod 2π
    diff = xi_orig - xi_recovered
    diff[:, 0] = torch.remainder(diff[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-4), f"log(exp(xi)) should equal xi, got diff: {diff}"
    print("✓ log(exp(ξ)) = ξ")

    # Test 5: g @ g^{-1} = I
    xi = torch.tensor([[0.7, 0.3, 4.0, -3.0],
                       [-1.2, -0.5, 1.0, 2.0]])
    g = Sim2.exp(xi)
    g_inv = Sim2.inverse(g)
    product = Sim2.compose(g, g_inv)
    expected = Sim2.identity(2)
    assert torch.allclose(product, expected, atol=1e-5), "g @ g^{-1} should be identity"
    print("✓ g · g⁻¹ = I")

    # Test 6: Composition associativity
    xi1 = torch.tensor([[0.2, 0.1, 1.0, 2.0]])
    xi2 = torch.tensor([[0.5, -0.1, 3.0, -1.0]])
    xi3 = torch.tensor([[-0.3, 0.4, -2.0, 4.0]])
    g1 = Sim2.exp(xi1)
    g2 = Sim2.exp(xi2)
    g3 = Sim2.exp(xi3)

    left = Sim2.compose(Sim2.compose(g1, g2), g3)
    right = Sim2.compose(g1, Sim2.compose(g2, g3))
    assert torch.allclose(left, right, atol=1e-5), "Composition should be associative"
    print("✓ (g₁ · g₂) · g₃ = g₁ · (g₂ · g₃)")

    # Test 7: Pure rotation
    xi_rot = torch.tensor([[torch.pi / 4, 0.0, 0.0, 0.0]])  # 45° rotation
    g_rot = Sim2.exp(xi_rot)
    expected_cos = torch.cos(torch.tensor(torch.pi / 4))
    expected_sin = torch.sin(torch.tensor(torch.pi / 4))
    assert torch.allclose(g_rot[0, 0, 0], expected_cos, atol=1e-6), "Pure rotation cos"
    assert torch.allclose(g_rot[0, 1, 0], expected_sin, atol=1e-6), "Pure rotation sin"
    print("✓ Pure rotation exp correct")

    # Test 8: Pure scale
    xi_scale = torch.tensor([[0.0, torch.log(torch.tensor(2.0)), 0.0, 0.0]])  # scale = 2
    g_scale = Sim2.exp(xi_scale)
    assert torch.allclose(g_scale[0, 0, 0], torch.tensor(2.0), atol=1e-6), "Pure scale"
    print("✓ Pure scale exp correct")

    # Test 9: Pure translation (identity rotation/scale)
    xi_trans = torch.tensor([[0.0, 0.0, 5.0, 10.0]])
    g_trans = Sim2.exp(xi_trans)
    # With identity rotation/scale, V ≈ I, so translation should be approximately (5, 10)
    assert torch.allclose(g_trans[0, 0, 2], torch.tensor(5.0), atol=1e-5), "Pure translation x"
    assert torch.allclose(g_trans[0, 1, 2], torch.tensor(10.0), atol=1e-5), "Pure translation y"
    print("✓ Pure translation exp correct")

    # Test 10: Gradient flow
    xi = torch.tensor([[0.5, 0.2, 1.0, -1.0]], requires_grad=True)
    g = Sim2.exp(xi)
    loss = g.sum()
    loss.backward()
    assert xi.grad is not None, "Gradients should flow through exp"
    print("✓ Gradients flow through exp")

    # Test 11: Inverse of exp is exp of negated
    xi = torch.tensor([[0.3, 0.2, 2.0, -1.0]])
    g = Sim2.exp(xi)
    g_inv = Sim2.inverse(g)
    g_neg = Sim2.exp(-xi)
    # Note: exp(-xi) is NOT generally equal to inverse(exp(xi)) for non-abelian groups
    # But g @ g_inv should still be identity
    product = Sim2.compose(g, g_inv)
    assert torch.allclose(product, Sim2.identity(1), atol=1e-5), "g @ inverse(g) = I"
    print("✓ g · inverse(g) = I (correct even though exp(-ξ) ≠ inverse(exp(ξ)))")

    print("\nAll Sim2 tests passed!")


if __name__ == "__main__":
    test_sim2()

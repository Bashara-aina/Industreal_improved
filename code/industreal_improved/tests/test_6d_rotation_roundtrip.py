"""
Q40 Guard — 6D Rotation Round-Trip Unit Tests (V1 doc 219 / MASTER_VERIFICATION Item 40)

Verifies that the Zhou et al. (CVPR 2019) 6D continuous rotation representation
round-trips faithfully through conversion to/from SO(3) matrices, and that the
geodesic loss function satisfies basic mathematical invariants.

All tests run CPU-only (no GPU required).
"""

import torch
import pytest

from src.models.head_pose_geo import (
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    geodesic_loss,
)
from src.losses.geodesic_loss import (
    _gram_schmidt_rotation,
    huberised_geodesic_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_rotation_matrix(batch_size: int = 1, seed: int = 0) -> torch.Tensor:
    """Generate a random SO(3) rotation matrix via QR decomposition."""
    torch.manual_seed(seed)
    A = torch.randn(batch_size, 3, 3)
    Q, R = torch.linalg.qr(A)
    # Ensure det = +1 (reflect if needed)
    det = torch.det(Q)
    Q[:, :, 0] *= det.sign().unsqueeze(1)
    return Q  # [B, 3, 3]


def _rotation_about_z(theta_rad: float) -> torch.Tensor:
    """Rotation matrix about Z axis by theta radians."""
    c, s = torch.cos(torch.tensor(theta_rad)), torch.sin(torch.tensor(theta_rad))
    return torch.tensor([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]]).unsqueeze(0)  # [1, 3, 3]


# ---------------------------------------------------------------------------
# Round-trip: rotation matrix → 6D → rotation matrix
# ---------------------------------------------------------------------------

class TestRotationRoundTrip:
    """Verify that matrix↔6D conversions are inverses within numerical tolerance."""

    REconstruction_tol = 1e-5

    def test_random_round_trip(self):
        """Random matrix → 6D → Gram-Schmidt matrix recovers original."""
        R = _random_rotation_matrix(batch_size=32, seed=42)
        d6 = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        err = (R - R_recovered).abs().max().item()
        assert err < self.REconstruction_tol, f"Max element error: {err:.2e}"

    def test_identity_round_trip(self):
        """Identity matrix round-trips exactly."""
        I = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        d6 = rotation_matrix_to_6d(I)
        R = rotation_6d_to_matrix(d6)
        err = (I - R).abs().max().item()
        assert err < self.REconstruction_tol, f"Identity error: {err:.2e}"

    def test_orthonormality_of_recovered(self):
        """Recovered matrix columns are orthonormal (R^T R = I)."""
        R = _random_rotation_matrix(batch_size=32, seed=99)
        d6 = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        RtR = R_recovered.transpose(1, 2) @ R_recovered
        I = torch.eye(3).unsqueeze(0).expand_as(RtR)
        err = (RtR - I).abs().max().item()
        assert err < self.REconstruction_tol, f"Orthonormality error: {err:.2e}"

    def test_positive_determinant(self):
        """Recovered matrix has det = +1 (SO(3) not O(3))."""
        R = _random_rotation_matrix(batch_size=32, seed=7)
        d6 = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        det = torch.det(R_recovered)
        assert (det - 1.0).abs().max().item() < self.REconstruction_tol, \
            f"Determinant deviates from +1"

    def test_batch_independence(self):
        """Each item in a batch round-trips independently (no cross-talk)."""
        R = _random_rotation_matrix(batch_size=16, seed=1)
        d6 = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        # Verify per-sample error is small
        per_sample_err = (R - R_recovered).reshape(16, 9).abs().max(dim=1).values
        assert (per_sample_err < self.REconstruction_tol).all(), \
            f"Max per-sample error: {per_sample_err.max().item():.2e}"


# ---------------------------------------------------------------------------
# Special-angle rotations (90°, 180°)
# ---------------------------------------------------------------------------

class TestSpecialAngles:
    """6D round-trip at known hard angles."""

    tol = 1e-5

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 60, 90, 120, 135, 150, 180, 270])
    def test_rotation_about_z(self, angle_deg):
        """Rotation about Z by {angle_deg}° round-trips correctly."""
        theta = torch.tensor(angle_deg * 3.141592653589793 / 180.0)
        R = _rotation_about_z(theta.item())
        d6 = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        err = (R - R_recovered).abs().max().item()
        assert err < self.tol, f"Z-rotation {angle_deg}° error: {err:.2e}"


# ---------------------------------------------------------------------------
# Geodesic loss invariants
# ---------------------------------------------------------------------------

class TestGeodesicLossInvariants:
    """Verify geodesic loss satisfies key mathematical properties.

    Note: self-loss is not exactly 0 because the geodesic loss clamps its
    arccos input to < 1 (``clamp(..., 1 - 1e-7)``) to prevent NaN from
    floating-point drift, introducing a ~5e-4 rad systematic error.
    """

    selftol = 1e-3   # self-loss tolerance (arccos clamp ~5e-4)
    symmtol = 1e-5   # symmetry tolerance

    def test_loss_self_zero(self):
        """geodesic_loss(R, R) ≈ 0 for various rotation matrices."""
        for seed in range(10):
            R = _random_rotation_matrix(batch_size=8, seed=seed)
            loss = geodesic_loss(R, R)
            assert loss.item() < self.selftol, f"Loss(R,R) = {loss.item():.2e} (seed={seed})"

    def test_loss_symmetry(self):
        """geodesic_loss(a, b) ≈ geodesic_loss(b, a)."""
        for seed in range(10):
            Ra = _random_rotation_matrix(batch_size=8, seed=seed)
            Rb = _random_rotation_matrix(batch_size=8, seed=seed + 1000)
            loss_ab = geodesic_loss(Ra, Rb)
            loss_ba = geodesic_loss(Rb, Ra)
            err = (loss_ab - loss_ba).abs().item()
            assert err < self.symmtol, f"Symmetry error: {err:.2e} (seed={seed})"

    def test_loss_triangle_inequality(self):
        """geodesic_loss(a, c) <= geodesic_loss(a, b) + geodesic_loss(b, c)."""
        Ra = _random_rotation_matrix(batch_size=1, seed=10)
        Rb = _random_rotation_matrix(batch_size=1, seed=20)
        Rc = _random_rotation_matrix(batch_size=1, seed=30)
        d_ab = geodesic_loss(Ra, Rb)
        d_bc = geodesic_loss(Rb, Rc)
        d_ac = geodesic_loss(Ra, Rc)
        # Allow small numerical slack
        assert d_ac.item() <= d_ab.item() + d_bc.item() + 1e-4, \
            f"Triangle inequality violated: {d_ac.item():.4f} > {d_ab.item():.4f} + {d_bc.item():.4f}"

    def test_loss_range(self):
        """Geodesic loss is in [0, pi] radians."""
        for seed in range(5):
            Ra = _random_rotation_matrix(batch_size=16, seed=seed)
            Rb = _random_rotation_matrix(batch_size=16, seed=seed + 500)
            loss = geodesic_loss(Ra, Rb)
            assert 0.0 <= loss.item() <= 3.1416, \
                f"Loss {loss.item():.4f} outside [0, pi]"


# ---------------------------------------------------------------------------
# Huberised geodesic loss invariants
# ---------------------------------------------------------------------------

class TestHuberisedGeodesicLoss:
    """Verify huberised_geodesic_loss basic properties.

    Note: self-loss is not exactly 0 because the loss clamps its arccos
    input to < 1 (``clamp(..., 1 - 1e-6)``) to prevent NaN, introducing
    a small systematic error (< 0.01°).
    """

    selftol = 1e-2  # degrees; accounts for the 1e-6 arccos clamp

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_loss_self_zero(self, batch_size):
        """huberised_geodesic_loss(x, x) ≈ 0 for any batch size."""
        R = _random_rotation_matrix(batch_size=batch_size, seed=0)
        d6 = rotation_matrix_to_6d(R)
        loss = huberised_geodesic_loss(d6, d6)
        assert loss.item() < self.selftol, \
            f"Self-loss = {loss.item():.2e} (batch={batch_size})"

    def test_non_negative(self):
        """Loss is always non-negative."""
        Ra = _random_rotation_matrix(batch_size=32, seed=10)
        Rb = _random_rotation_matrix(batch_size=32, seed=20)
        d6a = rotation_matrix_to_6d(Ra)
        d6b = rotation_matrix_to_6d(Rb)
        loss = huberised_geodesic_loss(d6a, d6b)
        assert loss.item() >= 0.0, f"Negative loss: {loss.item():.4f}"


# ---------------------------------------------------------------------------
# Gram-Schmidt consistency (cross-file)
# ---------------------------------------------------------------------------

class TestGramSchmidtConsistency:
    """Both head_pose_geo and geodesic_loss use the same Gram-Schmidt."""

    tol = 1e-5

    def test_equivalent_implementations(self):
        """rotation_6d_to_matrix and _gram_schmidt_rotation produce same result."""
        d6 = torch.randn(8, 6)
        R_geo = rotation_6d_to_matrix(d6)
        R_huber = _gram_schmidt_rotation(d6[:, :3], d6[:, 3:])
        err = (R_geo - R_huber).abs().max().item()
        assert err < self.tol, f"Gram-Schmidt mismatch: {err:.2e}"

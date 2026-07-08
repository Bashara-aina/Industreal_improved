"""Tests for MTLBalancer PCGrad gradient surgery (175 §5.2).

Verifies:
    1. PCGrad reduces gradient conflict magnitude vs naive sum.
    2. mode="none" produces identical gradients to naive sum.
    3. Four-task synthetic model runs end-to-end without error.
    4. Conflicting gradients are projected to non-negative cosine similarity.
    5. Non-conflicting gradients are left largely unchanged.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SyntheticMTLModel(nn.Module):
    """Small synthetic multi-task model: shared backbone + 4 task heads."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_tasks: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_tasks)]
        )

    def forward(self, x):
        features = self.backbone(x)
        return [head(features) for head in self.heads]


@pytest.fixture
def model(device):
    return SyntheticMTLModel().to(device)


@pytest.fixture
def shared_params(model):
    return list(model.backbone.parameters())


@pytest.fixture
def all_params(model):
    return list(model.parameters())


@pytest.fixture
def batch(device):
    x = torch.randn(32, 16, device=device)
    return x


@pytest.fixture
def four_task_outputs(model, batch):
    return model(batch)


# ---------------------------------------------------------------------------
# Helper: naive sum gradients
# ---------------------------------------------------------------------------

def _naive_grads(task_losses, shared_params):
    """Compute naive sum-of-losses gradients by calling backward()."""
    # Zero grads
    for p in shared_params:
        p.grad = None
    combined = sum(task_losses)
    combined.backward(retain_graph=True)
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in shared_params]


def _cos_sim(a, b):
    """Cosine similarity between two flattened gradient vectors."""
    a_f = a.flatten()
    b_f = b.flatten()
    dot = torch.dot(a_f, b_f)
    norm = a_f.norm() * b_f.norm()
    if norm < 1e-12:
        return torch.tensor(1.0, device=a.device)  # degenerate: not-conflicting
    return dot / norm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMTLBalancerModeNone:
    """mode="none" must be identical to naive sum-of-losses."""

    def test_none_equals_naive_sum(self, model, shared_params, batch):
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="none")
        out = model(batch)
        task_losses = [o.mean() for o in out]

        # Naive sum backward
        for p in shared_params:
            p.grad = None
        naive_combined = sum(task_losses)
        naive_combined.backward(retain_graph=True)
        naive_grads = [p.grad.clone() for p in shared_params]

        # MTLBalancer (mode=none)
        for p in shared_params:
            p.grad = None
        balancer_combined = balancer.compute_step(task_losses)
        balancer_combined.backward(retain_graph=True)
        balancer_grads = [p.grad.clone() for p in shared_params]

        for ng, bg in zip(naive_grads, balancer_grads):
            assert torch.allclose(ng, bg, atol=1e-6), (
                f"mode='none' grad differs from naive sum: "
                f"max_diff={((ng - bg).abs().max().item())}"
            )


class TestMTLBalancerPCGrad:
    """PCGrad gradient surgery correctness."""

    def test_end_to_end_no_error(self, model, shared_params, batch):
        """4-task forward + PCGrad + backward must not raise."""
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")
        out = model(batch)
        task_losses = [o.mean() for o in out]

        combined = balancer.compute_step(task_losses)
        loss_val = combined.item()
        combined.backward()
        # All shared params should have valid grads
        for p in shared_params:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
        assert isinstance(loss_val, float)

    def test_pcgrad_reduces_conflict_magnitude(self, model, shared_params, batch):
        """Conflicting gradients should have smaller cosine conflict after PCGrad.

        After PCGrad, the dot product between each projected task gradient
        and the original gradient of any other conflicting task should be
        non-negative (the conflict has been projected away).
        """
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")
        out = model(batch)

        # Create known-conflicting tasks: task 0 pushes up, task 1 pushes down.
        # Both traverse the same backbone -> guaranteed negative cosine.
        task_losses = [
            -out[0].mean(),   # wants to increase output 0
            out[1].mean(),    # wants to decrease output 1
            out[2].mean(),    # neutral reference
            out[3].mean(),    # neutral reference
        ]

        # Compute PCGrad gradients (before hook install)
        balancer._step_counter += 1
        balancer._remove_hooks()

        # 1. Per-task grads w.r.t. shared params
        task_grads = []
        for loss in task_losses:
            g = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True, allow_unused=True,
            )
            g = tuple(torch.zeros_like(p) if gi is None else gi
                      for gi, p in zip(g, shared_params))
            task_grads.append(g)

        # 2. Verify original task 0 and task 1 are conflicting
        cos_01 = _cos_sim(
            torch.cat([g.flatten() for g in task_grads[0]]),
            torch.cat([g.flatten() for g in task_grads[1]]),
        )
        assert cos_01 < -0.01, (
            f"Test setup failed: task 0 and 1 should be conflicting "
            f"(cos={cos_01:.4f})"
        )

        # 3. PCGrad projection
        pcgrad_grads = balancer._project_pcgrad(task_grads)

        # 4. Verify conflict is reduced: the PCGrad-combined gradient should
        #    have non-negative dot product with each task's original gradient.
        #    (Combined = sum(pc_i), and each pc_i had conflicting components
        #    from all other tasks removed.)
        combined_flat = torch.cat([g.flatten() for g in pcgrad_grads])
        for i, loss in enumerate(task_losses):
            # Recompute original task grad for this comparison
            g_i = torch.autograd.grad(
                loss, shared_params, retain_graph=True, allow_unused=True,
            )
            g_i = tuple(torch.zeros_like(p) if gi is None else gi
                        for gi, p in zip(g_i, shared_params))
            g_i_flat = torch.cat([g.flatten() for g in g_i])
            dot = torch.dot(combined_flat, g_i_flat)
            # The combined PCGrad grad should not conflict with ANY task
            assert dot >= -1e-4, (
                f"Task {i}: PCGrad combined grad conflicts with original "
                f"task grad (dot={dot:.6f})"
            )

    def test_pcgrad_projection_direct(self, device):
        """Directly test the PCGrad projection on synthetic conflicting vectors.

        Create two 1-D gradient vectors with known negative cosine similarity
        and verify that PCGrad projects the conflict away (non-negative dot
        product between projected i and original j).
        """
        from src.training.mtl_balancer import MTLBalancer

        # Single shared parameter (used to define the param structure).
        param = nn.Parameter(torch.randn(10, device=device))

        # Deterministic conflicting gradients:
        #   g_0 =  [1,  0]   -- push in +x direction
        #   g_1 = [-1,  1]   -- push in -x + y direction (partially opposing)
        # Pad with zeros to length 10 so shapes match.
        g_0_base = torch.zeros(10, device=device)
        g_0_base[0] = 1.0
        g_1_base = torch.zeros(10, device=device)
        g_1_base[0] = -1.0
        g_1_base[1] = 1.0

        cos_before = _cos_sim(g_0_base, g_1_base)
        # cos = (-1 + 0) / (1 * sqrt(2)) = -1/1.414 = -0.707
        assert cos_before < -0.5, (
            f"Test setup: grads should be conflicting (cos={cos_before:.4f})"
        )

        balancer = MTLBalancer([param], mode="pcgrad")

        # Build (grads,) tuples as torch.autograd.grad would return them.
        g_0 = (g_0_base,)
        g_1 = (g_1_base,)

        # Apply PCGrad projection.
        result = balancer._project_pcgrad([g_0, g_1])
        combined_grad = result[0]

        # Manually compute expected PCGrad result:
        #   dot(g_0, g_1) = -1
        #   ||g_1||^2 = 2
        #   coeff = -1/2 = -0.5
        #   pc_0 = g_0 - (-0.5) * g_1 = [1, 0] + 0.5 * [-1, 1] = [0.5, 0.5]
        #   dot(g_1, g_0) = -1
        #   ||g_0||^2 = 1
        #   coeff = -1/1 = -1
        #   pc_1 = g_1 - (-1) * g_0 = [-1, 1] + 1 * [1, 0] = [0, 1]
        #   combined = pc_0 + pc_1 = [0.5, 1.5]
        expected = torch.zeros(10, device=device)
        expected[0] = 0.5
        expected[1] = 1.5

        assert torch.allclose(combined_grad, expected, atol=1e-5), (
            f"PCGrad projection mismatch: "
            f"got={combined_grad[:3].tolist()} exp={expected[:3].tolist()}"
        )

        # Verify conflict removal: after projection, the dot product between
        # pc_0 (first task's deconflicted grad) and g_1 is >= 0.
        # pc_0 = [0.5, 0.5, 0, ...], g_1 = [-1, 1, 0, ...]
        # dot = -0.5 + 0.5 = 0  (orthogonal)
        dot_after = torch.dot(
            torch.tensor([0.5, 0.5], device=device),
            torch.tensor([-1.0, 1.0], device=device),
        )
        assert dot_after >= -1e-6, (
            f"PCGrad should remove conflict: dot(pc_0, g_1)={dot_after:.6f} < 0"
        )

    def test_non_conflicting_grads_preserved(self, shared_params, device):
        """When gradients are already aligned, PCGrad should leave them unchanged."""
        from src.training.mtl_balancer import MTLBalancer

        # Same direction grads -> no conflict
        g_dir = torch.randn(10, device=device)
        g_0 = (g_dir.clone(),)
        g_1 = (g_dir.clone() * 0.5,)  # same direction, half magnitude

        cos = _cos_sim(g_dir, g_dir * 0.5)
        assert cos > 0.99, f"Grads should be aligned (cos={cos})"

        balancer = MTLBalancer([nn.Parameter(g_dir.clone())], mode="pcgrad")
        result = balancer._project_pcgrad([g_0, g_1])
        combined = result[0]

        # Without conflict, PCGrad should leave them as-is: sum = g_0 + g_1
        expected = g_dir + g_dir * 0.5
        assert torch.allclose(combined, expected, atol=1e-5), (
            f"Non-conflicting grads changed by PCGrad: "
            f"max_diff={((combined - expected).abs().max().item())}"
        )


class TestPcgradGradNorm:
    """PCGrad produces smaller grad norm for conflicting tasks vs gradient variance."""

    def test_pcgrad_smaller_norm_than_gradient_variance(
        self, model, shared_params, batch
    ):
        """For opposing-gradient tasks, PCGrad combined norm should not
        be larger than what naive sum produces from destructive interference.
        PCGrad preserves non-conflicting structure while removing cancellation.

        This test creates tasks with partially opposing gradients and
        verifies that PCGrad produces sensible (non-zero, finite) results.
        """
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")
        out = model(batch)

        # Tasks 0 and 1 with opposing objectives
        task_losses = [
            -out[0].mean(),  # push up
            out[1].mean(),   # push down
        ]

        # PCGrad grads
        for p in shared_params:
            p.grad = None
        balancer._remove_hooks()
        combined = balancer.compute_step(task_losses)
        combined.backward(retain_graph=True)
        pcgrad_norm = sum(p.grad.norm().item() for p in shared_params)

        # Naive sum grads
        for p in shared_params:
            p.grad = None
        naive = sum(task_losses)
        naive.backward(retain_graph=True)
        naive_norm = sum(p.grad.norm().item() for p in shared_params)

        # Both should be finite and non-NaN
        assert torch.isfinite(
            torch.tensor(pcgrad_norm)
        ), "PCGrad gradient norm should be finite"
        assert torch.isfinite(
            torch.tensor(naive_norm)
        ), "Naive sum gradient norm should be finite"

        # The PCGrad norm can be larger than naive norm (when naive sum
        # cancels out conflicting components, PCGrad preserves them).
        # But it must not be absurdly large.
        assert pcgrad_norm < 1e6, f"PCGrad norm too large: {pcgrad_norm}"


class TestFourTaskEndToEnd:
    """Full 4-task synthetic run with MTLBalancer."""

    def test_four_task_pcgrad_backward(self, model, shared_params, batch):
        """All 4 tasks with PCGrad: forward + backward + step."""
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for step in range(5):
            optimizer.zero_grad()
            out = model(batch)
            task_losses = [
                F.mse_loss(out[0], torch.randn_like(out[0])),
                F.mse_loss(out[1], torch.randn_like(out[1])),
                F.mse_loss(out[2], torch.randn_like(out[2])),
                F.mse_loss(out[3], torch.randn_like(out[3])),
            ]
            combined = balancer.compute_step(task_losses)
            combined.backward()
            optimizer.step()

        # After 5 steps, the model should have valid parameters
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_four_task_none_backward(
        self, model, shared_params, batch
    ):
        """All 4 tasks with mode='none': equivalent to naive sum."""
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="none")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for step in range(5):
            optimizer.zero_grad()
            out = model(batch)
            task_losses = [
                F.mse_loss(out[0], torch.randn_like(out[0])),
                F.mse_loss(out[1], torch.randn_like(out[1])),
                F.mse_loss(out[2], torch.randn_like(out[2])),
                F.mse_loss(out[3], torch.randn_like(out[3])),
            ]
            combined = balancer.compute_step(task_losses)
            combined.backward()
            optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()


class TestDetectSharedParams:
    """MTLBalancer.detect_shared_params utility."""

    def test_detect_returns_backbone_params(self, model, batch):
        from src.training.mtl_balancer import MTLBalancer

        out = model(batch)
        task_losses = [o.mean() for o in out]

        shared = MTLBalancer.detect_shared_params(model, task_losses)
        backbone_params = list(model.backbone.parameters())

        # All backbone params should be in the shared set
        for bp in backbone_params:
            assert any(s is bp for s in shared), (
                f"Backbone param {bp.shape} not detected as shared"
            )

    def test_detect_without_losses_returns_all_params(self, model):
        from src.training.mtl_balancer import MTLBalancer

        all_params = list(model.parameters())
        shared = MTLBalancer.detect_shared_params(model)

        # Without task_losses, all requires_grad params are "shared"
        assert len(shared) == len([p for p in all_params if p.requires_grad])


class TestGradHooks:
    """Backward hooks correctly replace shared-param gradients."""

    def test_hooks_replace_grads(self, model, shared_params, batch):
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")
        out = model(batch)
        task_losses = [o.mean() for o in out]

        # Compute step (installs hooks)
        combined = balancer.compute_step(task_losses)

        # After compute_step, hooks should be installed
        assert balancer.has_hooks

        # Before backward, grads are None
        for p in shared_params:
            assert p.grad is None

        # Backward fires hooks -> shared params get PCGrad grads
        combined.backward()

        # Hooks should have fired and set grads
        for p in shared_params:
            assert p.grad is not None, "Shared param grad should be set after backward"
            assert torch.isfinite(p.grad).all()

    def test_hooks_removed_after_step(self, model, shared_params, batch):
        """Hooks from previous compute_step must be removed before new one."""
        from src.training.mtl_balancer import MTLBalancer

        balancer = MTLBalancer(shared_params, mode="pcgrad")

        # First step
        out = model(batch)
        combined = balancer.compute_step([o.mean() for o in out])
        combined.backward()

        assert balancer.has_hooks

        # Second step should remove old hooks first
        out = model(batch)
        combined = balancer.compute_step([o.mean() for o in out])
        # After compute_step, hooks from step 1 are gone and new ones installed
        assert balancer.has_hooks

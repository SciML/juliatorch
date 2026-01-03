"""Performance tests for juliatorch.

These tests verify that performance characteristics remain reasonable
and catch any major performance regressions.
"""
import time
import pytest
import numpy as np
import torch
from juliacall import Main as jl
from .. import JuliaFunction


# Test function defined at module level for reuse
_f_simple = None
_py_f_simple = None


def _get_test_functions():
    """Get or create test Julia functions (cached for performance)."""
    global _f_simple, _py_f_simple
    if _f_simple is None:
        _f_simple = jl.seval("f(x) = 2 .* x")
        _py_f_simple = lambda x: _f_simple(x)
    return _f_simple, _py_f_simple


class TestForwardPerformance:
    """Test forward pass performance characteristics."""

    def test_forward_small_array_performance(self):
        """Forward pass on small arrays should complete quickly."""
        _, py_f = _get_test_functions()
        x = torch.randn(10, 10, dtype=torch.double, requires_grad=True)

        # Warmup
        for _ in range(3):
            _ = JuliaFunction.apply(py_f, x)

        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = JuliaFunction.apply(py_f, x)
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        # Forward pass on 10x10 should be well under 10ms (warmed up)
        assert avg_time_ms < 10, f"Forward pass too slow: {avg_time_ms:.2f}ms (expected < 10ms)"

    def test_forward_medium_array_performance(self):
        """Forward pass on medium arrays should complete in reasonable time."""
        _, py_f = _get_test_functions()
        x = torch.randn(100, 100, dtype=torch.double, requires_grad=True)

        # Warmup
        for _ in range(3):
            _ = JuliaFunction.apply(py_f, x)

        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = JuliaFunction.apply(py_f, x)
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        # Forward pass on 100x100 should be under 50ms (warmed up)
        assert avg_time_ms < 50, f"Forward pass too slow: {avg_time_ms:.2f}ms (expected < 50ms)"


class TestBackwardPerformance:
    """Test backward pass performance characteristics."""

    def test_backward_small_array_performance(self):
        """Backward pass on small arrays should complete in reasonable time."""
        _, py_f = _get_test_functions()

        # Warmup
        for _ in range(3):
            x = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            y.sum().backward()

        # Measure
        times = []
        for _ in range(5):
            x = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            start = time.perf_counter()
            y.sum().backward()
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        # Backward pass on 10x10 should be under 50ms (warmed up)
        # Note: First few calls may be slower due to JIT compilation
        assert avg_time_ms < 50, f"Backward pass too slow: {avg_time_ms:.2f}ms (expected < 50ms)"

    def test_backward_medium_array_performance(self):
        """Backward pass on medium arrays should complete in reasonable time.

        Note: ForwardDiff gradient computation scales with input size,
        so 100x100 arrays will take significantly longer than 10x10.
        """
        _, py_f = _get_test_functions()

        # Warmup
        for _ in range(2):
            x = torch.randn(50, 50, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            y.sum().backward()

        # Measure on 50x50 (100x100 would be too slow for regular testing)
        times = []
        for _ in range(3):
            x = torch.randn(50, 50, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            start = time.perf_counter()
            y.sum().backward()
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        # 50x50 backward should be under 500ms (warmed up)
        assert avg_time_ms < 500, f"Backward pass too slow: {avg_time_ms:.2f}ms (expected < 500ms)"


class TestMemoryUsage:
    """Test memory usage characteristics."""

    def test_no_memory_leak_repeated_forward(self):
        """Repeated forward passes should not accumulate memory."""
        import tracemalloc
        _, py_f = _get_test_functions()
        x = torch.randn(50, 50, dtype=torch.double, requires_grad=True)

        # Warmup and establish baseline
        for _ in range(5):
            _ = JuliaFunction.apply(py_f, x)

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Run many iterations
        for _ in range(50):
            _ = JuliaFunction.apply(py_f, x)

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth_kb = (final_memory - initial_memory) / 1024
        # Memory growth should be minimal (under 1MB for 50 iterations)
        assert memory_growth_kb < 1024, f"Memory grew by {memory_growth_kb:.1f}KB over 50 iterations"

    def test_no_memory_leak_repeated_backward(self):
        """Repeated backward passes should not accumulate excessive memory."""
        import tracemalloc
        _, py_f = _get_test_functions()

        # Warmup
        for _ in range(3):
            x = torch.randn(30, 30, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            y.sum().backward()

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Run iterations
        for _ in range(20):
            x = torch.randn(30, 30, dtype=torch.double, requires_grad=True)
            y = JuliaFunction.apply(py_f, x)
            y.sum().backward()

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth_kb = (final_memory - initial_memory) / 1024
        # Memory growth should be under 5MB for 20 iterations
        assert memory_growth_kb < 5120, f"Memory grew by {memory_growth_kb:.1f}KB over 20 iterations"


class TestGradientCorrectness:
    """Test that gradients are computed correctly (extends test_all.py)."""

    def test_gradient_exp_function(self):
        """Test gradient computation for exp(-x^2)."""
        from torch.autograd import gradcheck

        f = jl.seval("f(x) = exp.(-x .^ 2)")
        py_f = lambda x: f(x)

        x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
        assert gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4)

    def test_gradient_complex_function(self):
        """Test gradient computation for a more complex function."""
        from torch.autograd import gradcheck

        f = jl.seval("f(x) = sin.(x) .* cos.(x)")
        py_f = lambda x: f(x)

        x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
        assert gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4)

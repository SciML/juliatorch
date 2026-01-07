"""
JuliaTorch: Automatic differentiation of Julia functions in PyTorch.

This package provides a PyTorch autograd.Function that allows you to call Julia
functions within PyTorch computational graphs with automatic differentiation.
The gradients are computed using ForwardDiff.jl on the Julia side.
"""

# Setup

from juliacall import Main as _jl # TODO: auto-install Julia

import numpy as _np
import torch as _torch
from torch.autograd import Function as _Function

_loss = _jl.seval("loss(f, grad) = x -> (sum(pyconvert(Array, f(x)) .* grad))")
_jl.seval('import Pkg; Pkg.activate("juliatorch", shared=true, io=devnull)')

try:
    gradient = _jl.seval("using ForwardDiff: gradient; gradient")
except:
    _jl.seval("import Pkg; Pkg.add(\"ForwardDiff\")")
    gradient = _jl.seval("using ForwardDiff: gradient; gradient")

# Implementation
class JuliaFunction(_Function):
    """
    PyTorch autograd function for calling Julia functions with automatic differentiation.

    This class wraps a Julia function and computes gradients using ForwardDiff.jl.
    Use the `apply` method to call Julia functions from PyTorch.

    Example:
        >>> from juliatorch import JuliaFunction
        >>> import juliacall, torch
        >>> f = juliacall.Main.seval("f(x) = exp.(-x .^ 2)")
        >>> x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
        >>> result = JuliaFunction.apply(f, x)
    """

    @staticmethod
    def forward(ctx, f, x):
        """
        Forward pass: evaluate the Julia function.

        Args:
            ctx: Context object to save information for backward pass
            f: Julia function to evaluate
            x: PyTorch tensor input

        Returns:
            PyTorch tensor containing the result of f(x)
        """
        ctx.f = f
        ctx.save_for_backward(x)
        np_x = x.detach().numpy()
        jl_res = f(np_x)
        np_res = _np.array(jl_res)
        torch_res = _torch.from_numpy(np_res)
        return torch_res

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using ForwardDiff.jl.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of (None, gradient), where the first None is for the function argument
            and the second element is the gradient with respect to x
        """
        f = ctx.f
        x, = ctx.saved_tensors
        np_x = x.detach().numpy()
        np_grad_output = grad_output.detach().numpy()
        ls = _loss(f, np_grad_output)
        jl_grad = gradient(ls, np_x)
        np_grad = _np.array(jl_grad)
        torch_grad = _torch.from_numpy(np_grad)
        return None, torch_grad

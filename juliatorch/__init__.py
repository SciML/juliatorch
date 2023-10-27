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
    @staticmethod
    def forward(ctx, f, x):
        ctx.f = f
        ctx.save_for_backward(x)
        np_x = x.detach().numpy()
        jl_res = f(np_x)
        np_res = _np.array(jl_res)
        torch_res = _torch.from_numpy(np_res)
        return torch_res

    @staticmethod
    def backward(ctx, grad_output):
        f = ctx.f
        x, = ctx.saved_tensors
        np_x = x.detach().numpy()
        np_grad_output = grad_output.detach().numpy()
        ls = _loss(f, np_grad_output)
        jl_grad = gradient(ls, np_x)
        np_grad = _np.array(jl_grad)
        torch_grad = _torch.from_numpy(np_grad)
        return None, torch_grad

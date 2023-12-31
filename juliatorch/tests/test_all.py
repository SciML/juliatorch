from .. import JuliaFunction
from juliacall import Main as jl
import torch
from torch.autograd import gradcheck
import pytest

def test_gradcheck_of_trivial_function():
    x = torch.randn(3,3,dtype=torch.double,requires_grad=True)
    f = jl.seval("f(x) = 2 .* x")
    f2 = lambda x: f(x) # hack to work around https://github.com/JuliaPy/PythonCall.jl/issues/390

    # Use it by calling the apply method:
    output0 = 2 * x
    output1 = JuliaFunction.apply(f, x)
    output2 = JuliaFunction.apply(f2, x)
    # folks should be able to perform multiplication by 2 precisely
    assert torch.all(torch.eq(output0, output1))
    assert torch.all(torch.eq(output0, output2))

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (f2, x)
    assert gradcheck(JuliaFunction.apply, input, eps=1e-6, atol=1e-4)

# JuliaTorch

[![CI](https://github.com/LilithHafner/juliatorch/workflows/CI/badge.svg)](https://github.com/LilithHafner/juliatorch/actions)

juliatorch lets you convert [Julia](https://julialang.org/) functions to
[PyTorch `autograd.Function`s](https://pytorch.org/docs/stable/autograd.html), automatically
differentiating the julia functions in the process.

If you have any questions, or just want to chat about using the package,
please feel free to chat in [TBD].

For bug reports, feature requests, etc., please submit an issue.

## Installation

To install juliatorch, use Python 3.11 and pip:

```
pip install git+https://github.com/LilithHafner/juliatorch.git
```

## Example usage

```pycon
>>> from juliatorch import JuliaFunction
>>> import juliacall, torch
>>> f = juliacall.Main.seval("f(x) = exp.(-x .^ 2)")
>>> py_f = lambda x: f(x)
>>> x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
>>> JuliaFunction.apply(f, x)
tensor([[0.8583, 0.9999, 0.9712],
        [0.7043, 0.1852, 0.6042],
        [0.9968, 0.8472, 0.9913]], dtype=torch.float64,
       grad_fn=<JuliaFunctionBackward>)
>>> from torch.autograd import gradcheck
>>> gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4)
True
```

<!--
## Collab Notebook Examples

- [Solving the Lorenz equation faster than SciPy+Numba](https://colab.research.google.com/drive/1SQCu1puMQO01i3oMg0TXfa1uf7BqgsEW?usp=sharing)

-->

### Benchmark

## Known Limitations

- Julia functions are falsely reported as subclassess of many abstract base classes,
  including `collections.abc.Iterator`. This causes pytorch to incorrectly treat julia
  functions as iterators. You can work around this by wrapping your Julia functions in
  python functions like this `py_f = lambda x: jl_f(x)`.

- Pytorch doesn't support python 3.12, so neither can this package. Use Python 3.11 instead.

## Testing

Unit tests can be run by [`tox`](http://tox.readthedocs.io).

```sh
tox
```

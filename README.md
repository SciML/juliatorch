# JuliaTorch

[![CI](https://github.com/LilithHafner/juliatorch/workflows/CI/badge.svg)](https://github.com/LilithHafner/juliatorch/actions)

juliatorch lets you convert [Julia](https://julialang.org/) functions to
[PyTorch `autograd.Function`s](https://pytorch.org/docs/stable/autograd.html), automatically
differentiating the julia functions in the process.

If you have any questions, or just want to chat about using the package,
please feel free to chat in [TBD].

For bug reports, feature requests, etc., please submit an issue.

## Installation

...

<!-- To install juliatorch, use pip:

```
pip install juliatorch
```

To install Julia packages required (and Julia if needed) for juliatorch, open up Python
interpreter then run:

```pycon
>>> import juliatorch
>>> juliatorch.install()
``` -->

and you're good!
<!--
## Collab Notebook Examples

- [Solving the Lorenz equation faster than SciPy+Numba](https://colab.research.google.com/drive/1SQCu1puMQO01i3oMg0TXfa1uf7BqgsEW?usp=sharing)

## General Flow

Import and setup the solvers available in *DifferentialEquations.jl* via the command:

```py
from juliatorch import de
```
In case only the solvers available in *OrdinaryDiffEq.jl* are required then use the command:
```py
from juliatorch import ode
```
The general flow for using the package is to follow exactly as would be done
in Julia, except add `de.` or `ode.` in front. Note that `ode.` has lesser loading time and a smaller memory footprint compared to `de.`.
Most of the commands will work without any modification. Thus
[the DifferentialEquations.jl documentation](https://github.com/SciML/DifferentialEquations.jl)
and the [DiffEqTutorials](https://github.com/SciML/DiffEqTutorials.jl)
are the main in-depth documentation for this package. Below we will show how to
translate these docs to Python code. -->

### Benchmark

## GPU Backend Choices

## Known Limitations

- Autodiff does not work on Python functions. When applicable, either define the derivative function
  as a Julia function or set the algorithm to use finite differencing, i.e. `Rodas5(autodiff=false)`.
  All default methods use autodiff.
- Delay differential equations have to use Julia-defined functions otherwise the history function is
  not appropriately typed with the overloads.

## Testing

Unit tests can be run by [`tox`](http://tox.readthedocs.io).

```sh
tox
```

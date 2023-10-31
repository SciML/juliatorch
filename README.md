# JuliaTorch

[![CI](https://github.com/SciML/juliatorch/workflows/CI/badge.svg)](https://github.com/SciML/juliatorch/actions)

juliatorch lets you convert [Julia](https://julialang.org/) functions to
[PyTorch `autograd.Function`s](https://pytorch.org/docs/stable/autograd.html), automatically
differentiating the julia functions in the process.

If you have any questions, or just want to chat about using the package,
please feel free to chat in [TBD].

For bug reports, feature requests, etc., please submit an issue.

## Installation

To install juliatorch, use Python 3.11 and pip:

```
pip install git+https://github.com/SciML/juliatorch.git
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

## Using Julia's differential equation solvers in PyTorch

```python
from juliatorch import JuliaFunction

import juliacall, torch

jl = juliacall.Main.seval

jl('import Pkg')
jl('Pkg.add("DifferentialEquations")')
jl('using DifferentialEquations')

f = jl("""
function f(u0)
    ode_f(u, p, t) = -u
    tspan = (0.0, 1.0)
    prob = ODEProblem(ode_f, u0, tspan)
    sol = DifferentialEquations.solve(prob)
    return sol.u[end]
end""")

print(f(1))
# 0.36787959342751697
print(f(2))
# 0.7357591870280833
print(f(2)/f(1))
# 2.0000000004703966

x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)

print(JuliaFunction.apply(f, x) / x)
# tensor([[0.3679, 0.3679, 0.3679],
#         [0.3679, 0.3679, 0.3679],
#         [0.3679, 0.3679, 0.3679]], dtype=torch.float64, grad_fn=<DivBackward0>)

from torch.autograd import gradcheck
py_f = lambda x: f(x)
print(gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4))
# True (wow, I honestly didn't expect that to work. Up to now
#       I'd only been using trivial Julia functions but it worked
#       on a full differential equation solver on the first try)
```

## Fitting a harmonic oscillator's parameter and initial conditions to match observations

This example uses [`diffeqpy`](https://github.com/SciML/diffeqpy) to solve the differential
equations and [`pytorch`](https://pytorch.org/) to optimize the parameters.

```python
from juliatorch import JuliaFunction
from diffeqpy import de
import juliacall, torch
jl = juliacall.Main.seval

# Define the ODE kernel
def ode_f(du, u, p, t):
    x = u[0]
    v = u[1]
    dx = v
    dv = -p * x
    du[0] = dx
    du[1] = dv

# Use diffeqpy to solve the differential equation for given parameters
def solve(parameters):
    x0, v0, p = parameters
    tspan = (0.0, 1.0)
    # Why not just use `de.ODEProblem`? That would pass gradcheck but fail in the
    # optimization loop. See https://github.com/SciML/juliatorch/issues/10
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(ode_f, [x0, v0], tspan, p)
    return de.solve(prob)

# Extract the desired results
def solve_and_query(parameters):
    sol = solve(parameters)
    return de.hcat(sol(.5), sol(1.0))

print(solve_and_query([1, 2, 3]))
# [1.5274653930969104 0.9791625277649281; -0.023690980408490492 -2.0306945154435274]

x = torch.randn(3, dtype=torch.double, requires_grad=True)
print(JuliaFunction.apply(solve_and_query, x))
# tensor([[-0.4471, -0.3979],
#         [ 0.3155, -0.1103]], dtype=torch.float64,
#        grad_fn=<JuliaFunctionBackward>)

# Verify that autograd through solve_and_query is correct
from torch.autograd import gradcheck
print(gradcheck(JuliaFunction.apply, (solve_and_query, x), eps=1e-6, atol=1e-4))
# True

parameters = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
observations = torch.tensor([[ 0.4301,  0.3577], # Hardcode for consistency
                       [-0.3892, -1.6914]])
weights = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
n_steps = 10000
for learning_rate in [.03, .01, .003]:
    optimizer = torch.optim.SGD([parameters], lr=learning_rate)
    for i in range(n_steps):
        optimizer.zero_grad()
        solution = JuliaFunction.apply(solve_and_query, parameters) # Solve the ODE
        loss = torch.norm(weights * (solution - observations)) # Define the loss function
        loss.backward() # Back-propagate the loss through all differentiable torch variables
        optimizer.step() # Update the parameters using the gradients computed by back-propagation

# It's worth rechecking that the gradient is still accurate because of Goodhart's Law:
print(gradcheck(JuliaFunction.apply, (solve_and_query, parameters), eps=1e-2, atol=1e-2))
# True

print(parameters)
# tensor([ 0.7748, -1.0569, -2.3015], requires_grad=True)
print(loss)
# tensor(0.0195, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)

# Plot the solution
from matplotlib import pyplot as plt
import numpy
def plot(parameters, observations):
    sol = solve(parameters.detach().numpy())
    t = numpy.linspace(0,1,100)
    u = sol(t)
    plt.plot(t,u[0,:],label="simulated x")
    plt.plot(t,u[1,:],label="simulated v")
    plt.plot([.5,1.0],observations[0,:],"o",label="observed x")
    plt.plot([.5],observations[1,0],"o",label="observed v")
    plt.legend()
    plt.show()

plot(parameters, observations)
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

- PyTorch doesn't support python 3.12, so neither can this package. Use Python 3.11 instead.

- The Julia function must accept a single Matrix as input as return a single Matrix as
  output

## Testing

Unit tests can be run by [`tox`](http://tox.readthedocs.io).

```sh
tox
```

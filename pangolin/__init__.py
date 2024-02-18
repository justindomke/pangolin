"""
Pangolin is a library for Bayesian inference, intended to be easy to use.

```python
from pangolin import *
x = normal(0,1)
y = normal(x,1)
print(E(x,y,1))
```

Output: `0.48653358`
"""

from .interface import *
from . import calculate, inference_numpyro
from . import *
from . import transforms

Calculate = calculate.Calculate

# for convenience, set up inference routines
calc = Calculate("numpyro", niter=100000)
sample = calc.sample
E = calc.E
var = calc.var
std = calc.std

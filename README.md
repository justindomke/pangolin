# Pangolin

Pangolin is an early-stage probabilistic inference research project. The focus is to make probabilistic inference **fun**.

![pangolin](pangolin.jpg)

## Installation

See [`INSTALL.md`](INSTALL.md)

## API Docs

All user-facing functions are documented [here](API.md), with examples, in a single 250-ish line file.

## Examples

Simple "probabilistic calculator":

```python
import pangolin as pg
x = pg.normal(0,2)  # x ~ normal(0,2)
y = pg.normal(x,6)  # y ~ normal(x,6)
print(pg.var(x,y,-2.0)) # E[x|y=-2] (close to -0.2)
```

Bayesian inference on the 8-schools model:

```python
import pangolin as pg

# data for 8 schools model
num_schools = 8
observed_effects = [28, 8, -3, 7, -1, 1, 18, 12]
stddevs = [15, 10, 16, 11, 9, 11, 10, 18]

# define model
mu = pg.normal(0,10)
tau = pg.exp(pg.normal(5,1))
theta = [pg.normal(mu,tau) for i in range(num_schools)]
y = [pg.normal(theta[i],stddevs[i]) for i in range(num_schools)]

# do inference
theta_samps = pg.inference.numpyro.sample_flat(theta, y, observed_effects)

# plot results (no pangolin here!)
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.swarmplot(np.array(theta_samps)[:,::50].T,s=2,zorder=0)
plt.xlabel('school')
plt.ylabel('treatment effect')
```

![](8schools_plot.png)

If you're in the market for a PPL, you might want to compare the above to the same (or close) model implemented in other PPLs:


| PPL | Comment                                                                                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Pyro](https://forum.pyro.ai/t/hierarchical-models-and-eight-schools-example/362) | Requires "sample" statements, passing variable names as strings and uses slightly mysterious `plate` construct.                                                                                   |
| [NumPyro](https://github.com/pyro-ppl/numpyro?tab=readme-ov-file#a-simple-example---8-schools) | Requires "sample" statements, passing variable names as strings and uses slightly mysterious `plate` construct.                                                                                   |
| [PyMC](https://github.com/stan-dev/posteriordb/issues/117#issuecomment-567552694) | Pretty good, though requires creating a "model" function and passing variables names as strings. (PyMC is attemptint                                                                              | 
| [JAGS](https://rstudio-pubs-static.s3.amazonaws.com/15236_9bc0cd0966924b139c5162d7d61a2436.html)  | Pretty good, both simple and explicit. We had this in 1991! Requires using a separate language.                                                                                                  |
| [Stan](https://www.maths.usyd.edu.au/u/jormerod/Workshop/Example1/Example1.html#:~:text=school_model3)  | Looks very simple, but uses somewhat subtle batching semantics. Could be written similarly to the JAGS model, just with mandatory declarations of all types/shapes. Requires a separate language. |
| [Tensorflow probability](https://www.tensorflow.org/probability/examples/Eight_Schools) | Legend has it that some find this a wee bit complicated.                                                                                                                                          |

For more examples, take a look at the [demos](demos/).

## Values

(For the current Python interface)

* **Gradual enhancement.** Easy things should be *really* easy. More complex features should be easily discoverable. Steep learning curves should be avoided.
* **Small API surface.** The set of abstractions the user needs to learn should be as small as possible.
* **Graceful interop.** As much as possible, the system should feel lke a natural part of the broader Python NumPy ecosystem, rather than a "new language".
* **Look like math.** As much as possible, calculations should resemble mathematical notation. Exceptions are allowed when algorithmic limitations make this impossible or where standard mathematical notation is ambiguous or bad.

## Long-term goals

Long-term, Pangolin has the following goals:

1. To "decouple" probabilistic models from inference algorithms. It should be possible to write a model *once*, and then perform inference using many inference "backends". (Among other things, this should facilitate benchmarks)
2. To make it easier to experiment with novel inference algorithms that inspect the target distribution. 
3. To support different possible interfaces, in different languages.
4. To be "unopinionated" about how users might specify models, and how inference might be done.


## See also

An earlier version of Pangolin is available and based on much the same ideas, except only supporting JAGS as a backend. It can be found with documentation, in the 
[`pangolin-jags`](pangolin-jags) directory.

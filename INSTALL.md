# Just want to test out pangolin?

If you have [uv](https://docs.astral.sh/uv/) installed, you can test pangolin in a temporary environment by just using `--with pangolin` at the command line.

For example, suppose you have the probabilistic model

```text
m ~ normal(0,1)
s ~ lognormal(0,1)
x ~ normal(m, s)
```

and `x` is observed to have value `x=3.5`. Then you can calculate the posterior mean of `m` and `s` like so:

```console
$ uv run --with pangolin python
Python 3.14.3 
Type "help", "copyright", "credits" or "license" for more information.
>>> from pangolin import interface as pi
>>> from pangolin.blackjax import E
>>> m = pi.normal(0,1)
>>> s = pi.lognormal(0,1)
>>> x = pi.normal(m, s)
>>> E([m, s], x, 3.5)
[Array(0.70249635, dtype=float32), Array(2.8581674, dtype=float32)]
```
# Want to install it "properly"?

Pangolin is [on pypi](https://pypi.org/project/pangolin/) so you can install it using your preferred tool:

* **pip**: `pip install pangolin`
* **uv (project)**: `uv add pangolin`
* **uv (pip interface)**: `uv pip install pangolin`
* **Poetry**: `poetry add pangolin`
* **pipenv**: `pipenv install pangolin`

Note that PyTorch is not installed by default, because PyTorch is huge and often perilous to try install. If you want to use the PyTorch backend you can either just install pytorch yourself (recommended) or you can install it along with pangolin by using one of these commands:

* **pip**: `pip install pangolin[pytorch]`
* **uv (project)**: `uv add pangolin[pytorch]`
* **uv (pip interface)**: `uv pip install pangolin[pytorch]`
* **Poetry**: `poetry add pangolin[pytorch]`
* **pipenv**: `pipenv install pangolin[pytorch]`

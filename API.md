# Pangolin API

Here is the full API for the Python interface, in brief, with examples.

## Creating random variables (basics)

Turn **constants** into random variables with `makerv`.

```python
import pangolin as pg
import numpy as np
x = pg.makerv(1)
y = pg.makerv([1,2,3])
z = pg.makerv(np.eye(3))
```

Create random variables using **basic distributions** with `normal`, `normal_prec`, `cauchy`, `bernoulli`, `bernoulli_logit`, `binomial`, `uniform`, `beta`, `beta_binomial`, `exponential`, `gamma`, `poisson`, `student_t`, `multi_normal`, `categorical`, `multinomial`, and `dirichlet`. In all cases, distribution names and arguments are exactly the same as in Stan.

```python
import pangolin as pg
n = pg.categorical([.1, .2, .7])  
p = pg.uniform(0,1)
x = pg.binomial(n, p)
```

Create random variables from other random variables using **basic deterministic functions** with `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `abs`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`, `cos`, `cosh`, `exp`, `inv_logit`/`expit`/`sigmoid` (all three equivalent), `sin`, `sinh`, `step`, `tan`, `tanh`, `matmul`, `inv`, `softmax`, and `sum`.

```python
import pangolin as pg
import numpy as np
x = pg.makerv(1.5)
y = pg.makerv(2.5)
z = pg.sin(x)
z = pg.add(x,y) # see below on operator overloading
z = pg.pow(x,y) # see below on operator overloading

x = pg.makerv(2*np.eye(3))
y = pg.makerv([1,2,3])
z = pg.inv(y)
z = pg.matmul(x,y) # see below on operator overloading

x = pg.makerv(np.eye(3))
y = pg.sum(x,axis=0)
```

For convenience, you can use the **infix operators** `+`, `-`, `*`, `/`, `**`, and `@`, rather than the functions `add`, `sub`, `mul`, `div`, `pow`, and `matmul`.

```python
import pangolin as pg
import numpy as np
x = pg.makerv(1.5)
y = pg.makerv(2.5)
z = x+y # same as pg.add(x,y)
z = x**y # same as pg.pow(x,y)

x = pg.makerv(2*np.eye(3))
y = pg.makerv([1,2,3])
z = x @ y # same as pg.matmul(x,y)
```

You can **index** random variables using other random variables.

```python
import pangolin as pg
import numpy as np
x = pg.makerv([1.5, 5.0, 100.0])
i = pg.categorical([.1, .2, .7])
y = x[i] # yes! you can index with a random variable!

x = pg.makerv(np.eye(5))
y = x[:,3] # yes! you can slice just like in numpy!
```

## Inference

The basic inference function is `sample(vars, given, vals)`. This draws samples from random variables `vars`, given that some random variables `given` have the values `vals`.

```python
import pangolin as pg
x = pg.normal(0,1)
y = pg.normal(x+1,5)
x_samps = pg.sample(x,y,2) # sample from P(x|y==2) 
```

This will return a 1D numpy array of samples.

If you have multiple random variables, you can just put them into a list, or a tuple.

```python
import pangolin as pg
x = pg.normal(0,1)
y = {'hi': pg.normal(x,1), 'there':pg.normal(x,1)}
z = [pg.normal(x,2), pg.normal(x,3)]
y_samps = pg.sample(y, z, [10.0, 15.0])
```

This will return a dictionary `d` where `d['hi']` and `d['there']` are 1D NumPy arrays. If you like, you can also use any recursive combination, e.g. dicts of lists of tuples of dicts.

You can also do inference using `E` (for expectations), `var` (for variances) and `std` (for standard deviations). These are equivalent to drawing samples and taking the mean.

```python
import pangolin as pg
x = pg.normal(0,1)
y = {'hi': pg.normal(x,1), 'there':pg.normal(x,1)}
z = [pg.normal(x,2), pg.normal(x,3)]
E_y = pg.E(y, z, [10.0, 15.0])
```

For all the inference methods, you can provide an optional `niter` argument. Larger values make inference slower but more accurate.

## Advanced random variable creation

Create efficient **loops** using `slot` and `Loop`. These loops never actually execute in Python, but become efficient fully batched operations at inference time. So this is much more efficient than using, e.g., list comprehensions.

```python
import pangolin as pg
x = pg.slot()
y = pg.slot()
z = pg.slot()
with pg.Loop(5) as i:
    x[i] = pg.normal(0,1)
    with pg.Loop(10) as j:
        y[i,j] = pg.normal(1,2) 
        z[i,j] = pg.normal(x[i] * y[i,j], 1+i+j) # OK to use i,j as numbers
```

Equivalently, you can create **vmapped** random variables using `vmap(fun, in_axes, axis_size)`. This function works exactly the same way as [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) (although without offering as many optional arguments). Under the hood, this is what the loops above do.

```python
import pangolin as pg
# similar to [normal(0,3), normal(1,4), normal(2,5)] but more efficient
x = pg.vmap(pg.normal)([0,1,2],[3,4,5])
x = pg.vmap(pg.normal,0)([0,1,2],[3,4,5]) # equivalent
x = pg.vmap(pg.normal,[0,0])([0,1,2],[3,4,5]) # equivalent

# similar to [normal(0,3), normal(1,3), normal(2,3)] but more efficient
x = pg.vmap(pg.normal,[0,None])([0,1,2],3) 

# similar to [normal(0,3), normal(0,3), normal(0,3)] but more efficient
x = pg.vmap(pg.normal,None)(0,3)
x = pg.vmap(pg.normal,[None,None])(0,3)
```

Create **autoregressive** random variables using `autoregressive(fun, length, in_axes=None)`, where `fun` is some function that takes the "previous" value of a random variable (and possibly some other mapped arguments) and returns a new value.

```python
import pangolin as pg
fun = pg.autoregressive(lambda last: pg.normal(last, 1), 10)
x = fun(0.0) # gaussian random walk starting at 0.0

fun = pg.autoregressive(lambda last, scale: pg.normal(last, scale), 10)
scales = range(10)
x = fun(0.0, scales) # gaussian random walk starting at 0.0 with variable noise 
```

Note that `fun` can only have a single distribution, which must right before the return.

## Inspecting models

You can get a sense of how random variables are **represented internally** using `print_upstream`.

```python
import pangolin as pg
import numpy as np
x = pg.exp(pg.normal(0,1))
y = pg.vmap(pg.normal,[0,None])(np.array([1,2,3]),x)
pg.print_upstream(y)
```

results in

```text
shape | statement
----- | ---------
(3,)  | a = [1 2 3]
()    | b = 0
()    | c = 1
()    | d ~ normal(b,c)
()    | e = exp(d)
(3,)  | f ~ VMap(normal,(0, '∅'),3)(a,e)
```

## Gotchas

Finally, here are a few things to watch out for.

### Implicit constants and `makerv`

Pangolin tries to **cast constants** or NumPy arrays to constant RVs as much as it can for convenience. For example, if you type

```python
import pangolin as pg
x = pg.normal(0,1)
```

this is equivalent to

```python
import pangolin as pg
x = pg.normal(pg.makerv(0),pg.makerv(1))
```

This *usually* works, but Pangolin can't do it if the constant is "on the outside". For example if you do:

```python
import pangolin as pg
x = [1.5,2.5]
i = pg.bernoulli(0.7)
y = x[i] # BAD. DO NOT DO. WILL NOT WORK.
```

_this will not work_, because `x` is a list and doesn't know what to do with a random variable. The solution is to make an explicit call, i.e. do

```python
import pangolin as pg
x = pg.makerv([1.5,2.5]) # fix
i = pg.bernoulli(0.7)
y = x[i] # now OK
```

instead.

### Implicit constants and `Loop`

Similarly, you might try to do something like this:

```python
import pangolin as pg
locs = [0, 1, 5]
x = pg.slot()
with pg.Loop(3) as i:
    x[i] = pg.normal(locs[i], 1) # BAD. DO NOT DO. WILL NOT WORK.
```

This won't work because `locs` is a list and doesn't know what to do with a `Loop` variable `i`. This is fixed by casting `locs` to a random variable before the `Loop` starts:

```python
import pangolin as pg
locs = pg.makerv([0, 1, 5]) # fix
x = pg.slot()
with pg.Loop(3) as i:
    x[i] = pg.normal(locs[i], 1) # now OK
```

### Implicit consts and `vmap`

For somewhat complex reasons `vmap` does not convert lists of scalars to arrays like many methods do. (This is actually impossible to do—the reason is that vmap operates directly on lists, and can't "guess" the user's intent.) So you can't do this:

```python
import pangolin as pg
scales = [1,2,3]
x = pg.vmap(pg.exponential)(scales) # BAD. DO NOT DO. WILL NOT WORK.
```

For this to work, you need to explicitly convert the list to a random variable:

```python
import pangolin as pg
scales = pg.makerv([1,2,3]) # fix
x = pg.vmap(pg.exponential)(scales) # now OK
```
# How is Pangolin organized currently?

The goal of this document is to give an overview of how Pangolin is internally 
organized.

## `CondDist`

The first layer of abstraction in Pangolin is a `CondDist`. This is a "conditional 
distribution" object with these properties:

* `cond_dist.get_shape(*parents_shapes)` — Compute the shape given the shapes of all 
  parents. Like Numpy arrays, shapes are always tuples of integers.
* `cond_dist.is_random` — A boolean representing if the cond_dist is random (a 
  "distribution" or a "deterministic function").
* `cond_dist.__repr__()` — Print a detailed representation.
* `cond_dist.__str__()` — Print a less-detailed but prettier representation.
* `cond_dist.name` — this is a name for the "type" of cond_dist, which is used for 
  *printing only*, never for functionality. It would typically be something like 
  `"Constant"` or `"normal_scale"`.
* `cond_dist._call()` — A shortcut to create `RV`s. `z = cond_dist(x,y)` is equivalent 
  to `z = RV(cond_dist,x,y)`. Note this is the only place that `CondDist` are even 
  aware that `RV`s exist.

`pangolin.interface` creates a large number of `CondDist` *objects*, e.g.:
* `interface. normal_scale`
* `interface.bernoulli`
* `interface.multi_normal_cov`
* `interface.add`
* `interface.pow`
* `interface.matmul`

It's important to note that these don't really *do* anything (other than compute shapes 
and print strings). They are mostly *markers* for inference algorithms to act upon.

`pangolin.interface` also creates some `CondDist` *classes*. These include:
* `interface.Sum` — Represents the sum of an array along a fixed axis. Initialized 
  using the axis (which must be an integer). Has one parent, the array to sum.
* `interface.Index` — Represents indexing into an array. Initialized using a 
  sequence of elements, each of which is either a slice, or `None`. The slices must 
  contain integers only (no `RV`s). Then, the parents of the `CondDist` are (1) 
  `RV` to index and (2) `RV`s representing the indices for all arrays where `None` 
  was initially provided rather than a slice. This is very carefully implemented to 
  make it possible to *exactly* reproduce numpy indexing semantics.
* `interface.VMapDist` — Given an input `CondDist` create a "vmapped" `CondDist`. 
  The new one will take be vectorized over certain inputs (specified by the user) 
  and not over others.
  
Thoughts:
* One could conceivably define `CondDist`s where a boolean value for `is_random` 
  would not exist. For example, consider a function that maps $a \in [0,1]$ to a 
  *pair*, consisting of a Bernoulli variable plus $a^2$.
* We currently provide `interface.CondProb` to represent a conditional probability. 
  Initialized using some `CondDist`, which must be random. Gives a new non-random `CondDist` which is 
  a deterministic function of the inputs and ouputs of the original distribution.  
  (Limited support, and not super confident in this design—can all the PPLs support 
  this?)
* `interface.Mixture` — Initialized using a single `CondDist` and an axis. Creates a 
  mixture distribution taking weights and parameters with one more dimension than 
  the original distribution. (Limited support, should we remove this? Should the 
  user create mixtures, or should they be created by program transformations?)

## `RV`

Fundamentally, a random variable is *very* simple. It is an object that has two things:

* `rv.cond_dist` — the `CondDist` that produces this random variable
* `rv.parents` — a list of parent `RV`s

That's all. It's just an object that lets us put `CondDist`s together into graphs.

On top of that very simple abstraction, `RV`s offer a number of things to make them 
feel more like NumPy arrays:

* `rv.shape` — the shape of the RV. This is computed when the `RV` is created using 
  `cond_dist.get_shape()` and the shapes of the parents.
* `rv.__add__()`, `rv_sub__(), etc — these infix operators are forwarded to 
  `interface.add`, `interface.sub`, etc., described below

## Functions to create `RV`s

In principle, you could create `RV`s by function calls like

`z = RV(interface.normal, x, y)`.

In practice, users would almost never need to do that, because they would use 
`inteface.normal(x,y)` as a shortcut. In addition, a bunch of other methods are 
provided to make it easy to simultaneously create new `CondDist`s and `RV`s:

* `pangolin.vmap(f, in_axes, axis_size)(*args)` — given a function `f` that takes 
  `RV`s and creates new `RV`s, this will create a "vmapped" function. Supports 
  `in_axes` and `axis_size` arguments and tries to support *exactly* the semantics 
  of [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html). E.
  g., `in_axes` can be a nested Python container of axes to map over, each of the 
  elements of `*args` can be a nested Python container, etc.
* `pangolin.plate(N=N)(f)` or `pangolin.plate(*args)(f)` — an alternative interface 
  to `vmap` where you put the arguments first and the function `f` later. Often much 
  more natural with probabilistic models 

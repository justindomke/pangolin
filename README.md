Pangolin is a probabilistic inference research project. To get a quick feel for how 
it works, see these examples:

* [8 schools](demos/demo-8-schools.ipynb)
* [logistic regression](demos/demo-logistic-regression.ipynb)
* [GP regression](demos/demo-GP-regression.ipynb)
* [Item-response theory models](demos/demo-IRT.ipynb)

It has the following goals:

* **Feel like numpy.** Provide an interface for interacting with probability 
  distributions that feels natural to anyone who's played around with 
  numpy. As much as possible, using Pangolin should feel like a natural extension of 
  that ecosystem rather than a new language.
* **Look like math.** Where possible, calculations should resemble mathematical 
  notation. Exceptions are allowed when algorithmic limitations make this impossible or 
  where accepted mathematical notation is bad.
* **Gradual enhancement.** There should be no barrier between using it as a 
  "calculator" for simple one-liner probabilistic calculations and building full 
  custom Bayesian models.
* **Multiple Backends.** We have used lots of different PPLs. Some of our favorites are:
  * [BUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/openbugs)
  * [JAGS](https://mcmc-jags.sourceforge.io/)
  * [Stan](https://mc-stan.org/)
  * [NumPyro](https://num.pyro.ai/)
  * [PyMC](https://www.pymc.io/)
  * [Tensorflow Probability](https://www.tensorflow.org/probability)  
   We want users to be able to write a model *once* and then seamlessly use any of 
    these to actually do inference.  
* **Support program transformations.** Often, users of different PPLs need to 
  manually "transform" their model to get good results. (E.g. manually integrating out 
  discrete latent variables, using non-centered transformations, etc.) We want to 
  provide an "intermediate representation" to make such transformations as easy to 
  define as possible.
* **Support inference algorithm developers.** Existing probabilistic programming 
  languages are often quite "opinionated" about how inference should proceed. This 
  can make it difficult to apply certain kinds of inference algorithms.
  We want to provide a representation that makes it as easy as possible for people 
  to create "new" backends with new inference algorithms.
* **No passing strings as variable names.** We love [NumPyro](https://num.pyro.ai/) 
  and [PyMC](https://www.pymc.io/). But we don't love writing `mean = sample('mean',
  Normal(0, 1))` or `mean = Normal('mean', 0, 1)`. And we *really* don't love 
  programmatically generating variable name strings inside of a for loop. We 
  appreciate that Tensorflow Probability made this mostly optional, but we feel this 
  idea was a mistake and we aren't going to be so compromising.

It remains to be seen to what degree all these goals can be accomplished at the same 
time. (That's what makes this a research project!)

Here are some design principles which we think serve the above goals:

* **Unopinionated.** Probabilistic inference is an open research question. We don't 
  know the best way to do it. So, where possible, we should avoid making assumptions 
  about how inference will be done. It should be easy to write a new program 
  transformation or create a new inference backend without having to conform to some 
  API we invented. 
* **Immutability.** Pangolin enforces that all random variables and distributions 
  are frozen after creation. This is crucial for how much of the code works. We also 
  think it makes it easier to reason about what's happening under the hood and makes 
  it easier to extend.
  * All the *current* inference methods and transformations are also immutable. (You 
    do `xs = sample(x)` rather than something like `mcmc.run(); xs = mcmc.
    get_samples()`.) However, in keeping with being *unopinionated*, there's nothing 
    stopping you from writing an inference method that works in a different way 
    including being mutable.
* **Follow existing conventions.** Wherever possible, we borrow syntax from our 
  favorite libraries:
  * Distribution names and arguments are borrowed from Stan.
  * Random variables are (with great effort) indexed exactly as in NumPy (including 
    full support for advanced indexing with slices and multi-dimensional indices, etc.)
  * For array-valued random variables, the `@` matrix-multiplication operator behaves 
    exactly as in NumPy. 
  * Array operations also behave as in NumPy. For example if `x` is a random 
    variable with `x.shape=(7,2)` you can do `y=pangolin.sum(x,axis=1)` to 
   get a new random variable with `y.shape=(7,)`. (Currently only few operations 
   are supported)
  * Random variables can be placed in PyTrees and manipulated as in Jax.
  * `pangolin.vmap` has exactly the same syntax as `jax.vmap`.

At the moment, we **do not advise** trying to use this code. However, an earlier 
version of Pangolin is available and based on much the same ideas, except only 
supporting JAGS as a backend. It can be found with documentation, in the 
[`pangolin-jags`](pangolin-jags) directory.

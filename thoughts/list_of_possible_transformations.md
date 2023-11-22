# List of program transformations

## Reversing edges with conjugate pairs

Exploit beta-binomial pairs, etc. (Sorta done)

## Reversing edges with discrete latent variables

Consider a model like this:

```python
z = bernoulli(0.75)
x = normal(1 + 2*z, 2)
```

We should be able to do inference on `x` by integrating out `z` with brute-force. 
Basically, we could create a "mixture".

(Question: Can the different PPLs all support mixtures? Stan yes, if you do it 
yourself. Numpyro yes if you do it yourself, I guess. JAGS no.)

(Question: Is this a "program transformation" or part of an "inference algorithm"?)

## Exploiting sufficient statistics

Consider any model where `z` are the parameters of some exponential family and `x` 
is a list of observations from that family:

```python
z = some_dist()
x = vmap(some_exponential_family,None,axis_size=10)(z)
E(z,x,x_obs)
```
We should be able to convert to a model that uses sufficient statistics instead.  

Examples:
* bernoulli -> binomial
* normal -> normal
* exponential -> exponential

## Deterministic RV merging

Consider the model

```python
z = normal(0,1)
x = [normal(0,exp(z)) for i in range(3)]
```

This will create 3 copies of the RV `exp(z)`. That is, the above code is equivalent to

```python
z = normal(0,1)
std = [exp(z) for i in range(3)]
x = [normal(0,s) for s in std]
```

There's no reason to have `std` be three copies of the same deterministic random 
variable. We should have a transformation that gives a model equivalent to

```python
z = normal(0,1)
s = exp(z)
x = [normal(0,s) for i in range(3)]
```

## Vmap splitting

Consider doing something like this:

```python
z = normal(0,1)
x = vmap(normal,None,axis_size=4)(z,1)
x_a = x[:2]
x_b = x[2:] 
E(x_a, x_b, x_b_obs)
```

This will cause trouble, because you're now providing observations for a 
deterministic (`Index`) node. The solution should be to generate code equivalent to 
this:

```python
z = normal(0,1)
x_a = vmap(normal,None,axis_size=2)(z,1)
x_b = vmap(normal,None,axis_size=2)(z,1)
x = concatenate([x_a, x_b]) # optional? need to create concatenate
E(x_a, x_b, x_b_obs)
```

## Joint dist splitting

Consider a model like

```python
mean = np.random.randn(10)
cov = np.random.randn(10)
cov = cov @ cov.T
z = multi_normal_cov(mean,cov)
```

and then doing some kind of inference

```python
z_a = z[:5]
z_b = z[5:] 
E(z_a, z_b, z_b_obs)
```

In JAGS you can *sometimes* get away with this kind of thing. But not in Stan or 
numpyro, which require you to basically define the variables separately. We should 
have a transformation so that `z_b` is sampled first and then `z_a` is drawn 
according to a conditional.

## Unconstraining

Suppose a model happens in a constrained space. We should be able to create another 
model that happens in an unconstrained space.

(Or should this be part of "inference" rather than "transformations"?)

## Deterministic / dist fusing

If someone does:

```python
m = inverse_logit(a)
z = bernoulli(m)
```

We should (probably?) translate that into this: 

```python
z = bernoulli_logit(a)
m = inverse_logit(a)
```

If the user never makes explicit use of `m` then it will just be ignored.
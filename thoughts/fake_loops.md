# Fake loops

Would be nice to make some notation like

```python
x = pangolin.array_rv()
for i in pangolin.range(5):
    x[i] = normal(1,10)
```

Or perhaps just a nicer plate notation like

```python
x = normal(0,1)
y = plate(10)(
    normal(x,1)
)
```

Whatever you do, there seems to be two ways to make vmap RVs:
1. Lazy—Create "abstract" RVs and later come back and "trace" them and create non-abstract ones.
2. Eager—Create vmapped RVs in the first run.

## How could "eager vmap" work?

First off, whenever a CondDist is called with a "non-vectorized" syntax, we need to intercept that call and create a parallel vectorized RV.

Plan:

1. Make CondDist look for most specific RV class involved in parents, call that one.
2. Create a Loop class, which basically just stores an int (the loop length)
3. Create a LoopRV class with constructor `LoopRV(var, dim, loop)`. This should take an existing "full" variable, and create a new variable that simulates a slice along a given dimension.
4. Create a second (effective) constructor `LoopRV(cond_dist, *parents)` copies an existing RV interface. It should vmap or not over all the parents depending on if they are loop_rv or not and create a new VMapDist. That VMapDist will then define a new "full" node that points to the full nodes of the parents. Then it simulates a "slice" of that full node.

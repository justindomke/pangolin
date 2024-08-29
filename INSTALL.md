# How to install pangolin

**Step 1** (Optional but recommended) Install Anaconda and create a new environment, then activate it.

```
conda create -n pangolin-test python=3.12
conda activate pangolin-test
```

(If you are a Python expert, you could also use any other Python virtual environment tools, but Python 3.11 or higher is required.)

**Step 2** Download pangolin source code and extract it. You could do this in either of the following ways:

1.  Go to [`https://github.com/justindomke/pangolin`](https://github.com/justindomke/pangolin) in your browser, then click on "Code" and then "Download zip". Then double-click the file to extract it and past the contents wherever you want to store them.

2. Type the following at the command line (assumes you have `wget` and `unzip` installed):

    ```
    cd path/to/wherever/
    wget https://github.com/justindomke/pangolin/archive/refs/heads/main.zip
    unzip main.zip
    ```

**Step 3**. Install required packages. At the command line, go to wherever you put the pangolin source code, and type the following:

```
pip install -r pangolin-main/requirements.txt
```

**Step 4.** Install pangolin from local directory. At the command line, go to wherever you put the pangolin source code, and type the following:

```
pip install pangolin-main/
```

At this point, technically you're done. But let's test to make sure installation worked.

**Step 5.** Start Python make sure Pangolin works.

```
% python
Python 3.12.4
Type "help", "copyright", "credits" or "license" for more information.
>>> import pangolin as pg
>>> x = pg.normal(0,1)
>>> pg.print_upstream(x)
```

You should see something like this:

```
shape | statement
----- | ---------
()    | a = 0
()    | b = 1
()    | c ~ normal(a,b)
```

**Step 6.** Make sure that Pangolin's dependencies (Jax and NumPyro) are correctly installed. Start Python and do:

```python
import numpyro
import jax

def model():
    x = numpyro.sample('x',numpyro.distributions.Normal(0,1))
    
mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(model), num_warmup=100, num_samples=100)
mcmc.run(jax.random.PRNGKey(0))
```

You should see some output like

```
sample: 100%|██████████| 200/200 [00:01<00:00, 123.72it/s, 3 steps of size 1.40e+00. acc. prob=0.84]
```
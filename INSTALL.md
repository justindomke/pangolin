# How to install pangolin

1. Install Anaconda and create a new environment. (optional but recommended, you can of course use other Python environment tools if you want.)

```
conda create -n pangolin-fresh-install-test python=3.12
```

2. Activate the environment (optional but recommended)

```
conda activate pangolin-fresh-install-test
```

3. Go to whatever directory you want to store the code and download pangolin source code

```
cd path/to/wherever/
wget https://github.com/justindomke/pangolin/archive/refs/heads/main.zip
```

4. Unzip Pagolin source code

```
unzip main.zip
```

5. Install required packages

```
pip install -r pangolin-main/requirements.txt
```

6. Install pangolin from local directory

```
python3 -m pip install pangolin-main/
```

7. Start pangolin and make sure it works.

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

8. Make sure that Jax and NumPyro are correctly installed. Start python and do:

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
from pangolin import ezstan
import numpy as np

def test_stan_1chain():
    code = """
    parameters {
        real x;
    }
    model {
        x ~ normal(0,1);
    }
    """
    [xs] = ezstan.stan(code,['x'],nchains=1,niter=10000)
    assert xs.shape == (10000,)
    assert abs(np.mean(xs) - 0) < .05

def test_stan_4chain():
    code = """
    parameters {
        real x;
    }
    model {
        x ~ normal(0,1);
    }
    """
    [xs] = ezstan.stan(code, ['x'], nchains=4, niter=10000)
    print(f"{xs=}")
    assert xs.shape == (40000,)
    assert abs(np.mean(xs)-0) < .05

def test_stan_data():
    code = """
    data {
        real y;
    }
    parameters {
        real x;
    }
    model {
        x ~ normal(0,1);
        y ~ normal(x,1);
    }
    """
    evidence = {'y':1.}

    xs = ezstan.stan(code,'x',nchains=100,niter=10000,**evidence)

    assert abs(np.mean(xs)-0.5) < .05

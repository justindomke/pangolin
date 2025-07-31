from pangolin import *
# from pangolin.loops import VMapRV, Loop
# from pangolin.automap_simple import automap as roll
# from pangolin.arrays import Array
import pangolin as pg

posterior_name = 'dogs-dogs'
import numpy as np


def getmodel(values):
    n_dogs = values['n_dogs']
    n_trials = values['n_trials']
    y_obs = np.array(values['y'])
    assert y_obs.shape == (n_dogs,n_trials)
    
    n_shock = pg.makerv(np.cumsum(y_obs,axis=1)-y_obs)
    n_avoid = pg.makerv(np.cumsum(1-y_obs,axis=1)-(1-y_obs))

    assert n_shock.shape == (n_dogs, n_trials)
    assert n_avoid.shape == (n_dogs, n_trials)

    beta = pg.slot()
    with pg.Loop(3) as i:
        beta[i] = pg.normal(0.0,100)

    #y = Array((n_dogs,n_trials),check=True)
    y = pg.slot()
    with pg.Loop(n_dogs) as j:
        with pg.Loop(n_trials) as t:
            y[j, t] = pg.bernoulli_logit(beta[0] + beta[1] * n_avoid[j, t] + beta[2] * n_shock[j,t])
            #y[j, t] = pg.bernoulli_logit(0.0)

    #pg.print_upstream(y)

    return {"beta":beta}, y, y_obs
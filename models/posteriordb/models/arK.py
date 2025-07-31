#from pangolin import *
#from pangolin.loops import VMapRV, Loop
#from pangolin.automap_old import automap as roll
#from pangolin.arrays import Array

posterior_name = 'arK-arK'
import pangolin as pg
import numpy as np

def getmodel(values):
    K = values['K']
    T = values['T']
    y_obs = pg.makerv(np.array(values['y']))

    alpha = pg.normal(0,10)
    beta = pg.vmap(pg.normal,None,K)(0,10)
    #sigma = Truncated(cauchy,lo=0)(0,2.5)
    sigma = pg.exp(pg.normal(0,1))

    #y_obs_flipped = pg.makerv(y_obs)

    # # loop version (works)
    # mu = Array(T-K)
    # y = Array(T-K)
    # for t in range(K,T):
    #     mu[t-K] = alpha + sum(beta * np.flip(y_obs[t-K:t]), axis=0)
    #     y[t-K] = normal(mu[t-K], sigma)

    beta_flipped = beta[np.flip(np.arange(K))]

    # loop version with temporary variable (works)
    y = pg.slot()
    #for t in range(K,T):
    with pg.Loop(T-K) as t_minus_K:
        t = t_minus_K+K
        #mu = alpha + pg.sum(beta * np.flip(y_obs[t_minus_K:t]), axis=0)
        # TODO: flip beta
        #mu = alpha + pg.sum(beta_flipped * y_obs[t_minus_K:t], axis=0)
        mu = alpha + pg.sum(beta_flipped, axis=0)
        y[t_minus_K] = pg.normal(mu, sigma)


    # single-line list comprehension (works)
    # y = roll([normal(alpha + sum(beta * np.flip(y_obs[t-K:t]), axis=0), sigma)
    #      for t in range(K,T)])

    # # double list comprehension (works)
    # mu = roll([alpha + sum(beta * np.flip(y_obs[t - K:t]), axis=0) for t in range(K, T)])
    # y = roll([normal(mu_i,sigma) for mu_i in mu])

    # # double list comprehension with mu unrolled (also works)
    # mu = [alpha + sum(beta * np.flip(y_obs[t - K:t]), axis=0) for t in range(K, T)]
    # y = roll([normal(mu_i, sigma) for mu_i in mu])

    #print_ir(y,name_space=6,include_shapes=False)

    #print_upstream(y)

    assert y.shape == y_obs[K:].shape
    return {"alpha":alpha,"beta":beta,"sigma":sigma}, y, y_obs[K:]

# def getmodel_naive(values):
#     K = values['K']
#     T = values['T']
#     y_obs = np.array(values['y'])
#
#     alpha = normal(0,10)
#     beta = [normal(0,10) for k in range(K)]
#     sigma = Truncated(cauchy,lo=0)(0,2.5)
#
#     # # loop version (works)
#     # mu = Array(T-K)
#     # y = Array(T-K)
#     # for t in range(K,T):
#     #     mu[t-K] = alpha + sum(beta * np.flip(y_obs[t-K:t]), axis=0)
#     #     y[t-K] = normal(mu[t-K], sigma)
#
#     # loop version with temporary variable (works)
#     y = [None]*(T-K)
#     for t in range(K,T):
#         #mu = alpha + sum(beta * np.flip(y_obs[t-K:t]), axis=0)
#         mu = alpha
#         for k in range(len(beta)):
#             mu = mu + beta[k] * y_obs[t-k]
#         y[t-K] = normal(mu, sigma)
#
#
#     # single-line list comprehension (works)
#     # y = roll([normal(alpha + sum(beta * np.flip(y_obs[t-K:t]), axis=0), sigma)
#     #      for t in range(K,T)])
#
#     # # double list comprehension (works)
#     # mu = roll([alpha + sum(beta * np.flip(y_obs[t - K:t]), axis=0) for t in range(K, T)])
#     # y = roll([normal(mu_i,sigma) for mu_i in mu])
#
#     # # double list comprehension with mu unrolled (also works)
#     # mu = [alpha + sum(beta * np.flip(y_obs[t - K:t]), axis=0) for t in range(K, T)]
#     # y = roll([normal(mu_i, sigma) for mu_i in mu])
#
#     #print_ir(y,name_space=6,include_shapes=False)
#
#     #print_upstream(y)
#
#     #assert y.shape == y_obs[K:].shape
#     #return {"alpha":alpha,"beta":beta,"sigma":sigma}, y, y_obs[K:]
#     return {"alpha": alpha, "beta": beta, "sigma": sigma}, y, [yi for yi in y_obs[K:]]
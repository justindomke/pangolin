from posteriordb import PosteriorDatabase
import os
import importlib
import numpy as np
from pangolin.interface import *
from pangolin import inference
import jax
import numpyro

def load_data(posterior_name):
    #pdb_path = os.path.join(os.getcwd(), "/test_posteriordb/posteriordb-old/posterior_database")
    pdb_path = os.getcwd() + "/test_posteriordb/posteriordb-old/posterior_database"
    my_pdb = PosteriorDatabase(pdb_path)

    posterior = my_pdb.posterior(posterior_name)
    values = posterior.data.values()
    return values

# disabled for now
# def test_eight_schools_centered():
#     values = load_data("eight_schools-eight_schools_centered")
#     J = values['J']
#     y_obs = np.array(values['y'])
#     sigma = np.array(values['sigma'])
#
#     # TODO: truncated!
#     tau = cauchy(0,5)
#     mu = normal(0,5)
#     theta = vmap(normal,None,J)(mu,tau)
#     y = vmap(normal)(theta,sigma)
#
#     model, var_to_name = inference.numpyro.get_model_flat([tau,mu,theta,y],[y],[y_obs])
#
#     nsamps = 1000
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=nsamps // 2, num_samples=nsamps)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary()
#     out = mcmc.get_samples()
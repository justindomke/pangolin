import numpy as np
import itertools
from . import util
import random
import hashlib

from cmdstanpy import CmdStanModel, write_stan_json

# import pathlib
import os


def stan(code, monitor_vars, *, inits=None, niter=10000, nchains=1, **evidence):
    assert hasattr(monitor_vars, "__iter__")

    newpath = '.stan'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # id = random.randint(0, 10**12)
    h = hashlib.new('sha256')
    h.update(code.encode())
    id = h.hexdigest()[:8]

    stan_exe = os.path.join('.stan', f"model{id}")

    if os.path.exists(stan_exe):
        model = CmdStanModel(exe_file=stan_exe)
    else:

        # write the model to a file
        stan_file = os.path.join('.stan', f"model{id}.stan")

        f = open(stan_file, "w")
        f.write(code)
        f.flush()
        f.close()

        model = CmdStanModel(stan_file=stan_file)

    print(f"{evidence=}")

    # data_file = os.path.join('.stan', f"model{id}.json")
    #
    # write_stan_json(data_file, evidence)
    #
    # import time
    # time.sleep(1)

    fit = model.sample(data=evidence, chains=nchains, iter_warmup=niter, \
                                                              iter_sampling=niter)

    return [fit.stan_variable(v) for v in monitor_vars]

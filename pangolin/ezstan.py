import numpy as np
import itertools
from . import util
import random
import hashlib
import time

# quiet down cmdstanpy
# import logging

# logger = logging.getLogger("cmdstanpy")
# logger.addHandler(logging.NullHandler())
# logger.propagate = False
# logger.setLevel(logging.WARNING)

from cmdstanpy import CmdStanModel, write_stan_json

# import pathlib
import os


def stan(code, monitor_vars, *, inits=None, niter=10000, nchains=1, **evidence):
    assert hasattr(monitor_vars, "__iter__")

    newpath = ".stan"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # id = random.randint(0, 10**12)
    h = hashlib.new("sha256")
    h.update(code.encode())
    id = h.hexdigest()[:8]

    stan_exe = os.path.join(".stan", f"model{id}")

    if os.path.exists(stan_exe):
        model = CmdStanModel(exe_file=stan_exe)
    else:
        # write the model to a file
        stan_file = os.path.join(".stan", f"model{id}.stan")

        f = open(stan_file, "w")
        f.write(code)
        f.flush()
        f.close()

        # unclear if this option actually does anything
        # logger = logging.getLogger("simple_example")
        # logger.setLevel(logging.WARNING)
        t0 = time.time()
        model = CmdStanModel(stan_file=stan_file)
        t1 = time.time()
        print(f"{t1-t0=}")
        # model = CmdStanModel(stan_file=stan_file, cpp_options={"CXXFLAGS": "-O0"})

    # print(f"{evidence=}")

    fit = model.sample(
        data=evidence,
        chains=nchains,
        iter_warmup=niter,
        iter_sampling=niter,
        show_progress=False,
    )

    return [fit.stan_variable(v) for v in monitor_vars]

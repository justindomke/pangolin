# from pangolin.inference.numpyro import (
#     sample,
# )
# from pangolin.inference.numpyro.sampling import sample_flat


def inf_until_match(inf, vars, given, vals, testfun, niter_start=1000, niter_max=100000):
    from time import time

    niter = niter_start
    while niter <= niter_max:
        t0 = time()
        out = inf(vars, given, vals, niter=niter)
        t1 = time()
        #print(f"{niter=} {t1 - t0}")
        if testfun(out):
            assert True
            return
        else:
            niter *= 2
    assert False

def ancestor_sample_until_match(ancestor_sample, vars, testfun, niter_start=1000, niter_max=100_000_000):
    from time import time

    niter = niter_start
    while niter <= niter_max:
        t0 = time()
        out = ancestor_sample(vars, size=niter)
        t1 = time()
        print(f"{niter=} {t1 - t0}")
        if testfun(out):
            print("RETURNING!")
            assert True
            return
        else:
            niter *= 2
    assert False

# import functools

# sample_until_match = functools.partial(inf_until_match, sample)

# def sample_flat_until_match(vars, given, vals, testfun, niter_start=1000, niter_max=100000):
#     new_testfun = lambda stuff: testfun(stuff[0])
#     return inf_until_match(sample_flat, vars, given, vals, new_testfun, niter_start, niter_max)

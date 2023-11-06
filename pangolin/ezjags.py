import numpy as np
import itertools
from . import util
import random

# import pathlib
import os


def vecstring(a):
    """
    Given an input of a=np.array([1,2,3,4])
    returns "c(1,2,3,4)"
    only works for 1-d vectors
    """
    a = np.array(a)
    assert a.ndim == 1
    ret = "c("
    for i, ai in enumerate(a):
        if np.isnan(ai):
            ret += "NA"
        else:
            ret += str(ai)
        if i < a.shape[0] - 1:
            ret += ","
    ret += ")"
    return ret


def read_coda(nchains):
    # first get all the variables

    f = open("CODAindex.txt")
    lines = f.readlines()
    var_start = {}
    var_end = {}
    nsamps = None
    for line in lines:
        varname, start, end = line.split(" ")
        var_start[varname] = int(start) - 1
        var_end[varname] = int(end)

        my_nsamps = var_end[varname] - var_start[varname]
        if nsamps is not None:
            assert nsamps == my_nsamps, "assume # samps is same for all variables"
        nsamps = my_nsamps
    f.close()

    # now, collect things according to arrays
    scalar_vars = []
    vector_lb = {}
    vector_ub = {}
    for var in var_start:
        if "[" in var:
            varname, rest = var.split("[")

            num_str = var.split("[")[1].split("]")[0]
            nums = [int(s) for s in num_str.split(",")]
            if varname not in vector_lb:
                vector_lb[varname] = np.iinfo(np.int32).max + np.zeros(
                    len(nums), dtype=int
                )
                vector_ub[varname] = np.iinfo(np.int32).min + np.zeros(
                    len(nums), dtype=int
                )

            for i, num in enumerate(nums):
                vector_lb[varname][i] = min(vector_lb[varname][i], num)
                vector_ub[varname][i] = max(vector_lb[varname][i], num)
        else:
            scalar_vars.append(var)

    # vector_lb and vector_ub are now arrays with the lowest and highest index seen in each dim
    # this is so we can figure out the size

    data = [
        np.loadtxt("CODAchain" + str(chain + 1) + ".txt") for chain in range(nchains)
    ]
    # add extra dimension if necessary (because simple sample)
    data = [di[None, :] if di.ndim == 1 else di for di in data]

    def read_range(start, end):
        return np.concatenate([data[chain][start:end, 1] for chain in range(nchains)])

    outs = {}
    for var in scalar_vars:
        outs[var] = read_range(var_start[var], var_end[var])
    for var in vector_lb:
        # get names including array indices
        shape = vector_ub[var] - vector_lb[var] + 1
        outs[var] = np.zeros([nsamps, *shape])

        if len(shape) == 1:
            for i in range(shape[0]):
                full_name = var + "[" + str(i + 1) + "]"
                # likely inefficient to keep re-reading all the data...
                stuff = read_range(var_start[full_name], var_end[full_name])
                outs[var][:, i] = stuff
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # TODO: I thiiiinnnkk row-major vs column-major doesn't matter
                    full_name = var + "[" + str(i + 1) + "," + str(j + 1) + "]"
                    stuff = read_range(var_start[full_name], var_end[full_name])
                    outs[var][:, i, j] = stuff
        elif len(shape) == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        full_name = (
                            var
                            + "["
                            + str(i + 1)
                            + ","
                            + str(j + 1)
                            + ","
                            + str(k + 1)
                            + "]"
                        )
                        stuff = read_range(var_start[full_name], var_end[full_name])
                        outs[var][:, i, j, k] = stuff
        else:
            indices = itertools.product(*[range(s) for s in shape])

            for n, index in enumerate(indices):
                index2 = [str(i + 1) for i in index]
                full_name = var + "[" + util.comma_separated(index2, parens=False) + "]"
                stuff = read_range(var_start[full_name], var_end[full_name])

                outs[var][(slice(None, None, None),) + index] = stuff

    return outs


def jags(code, monitor_vars, *, inits=None, niter=10000, nchains=1, **evidence):
    assert hasattr(monitor_vars, "__iter__")
    # assert niter > 1, "niter must be at least 2 to avoid some subtle bug"

    id = random.randint(0, 10**12)

    # write the model to a file
    model_fname = f"model{id}.bug"
    f = open(model_fname, "w")
    f.write(code)
    f.flush()
    f.close()

    # TODO: avoid repetition below and deal with more than 2 dimensions

    # write data
    data_fname = f"data{id}.R"
    f = open(data_fname, "w")
    for var in evidence:
        # assert var.shape==()
        data = np.array(evidence[var])
        if data.shape == ():
            f.write(var + " <- " + str(evidence[var]) + "\n")
        elif len(data.shape) == 1:
            f.write(var + " <- " + vecstring(data) + "\n")
        elif len(data.shape) >= 2:
            # TODO: can't be right: transpose won't work in ND
            f.write(
                "`"
                + var
                + "`"
                + " <- structure("
                + vecstring(data.T.ravel())
                + ", .Dim="
                + vecstring(data.shape)
                + ")\n"
            )
        else:
            raise NotImplementedError("VAR" + str(data.shape))
    f.flush()
    f.close()

    if True:  # inits is not None:
        # write inits
        inits_fname = f"inits{id}.R"
        f = open(inits_fname, "w")
        f.write(".RNG.seed <- " + str(np.random.randint(0, 10000000)) + "\n")
        f.write('.RNG.name <- "base::Wichmann-Hill"\n')
        if inits is not None:
            # for var in inits:
            # assert var.shape==()
            data = np.array(inits[var])
            if data.shape == ():
                f.write(var + " <- " + str(inits[var]) + "\n")
            elif len(data.shape) == 1:
                f.write(var + " <- " + vecstring(data) + "\n")
            elif len(data.shape) >= 2:
                f.write(
                    "`"
                    + var
                    + "`"
                    + " <- structure("
                    + vecstring(data.T.ravel())
                    + ", .Dim="
                    + vecstring(data.shape)
                    + ")\n"
                )
            else:
                raise NotImplementedError("VAR" + str(data.shape))
        f.flush()
        f.close()

    # write a script
    script = f'model in "{model_fname}"\n'
    if "dnormmix" in code:  # load mixture module if needed
        script += "load mix\n"
    if evidence != {}:
        script += f'data in "{data_fname}"\n'
    script += "compile, nchains(" + str(nchains) + ")\n"
    if True:  # inits:
        script += f'parameters in "{inits_fname}"\n'
    script += "initialize\n"
    script += "update " + str(niter) + "\n"
    for var in monitor_vars:
        script += "monitor " + var + "\n"
    script += "update " + str(niter) + "\n"
    script += "coda *\n"
    script_fname = f"script{id}.txt"
    f = open(script_fname, "w")
    f.write(script)
    f.close()

    # actually run it
    import subprocess

    try:
        output = subprocess.check_output(
            ["jags", script_fname], stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        if "is a logical node and cannot be observed" in output:
            raise Exception(
                "JAGS gave 'is a logical node and cannot be observed' error. Usually this is the result of trying to condition in ways that aren't supported."
            ) from None

        if "Node inconsistent with parents" in output:
            raise Exception(
                "JAGS gave 'Node inconsistent with parents' error. Often this is resolved by providing initial values"
            ) from None

        print("JAGS ERROR!")
        print("---------------------------------")
        print(output)
        print("---------------------------------")
        print("JAGS CODE")
        print("---------------------------------")
        print(code)
        print("---------------------------------")

        raise Exception("JAGS error (this is likely triggered by a bug in Pangolin)")
        # return [None for v in monitor_vars]

    # read in variable information
    f = open("CODAindex.txt")
    lines = f.readlines()
    var_start = {}
    var_end = {}
    for line in lines:
        varname, start, end = line.split(" ")
        var_start[varname] = int(start) - 1
        var_end[varname] = int(end)
    f.close()

    outs = read_coda(nchains)  # gets a dict
    # if len(monitor_vars)==1:
    #    return outs[monitor_vars[0]]
    # else:
    #    return [outs[v] for v in monitor_vars]

    # clean up
    for fname in [model_fname, data_fname, inits_fname, script_fname]:
        os.remove(fname)
        # pathlib.Path(fname).unlink() # supposedly better, doesn't work

    return [outs[v] for v in monitor_vars]

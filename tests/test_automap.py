from pangolin.interface import (
    RV,
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    vmap,
    VMapDist,
    viz_upstream,
    print_upstream,
    add,
)
import numpy as np  # type: ignore
from pangolin.loops import Loop, SlicedRV, slice_existing_rv, make_sliced_rv, VMapRV
from pangolin import *
from pangolin.automap_simple import automap, which_slice_kth_arg, is_pointless_rv, UnmergableParentsError
from pangolin.arrays import Array
from pangolin.calculate import Calculate
from pangolin import inference_stan, inference_numpyro_modelbased

def test_paper_example():
    x = exponential(0.5)
    #y = automap([[exponential(i*j) for j in range(1,4)] for i in range(1,3)])
    y = automap([[exponential((10*i+0.1*j)*x) for j in range(3)] for i in range(2)])
    print_upstream(y)

def test_simplify1():
    a = np.array([0,0,0])
    b = np.array([1,1,1])
    x = vmap(add)(a,b)
    new_x = simplify(x)
    # print(x)
    # print(new_x)

    assert new_x.cond_dist == VMapDist(add,(None,None),3)
    assert new_x.parents[0].cond_dist == Constant(0)
    assert new_x.parents[1].cond_dist == Constant(1)

def test_simplify2():
    a = np.array([0,0,0])
    b = np.array([1,0,1])
    x = vmap(add)(a,b)
    new_x = simplify(x)
    # print(x)
    # print(new_x)

    assert new_x.cond_dist == VMapDist(add,(None,0),3)
    assert new_x.parents[0].cond_dist == Constant(0)
    assert new_x.parents[1].cond_dist == Constant([1,0,1])

def test_simplify3():
    a = np.array([[0,1,2],[0,1,2]])
    b = np.array([[3,4,5],[3,4,5]])
    x = vmap(vmap(add))(a,b)
    new_x = simplify(x)
    print(x)
    print(new_x)

    assert new_x.cond_dist == VMapDist(VMapDist(add,(0,0),3),(None,None),2)
    assert new_x.parents[0].cond_dist == Constant([0,1,2])
    assert new_x.parents[1].cond_dist == Constant([3,4,5])

def test_simplify4():
    a = np.array([[0,0,0],[1,1,1]])
    b = np.array([[2,2,2],[3,3,3]])
    x = vmap(vmap(add),1)(a,b)
    new_x = simplify(x)
    print(x)
    print(new_x)

    assert x.cond_dist     == VMapDist(VMapDist(add, (0, 0), 2), (1, 1), 3)
    assert new_x.cond_dist == VMapDist(VMapDist(add,(0,0),2),(None,None),3)
    assert new_x.parents[0].cond_dist == Constant([0,1])
    assert new_x.parents[1].cond_dist == Constant([2,3])

def test_simplify5():
    """
    This should pass but don't have time to improve simplify now.
    (Maybe we should just replace VMapDist with Broadcast?)
    """

    a = np.array([[0,0,0],[1,1,1]])
    b = np.array([[2,2,2],[3,3,3]])
    x = vmap(vmap(add))(a,b)
    new_x = simplify(x)
    print(x)
    print(new_x)

    assert x.cond_dist     == VMapDist(VMapDist(add, (0, 0), 3), (0, 0), 2)
    # TODO: add back!
    #assert new_x.cond_dist == VMapDist(VMapDist(add,(None,None),3),(0,0),2)
    #assert new_x.parents[0].cond_dist == Constant([0,1])
    #assert new_x.parents[1].cond_dist == Constant([2,3])

# good test, just slow
# def test_simpledogs_inference_numpyro():
#     # breaks
#     n_dogs = 2
#     n_trials = 3
#     n_shock = np.array([[1.1, 1.1, 1.1],
#                         [0.1, 0.1, 1.1]])
#     n_avoid = np.array([[0.1, 0.1, 1.1],
#                         [0.1, 1.1, 0.1]])
#     beta = automap(normal(0,100) for i in range(3))
#     y = Array((n_dogs,n_trials))
#     for j in range(n_dogs):
#         for t in range(n_trials):
#             y[j, t] = bernoulli_logit(beta[0] + beta[1] * n_avoid[j, t] + beta[2] * n_shock[j, t])
#     print_upstream(y)
#
#     calc = Calculate(inference_numpyro_modelbased,niter=100)
#     samps = calc.sample(y)


# breaks—does this reveal a bug in stan backend?
# def test_simpledogs_inference_stan():
#     n_dogs = 2
#     n_trials = 3
#     n_shock = np.array([[1.1, 1.1, 1.1],
#                         [0.1, 0.1, 1.1]])
#     n_avoid = np.array([[0.1, 0.1, 1.1],
#                         [0.1, 1.1, 0.1]])
#     beta = automap(normal(0,100) for i in range(3))
#     y = Array((n_dogs,n_trials))
#     for j in range(n_dogs):
#         for t in range(n_trials):
#             y[j, t] = bernoulli_logit(beta[0] + beta[1] * n_avoid[j, t] + beta[2] * n_shock[j, t])
#     print_upstream(y)
#
#     calc = Calculate(inference_stan,niter=100)
#     samps = calc.sample(y)


def test_simpledogs1():
    # breaks
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1.1, 1.1, 1.1],
                        [0.1, 0.1, 1.1]])
    n_avoid = np.array([[0.1, 0.1, 1.1],
                        [0.1, 1.1, 0.1]])
    beta = automap(normal(0,100) for i in range(3))
    y = Array((n_dogs,n_trials))
    beta0 = beta[0]
    beta1 = beta[1]
    beta2 = beta[2]
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta0 + beta1 * n_avoid[j, t] + beta2 * n_shock[j, t])

    # beta0 = beta[0]
    # beta1 = beta[1]
    # beta2 = beta[2]
    # y = Array((n_dogs,n_trials))
    # for j in range(n_dogs):
    #     for t in range(n_trials):
    #         y[j, t] = bernoulli_logit(beta0 + (beta1 * n_avoid[j, t] + beta2 * n_shock[j, t]))
    print_upstream(y)

def test_simpledogs2():
    # breaks
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1, 1, 1],
                        [0, 0, 1]])
    n_avoid = np.array([[0, 0, 1],
                        [0, 1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_avoid[j, t] + n_shock[j, t])
    print_upstream(y)

def test_simpledogs3(): # OK
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1, 1, 1],
                        [0, 0, 1]])
    n_avoid = np.array([[0, 0, 1],
                        [0, 1, 0]])
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(n_avoid[j, t] + n_shock[j, t])
    print_upstream(y)

def test_simpledogs4():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1, 1, 1],
                        [0, 0, 1]])
    n_avoid = np.array([[0, 0, 1],
                        [0, 1, 0]])
    beta = normal(0, 100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(n_avoid[j, t] + n_shock[j, t] + beta)
    print_upstream(y)

def test_simpledogs5():
    # breaks
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0, 0, 1]])
    n_avoid = np.array([[0., 0, 1],
                        [0, 1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_avoid[j, t] + n_shock[j, t])
    print_upstream(y)

def test_simpledogs6():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0, 0, 1]])
    n_avoid = np.array([[0., 0, 1],
                        [0, 1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(n_avoid[j, t] + beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs7():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0.1, 0, 1]])
    n_avoid = np.array([[0.1, 0, 1],
                        [0.2, 1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_avoid[j, t] + n_shock[j, t])
    print_upstream(y)

def test_simpledogs8():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0.1, 0, 1]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs9():
    n_dogs = 2
    n_trials = 3
    n_avoid = np.array([[0.1, 0, 1],
                        [0.2, 1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_avoid[j, t])
    print_upstream(y)

def test_simpledogs10(): # OK
    n_dogs = 6
    n_shock = np.array([1., 1, 1, 0.1, 0, 1])
    beta = normal(0,100)
    y = Array(n_dogs)
    for j in range(n_dogs):
        y[j] = bernoulli_logit(beta + n_shock[j])
    print_upstream(y)

def test_simpledogs11(): # OK
    n_dogs = 3
    n_shock = np.array([1., 1, 1])
    beta = normal(0,100)
    y = Array(n_dogs)
    for j in range(n_dogs):
        y[j] = bernoulli_logit(beta + n_shock[j])
    print_upstream(y)

def test_simpledogs12(): # OK
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [1, 1, 1]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs13(): # OK
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0, 0, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs14():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [0, 0, 1]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs15():
    n_dogs = 2
    n_trials = 3
    n_shock = np.array([[1., 1, 1],
                        [1, 0, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs16():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1., 1],
                        [1, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs17():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1., 1],
                        [2, 0]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs18():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1.1, 2.2],
                        [3.3, 4.4]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

def test_simpledogs19():
    x = np.array([[1.1, 2.2],
                  [3.3, 3.3]])
    beta = normal(0,11)
    y = Array(x.shape)
    for j in range(x.shape[0]):
        for t in range(x.shape[1]):
            y[j, t] = bernoulli_logit(beta + x[j, t])
    print_upstream(y)

def test_simpledogs20():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1.1, 2.2],
                        [1.1, 3.3]])
    beta = normal(0,100)
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = bernoulli_logit(beta + n_shock[j, t])
    print_upstream(y)

# it seems like the problem is having consistency in some rows but not others
# with columns, no such issue!

def test_simpledogs21():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1.1, 2.2],
                        [3.3, 3.3]])
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = add(n_shock[j, t],1)
    print_upstream(y)

def test_simpledogs22():
    n_dogs = 2
    n_trials = 2
    n_shock = np.array([[1.1, 2.2],
                        [3.3, 3.3]])
    y = Array((n_dogs,n_trials))
    for j in range(n_dogs):
        for t in range(n_trials):
            y[j, t] = makerv(n_shock[j, t])
    print_upstream(y)

def test_simpledogs23():
    n_shock = np.array([[1.1, 2.2],
                        [3.3, 3.3]])
    y = automap([[add(n,1) for n in ni] for ni in n_shock])
    print_upstream(y)

def test_simpledogs24():
    x = [[0.5, 0.5],[0.1, 0.9]]
    y = automap([[bernoulli(xij) for xij in xi] for xi in x])
    print_upstream(y)


# def test_dogs():
#     # something like the original case that gave problems
#     n_dogs = 2
#     n_trials = 3
#     n_shock = np.array([[0,0,1], [2,1,2]])
#     n_avoid = np.array([[0,1,0], [1,0,2]])
#
#     beta = automap(normal(0, 100) for i in range(3))
#
#     y = Array((n_dogs, n_trials))
#     for j in range(n_dogs):
#         for t in range(n_trials):
#             y[j, t] = bernoulli_logit(beta[0] + beta[1] * n_avoid[j, t] + beta[2] * n_shock[j, t])
#
# def test_dogs_simplified1():
#     n_dogs = 2
#     n_shock = np.array([1.1,2.2,3.3])
#     n_avoid = np.array([1.1,2.2,3.3])
#
#     beta = automap(normal(0, 100) for i in range(3))
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta[0] + beta[1] * n_avoid[j] + beta[2] * n_shock[j])
#
# def test_dogs_simplified2():
#     n_dogs = 3
#     n_shock = np.array([1.1,2.2,3.3])
#     n_avoid = np.array([1.1,2.2,3.3])
#
#     beta0 = normal(0,100)
#     beta1 = normal(0, 100)
#     beta2 = normal(0, 100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta0 + beta1 * n_avoid[j] + beta2 * n_shock[j])
#
# def test_dogs_simplified3():
#     n_dogs = 3
#     n_shock = np.array([1.1,2.2,3.3])
#     n_avoid = np.array([1.1,2.2,3.3])
#
#     beta = normal(0,100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta + n_avoid[j] + n_shock[j])
#
#
# def test_dogs_simplified4():
#     n_dogs = 3
#     n_avoid = np.array([1.1,2.2,3.3])
#
#     beta = normal(0,100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta + n_avoid[j])
#
# def test_dogs_simplified5():
#     n_dogs = 3
#
#     beta = normal(0,100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta)
#
#
# def test_dogs_simplified6():
#     n_dogs = 3
#     n_avoid = np.random.randn(3)
#
#     beta = normal(0,100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = bernoulli_logit(beta + n_avoid[j])
#
# def test_dogs_simplified7():
#     n_dogs = 3
#     n_avoid = np.random.randn(3)
#
#     beta = normal(0,100)
#
#     y = Array((n_dogs,))
#     for j in range(n_dogs):
#             y[j] = normal(beta + n_avoid[j],1)
#
# def test_dogs_simplified8():
#     n_dogs = 3
#     n_avoid = np.random.randn(3)
#
#     beta = normal(0,100)
#
#     y = automap(normal(beta+n_avoid[j],1) for j in range(3))
#
def test1():
    loc = makerv(0)
    scale = makerv(1)
    y = automap([normal(loc, scale) for i in range(5)])

    print_upstream(y)

    assert y.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert y.parents == (loc, scale)
    for i, yi in enumerate(y):
        assert yi.parents[0] == y
        assert yi.cond_dist == Index(None)
        assert yi.parents[1].cond_dist == Constant(i)


def test2():
    loc = makerv(0)
    scale = makerv([1.0, 2, 3, 4, 5])
    x = [normal(loc, scale[i]) for i in range(5)]
    y = automap(x)
    assert y.cond_dist == VMapDist(normal_scale, (None, 0), 5)
    assert y.parents == (loc, scale)


def test3():
    loc = makerv(0)
    scale = makerv(1)
    x = [[normal(loc, scale) for i in range(5)] for i in range(3)]
    y = automap(x)
    assert y.cond_dist == VMapDist(
        VMapDist(normal_scale, (None, None), 5), (None, None), 3
    )
    assert y.parents == (loc, scale)


def test4():
    y = automap(normal(0, 1) for i in range(5))
    assert y.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert y.parents[0].cond_dist == Constant(0)
    assert y.parents[1].cond_dist == Constant(1)


# this is forbidden for now—can't recurse to random parents
# def test5():
#     y = automap(exp(normal(0, 1)) for i in range(5))
#     print_upstream(y)
#     assert y.cond_dist == VMapDist(exp, (0,), 5)
#     [z] = y.parents
#     assert z.cond_dist == VMapDist(normal_scale, (None, None), 5)
#     assert z.parents[0].cond_dist == Constant(0)
#     assert z.parents[1].cond_dist == Constant(1)


#def test5a():


def test6a():
    # user vectorizes x and y

    x = automap([normal(1.1, 2.2) for i in range(5)])
    y = automap([normal(xi, 3.3) for xi in x])

    print_upstream(y)

    print("---")

    print_upstream(x)

    assert x.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert x.parents[0].cond_dist == Constant(1.1)
    assert x.parents[1].cond_dist == Constant(2.2)

    assert y.cond_dist == VMapDist(normal_scale, (0, None), 5)
    assert y.parents[0] == x
    assert y.parents[1].cond_dist == Constant(3.3)


def test6b():
    # User forgot to automap x
    # result should be exactly the same as above, except x is INDICES INTO y

    x = automap(normal(1.1, 2.2) for i in range(5))
    y = automap(normal(xi, 3.3) for xi in x)

    print_upstream(y)

    print("---")

    print_upstream(x)

    x_vec = y.parents[0]

    assert x_vec.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert x_vec.parents[0].cond_dist == Constant(1.1)
    assert x_vec.parents[1].cond_dist == Constant(2.2)

    assert y.cond_dist == VMapDist(normal_scale, (0, None), 5)
    assert y.parents[0] == x_vec
    assert y.parents[1].cond_dist == Constant(3.3)

    for n in range(5):
        assert isinstance(x[n].cond_dist, Index)
        assert x[n].parents[0] == x_vec
        assert x[n].parents[1].cond_dist == Constant(n)


def test7():
    x = automap([makerv(2 * i) for i in range(5)])
    assert x.cond_dist == Constant([2 * i for i in range(5)])


def test8():
    x = automap([[makerv(2 * i + j) for i in range(2)] for j in range(3)])
    y = makerv([[2 * i + j for i in range(2)] for j in range(3)])
    assert x.cond_dist == y.cond_dist


def test9():
    x = automap([makerv([2 * i + j for i in range(2)]) for j in range(3)])
    y = makerv([[2 * i + j for i in range(2)] for j in range(3)])
    assert x.cond_dist == y.cond_dist


def test10():
    M = 3
    K = 4
    N = 5
    V = 6
    α = np.ones(K)
    β = np.ones(V)
    θ = automap([dirichlet(α) for m in range(M)])
    φ = automap([dirichlet(β) for k in range(K)])
    z = automap([[categorical(θm) for n in range(N)] for θm in θ])
    w = automap([[categorical(φ[z[m, n], :]) for n in range(N)] for m in range(M)])
    print_upstream(w)


def test10b():
    # TODO: to solve this one, need to do "index simplification" — if you index into an index
    # then collapse down into a single index
    M = 3
    K = 4
    N = 5
    V = 6
    α = np.ones(K)
    β = np.ones(V)
    θ = automap([dirichlet(α) for m in range(M)])
    φ = automap([dirichlet(β) for k in range(K)])
    z = automap([[categorical(θm) for n in range(N)] for θm in θ])
    w = automap([[categorical(φ[z[m][n]]) for n in range(N)] for m in range(M)])
    print_upstream(w)


def test10c():
    M = 3
    K = 4
    N = 5
    V = 6
    α = np.ones(K)
    β = np.ones(V)
    θ = automap([dirichlet(α) for m in range(M)])
    φ = automap([dirichlet(β) for k in range(K)])
    z = automap([[categorical(θ[m]) for n in range(N)] for m in range(M)])
    w = automap([[categorical(φ[z[m][n]]) for n in range(N)] for m in range(M)])
    print_upstream(w)

def test11():
    # simplest case where automap creates arrays of ints instead of vectorizing properly
    # x = automap([[normal(0, 1) for i in range(5)] for j in range(3)])
    # y = automap([[normal(x[i, j], 1) for i in range(5)] for j in range(3)])
    loc = makerv(np.random.randn(2, 3))
    x = automap([[normal(loc[i, j], 1) for i in range(2)] for j in range(3)])
    # y = automap([normal(xi, 1) for xi in x])
    print_upstream(x)


# # meaningless when unmapped parents aren't a thing
# def test12():
#     # try a very difficult case
#     x = [normal(1.1, 2.2) for i in range(3)]
#     a = [normal(3.3, 4.4) for j in range(4)]
#     y = [exp(ai) for ai in a]
#     z = automap([[normal(x[i], y[j]) for j in range(4)] for i in range(3)])
#     print_upstream(z)
#
#     assert isinstance(y[0].cond_dist, Index)
#
#     for n, xn in enumerate(x):
#         # check that xi is an index into vectorized x
#         assert xn.cond_dist == Index(None)
#         assert xn.parents[0] == z.parents[0]
#         assert xn.parents[0].cond_dist == VMapDist(normal_scale, (None, None), 3)
#         assert xn.parents[1].cond_dist == Constant(n)
#
#     for n, yn in enumerate(y):
#         # check that xi is an index into vectorized x
#         assert yn.cond_dist == Index(None)
#         assert yn.parents[0] == z.parents[1]
#         assert yn.parents[0].cond_dist == VMapDist(exp, (0,), 4)
#         assert yn.parents[1].cond_dist == Constant(n)
#
#     for n, an in enumerate(a):
#         assert an.cond_dist == Index(None)
#         assert an.parents[0] == z.parents[1].parents[0]
#         assert an.parents[0].cond_dist == VMapDist(normal_scale, (None, None), 4)
#         assert an.parents[1].cond_dist == Constant(n)


def test13():
    # does autovmap work when you have "cycles"?
    x = automap([normal(1.1, 2.2) for i in range(5)])
    loc = automap([xi * 3.3 for xi in x])
    scale = automap([exp(xi) for xi in x])
    y = automap([normal(loci, scalei) for (loci, scalei) in zip(loc, scale)])
    print_upstream(y)

    for i in range(len(scale)):
        assert scale[i].cond_dist == Index(None)
        assert scale[i].parents[0] == y.parents[1]
        assert scale[i].parents[0].cond_dist == VMapDist(exp, (0,), 5)
        assert scale[i].parents[1].cond_dist == Constant(i)

    for i in range(len(loc)):
        assert loc[i].cond_dist == Index(None)
        assert loc[i].parents[0] == y.parents[0]
        assert loc[i].parents[0].cond_dist == VMapDist(mul, (0, None), 5)
        assert loc[i].parents[0].parents[1].cond_dist == Constant(3.3)
        assert loc[i].parents[1].cond_dist == Constant(i)

    for i in range(len(x)):
        assert x[i].cond_dist == Index(None)
        assert x[i].parents[0] == y.parents[0].parents[0]
        assert x[i].parents[0] == y.parents[1].parents[0]
        assert x[i].parents[0].cond_dist == VMapDist(normal_scale, (None, None), 5)
        assert x[i].parents[0].parents[0].cond_dist == Constant(1.1)
        assert x[i].parents[0].parents[1].cond_dist == Constant(2.2)
        assert x[i].parents[1].cond_dist == Constant(i)


def test14():
    x = [normal(1.1, 5.5), normal(2.2, 6.6), normal(3.3, 7.7)]
    y = automap(x)
    print_upstream(y)

    assert y.cond_dist == VMapDist(normal_scale, (0, 0), 3)
    assert y.parents[0].cond_dist == Constant([1.1, 2.2, 3.3])
    assert y.parents[1].cond_dist == Constant([5.5, 6.6, 7.7])


def test15():
    # test a case where the inputs CAN'T be automapped
    try:
        x = automap([normal(1.1, 2.2), exponential(5), normal(3.3, 4.4)])
    except (AssertionError,UnmergableParentsError):
        assert True
        return
    assert False, "failed to raise error as expected"

def test16():
    x = [normal(i, 1) for i in range(3)]
    y = automap(x)
    # print_upstream(y)
    assert y.shape == (3,)


    # incoherent without upstream overwriting
    # for i in range(3):
    #     assert x[i].cond_dist == y[i].cond_dist
    #     assert x[i].parents[0] == y
    #     assert x[i].parents[1].cond_dist == Constant(i)
    #
    # print_upstream(y)
    # print("---")
    # print_upstream(x)
    #
    # (xs, ys) = sample((x, y))
    # for n in range(100):
    #     for i in range(3):
    #         # note how the indices are reversed. This is "correct" because samples
    #         # put a new dimension at the start of each RV and x is a collection of
    #         # hile y is a single RV
    #         assert xs[i][n] == ys[n, i]


def test17():
    x = [[normal(i, 1 + j) for j in range(4)] for i in range(3)]
    y = automap(x)
    # print_upstream(y)
    assert y.shape == (3, 4)

    # print_upstream(y)
    # print("---")
    # print_upstream(x)
    #
    # (xs, ys) = sample((x, y))
    # for n in range(100):
    #     for i in range(3):
    #         for j in range(4):
    #             assert xs[i][j][n] == ys[n, i, j]

def test18():
    val = np.random.randn(5,3)
    x = makerv(val)
    y = automap([xi for xi in x])
    assert isinstance(y,RV)
    print(x)
    print_upstream(y)
    assert y.cond_dist == Constant(val)

def test18b():
    val = np.random.randn(5)
    x = makerv(val)
    y = automap([xi for xi in x])
    print_upstream(y)
    assert isinstance(y,RV)
    print(x)
    assert y.cond_dist == Constant(val)


def test19():
    x = makerv(np.random.randn(5, 3))
    y = automap([x[i] for i in range(5)])
    assert y.cond_dist == x.cond_dist


def test20():
    x = makerv(np.random.randn(5, 3))
    y = automap([x[i,:] for i in range(5)])
    assert y.cond_dist == x.cond_dist


def test21():
    # do a list comprehension that acts like a transpose

    # first confirm how numpy does it
    x0 = np.random.randn(5,3)
    y0 = np.array([x0[:,j] for j in range(3)])
    assert np.all(y0 == x0.T)

    x = makerv(x0)
    y = automap([x[:,j] for j in range(3)])
    assert y.shape == (3,5)
    #assert y.cond_dist == Constant(x0.T)

    print_upstream(y)

    # ys = sample(y)
    # for n in range(100):
    #     assert np.all(ys[n] == x0.T)

def test22():
    x0 = np.random.randn(5, 3)
    y0 = np.array([[x0[i,j] for j in range(3)] for i in range(5)])
    assert np.all(y0 == x0)

    x = makerv(x0)
    y = automap([[x[i,j] for j in range(3)] for i in range(5)])
    print_upstream(y)

def test22b():
    x0 = np.random.randn(5, 3)
    y0 = np.array([[x0[i][j] for j in range(3)] for i in range(5)])
    assert np.all(y0 == x0)

    x = makerv(x0)
    y = automap([[x[i][j] for j in range(3)] for i in range(5)])
    print_upstream(y)


def test_which_slice_kth_arg1():
    d = Index(None,None,None)
    assert which_slice_kth_arg(d, 1) == 0
    assert which_slice_kth_arg(d, 2) == 1
    assert which_slice_kth_arg(d, 3) == 2


def test_which_slice_kth_arg2():
    d = Index(slice(None),None,None)
    assert which_slice_kth_arg(d, 1) == 1
    assert which_slice_kth_arg(d, 2) == 2

def test_which_slice_kth_arg3():
    d = Index(None,slice(None),None)
    assert which_slice_kth_arg(d, 1) == 0
    assert which_slice_kth_arg(d, 2) == 2

def test_which_slice_kth_arg4():
    d = Index(None,None,slice(None))
    assert which_slice_kth_arg(d, 1) == 0
    assert which_slice_kth_arg(d, 2) == 1

def test_which_slice_kth_arg5():
    d = Index(slice(None),slice(None),None)
    assert which_slice_kth_arg(d, 1) == 2

def test_which_slice_kth_arg6():
    d = Index(slice(None),None,slice(None))
    assert which_slice_kth_arg(d, 1) == 1

def test_which_slice_kth_arg7():
    d = Index(None,slice(None),slice(None))
    assert which_slice_kth_arg(d, 1) == 0


def test_pointless_rv1():
    x = makerv(np.random.randn(5))
    y = x[:]
    assert is_pointless_rv(y)


def test_pointless_rv2():
    x = makerv(np.random.randn(5, 3))
    y = x[:, :]
    assert is_pointless_rv(y)


def test_pointless_rv3():
    x = makerv(np.random.randn(5, 3))
    y = x[:, np.arange(3)]
    assert is_pointless_rv(y)


def test_pointless_rv4():
    x = makerv(np.random.randn(5, 3))
    y = x[np.arange(5), :]
    assert is_pointless_rv(y)


def test_pointless_rv5():
    x = makerv(np.random.randn(5, 3))
    y = x[np.arange(5), :]
    assert is_pointless_rv(y)


def test_pointless_rv6():
    x = makerv(np.random.randn(5))
    y = x[[2, 0, 1]]
    assert not is_pointless_rv(y)


def test_pointless_rv7():
    x = makerv(np.random.randn(3))
    y = VMapDist(Index(None), (None, 0))(x, makerv(range(3)))
    assert is_pointless_rv(y)


def test_pointless_rv8():
    x = makerv(np.random.randn(3))
    y = vmap(lambda i: x[i])(makerv(range(3)))
    assert is_pointless_rv(y)


# def test_arrays1():
#     x = Array1D(5)
#     for i in range(5):
#         x[i] = normal(0, 1)
#     print_upstream(x)
#
#
# def test_arrays2():
#     x = Array1D(5)
#     y = Array1D(5)
#     for i in range(5):
#         x[i] = normal(0, 1)
#         y[i] = normal(x[i], 1)
#     print_upstream(y)
#
#
# def test_arrays3():
#     x = Array1D(5)
#     for i in range(5):
#         x[i] = normal(0, i / 2)
#     print_upstream(x)
#
#
# def test_arrays4():
#     x = Array1D(5)
#     y = Array1D(5)
#     for i in range(5):
#         t = normal(0, i / 2)
#         x[i] = t
#         y[i] = x[i] * 2
#     print_upstream(y)
#
#
# def test_arrays5():
#     x = Array((2,))
#     for i in range(2):
#         x[i] = normal(0, 1)
#     print(f"{x.arr=}")
#     print_upstream(x)
#
#
# def test_arrays6():
#     x = Array((2, 3))
#     for i in range(2):
#         for j in range(3):
#             x[i, j] = normal(0, 1)
#     print_upstream(x)
#
#
# def test_arrays7():
#     x = Array((2,))
#     y = Array((2,))
#     for i in range(2):
#         x[i] = normal(0, 1)
#         y[i] = normal(x[i], 1)
#     print_upstream(y)
#
#
# def test_arrays8():
#     x = Array((2,))
#     y = Array((2, 3))
#     for i in range(2):
#         x[i] = normal(0, 1)
#         for j in range(3):
#             y[i, j] = normal(0, 1)
#
#     print_upstream(y)


# def test_arrays9():
#     x = Array((2,))
#     y = Array((2, 3))
#     loc = makerv(0)
#     scale = makerv(1)
#     for i in range(2):
#         x[i] = normal(loc, scale)
#         for j in range(3):
#             y[i, j] = normal(x[i], scale)
#
#     print_upstream(y)
#
#
# def test_arrays10():
#     x = makerv([1.1, 2.2])
#     y = Array((2, 3))
#     scale = makerv(1)
#     for i in range(2):
#         for j in range(3):
#             y[i, j] = normal(x[i], scale)
#
#     print_upstream(y)


def test_arrays11():
    x = makerv([1.1, 2.2, 3.3])
    y = Array((2, 3))
    for i in range(2):
        for j in range(3):
            y[i, j] = normal(0, x[j])

    print_upstream(y)


# def test_arraynd1():
#     x = ArrayND((5,))
#     for i in range(5):
#         x[i] = normal(0, 1)
#     print_upstream(x)
#
#
# def test_arraynd2():
#     x = ArrayND((5,))
#     for i in range(5):
#         x[i] = exp(normal(0, 1))
#     print_upstream(x)
#
#
# def test_arraynd3():
#     x = ArrayND((5,))
#     y = ArrayND((5,))
#     for i in range(5):
#         x[i] = normal(0, 1)
#         y[i] = normal(x[i], 1)
#     print_upstream(y)
#
#
# def test_arraynd4():
#     x = ArrayND((5,))
#     for i in range(5):
#         x[i] = normal(0, i / 2)
#     print_upstream(x)
#
#
# def test_arraynd5():
#     x = ArrayND((5,))
#     y = ArrayND((5,))
#     for i in range(5):
#         t = normal(0, i / 2)
#         x[i] = t
#         y[i] = x[i] * 2
#     print_upstream(y)
#
#
# def test_arraynd6():
#     x = ArrayND((2,))
#     for i in range(2):
#         x[i] = normal(0, 1)
#     print_upstream(x)
#
#
# def test_arraynd7():
#     x = ArrayND((2, 3))
#     for i in range(2):
#         for j in range(3):
#             x[i, j] = normal(0, 1)
#     print_upstream(x)
#
#
# def test_arraynd8():
#     x = ArrayND((2,))
#     y = ArrayND((2,))
#     for i in range(2):
#         x[i] = normal(0, 1)
#         y[i] = normal(x[i], 1)
#     print_upstream(y)
#
#
# def test_arraynd9():
#     x = ArrayND((2,))
#     y = ArrayND((2, 3))
#     for i in range(2):
#         x[i] = normal(0, 1)
#         for j in range(3):
#             y[i, j] = normal(0, 1)
#
#     print_upstream(y)
#
#
# def test_arraynd10():
#     x = ArrayND((2,))
#     y = ArrayND((2, 3))
#     loc = makerv(0)
#     scale = makerv(1)
#     for i in range(2):
#         x[i] = normal(loc, scale)
#         for j in range(3):
#             y[i, j] = normal(x[i], scale)
#
#     print_upstream(y)
#
#
# def test_arraynd11():
#     x = makerv([1.1, 2.2])
#     y = ArrayND((2, 3))
#     scale = makerv(1)
#     for i in range(2):
#         for j in range(3):
#             y[i, j] = normal(x[i], scale)
#
#     print_upstream(y)
#
#
# # def test_arrays11():
# #     x = makerv([1.1, 2.2, 3.3])
# #     y = Array((2, 3))
# #     for i in range(2):
# #         for j in range(3):
# #             y[i, j] = normal(0, x[j])
# #
# #     print_upstream(y)

def test_simplify_constants1():
    x = [makerv(1), makerv(2), makerv(3)]
    y = automap(x)
    assert y.cond_dist == Constant([1,2,3])

def test_simplify_constants2():
    x = [makerv([1,2]), makerv([3,4]), makerv([5,6])]
    y = automap(x)
    assert y.cond_dist == Constant([[1,2],[3,4],[5,6]])



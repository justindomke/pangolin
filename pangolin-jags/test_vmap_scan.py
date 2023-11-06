from pangolin import Given, d, t, I, IID, vmap, scan, recurse, sample, E, P, var, std, cov, corr, jags_code, makerv
import numpy as np
np.set_printoptions(formatter={'float': '{:6.2f}'.format}) # print nicely

# tests to write

# an expectation that just uses 2 samples
# (TODO: 1 somehow causes shape problems)
def fastE(*rvs):
    return E(*rvs,niter=2)

def assert_close(output,expected):
    print(output)
    print(expected)
    assert np.allclose(output,expected), "error: " + str(output) + " vs " + str(expected)

def run_vmap_test(f,expected,*inputs,vec_pars=None):
    x = vmap(f,vec_pars)(*inputs)

    output = fastE(x)
    #expected = np.array(list(map(f,*inputs)))

    print(output)
    print(expected)

    assert_close(output,expected)

def test1():
    "square each of an array of scalars"
    def f(input):
        return input**2    
    inputs = np.random.randn(5)
    expected = inputs**2
    run_vmap_test(f,expected,inputs)

def test2():
    "elementwise multiplication"
    inputs1 = np.random.randn(5)
    inputs2 = np.random.randn(5)
    def f(a,b):
        return a*b
    expected = inputs1*inputs2
    run_vmap_test(f,expected,inputs1,inputs2)

def test3():
    "multiply a vector by a mapped but non-vectorized constant"
    inputs1 = np.random.randn(5)
    input2  = np.random.randn()
    def f(a,b):
        return a*b
    expected = inputs1*input2
    run_vmap_test(f,expected,inputs1,input2,vec_pars=[True,False])

def test4():
    "multiply a vector by a non-mapped constant"
    a = np.random.randn()
    inputs = np.random.randn(5)
    def f(input):
        return input*a
    expected = inputs*a
    run_vmap_test(f,expected,inputs)

def test5():
    "ignore input and just return a constant"
    a = np.random.randn()
    inputs = np.random.randn(5)
    def f(input):
        return a
    expected = a+0*inputs
    run_vmap_test(f,expected,inputs)

def test6():
    "element-wise multiplication plus a constant"
    inputs1 = np.random.randn(5)
    inputs2 = np.random.randn()
    def f(input1,input2):
        return input1*input2, input2
    expected = inputs1*inputs2, inputs2+0*inputs1
    run_vmap_test(f,expected,inputs1,inputs2,vec_pars=[True,False])

def test7():
    "intermediate calculations"

    def f(input1,input2):
        out = input1
        for reps in range(3):
            out = out * input2
        return out

    inputs1 = np.random.randn(5)
    inputs2 = np.random.randn()

    expected = inputs1 * inputs2**3
    run_vmap_test(f,expected,inputs1,inputs2,vec_pars=[True,False])

def test8():
    "recursive vmap to implement matrix multiplication"
    A0 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    B0 = np.array([[5,2,2],[2,5,7],[2,2,0]])
    x0 = np.array([1,2,3])
    y0 = np.array([4,8,10])
    A = makerv(A0)
    B = makerv(B0)
    x = makerv(x0)
    y = makerv(y0)

    elementwise = vmap(lambda a,b:a*b,[True,True])
    inner = lambda a,b: t.sum(elementwise(a,b))
    output = fastE(inner(x,y))
    expected = x0 @ y0
    assert_close(output,expected)

    mat_times_vec = vmap(inner,[True,False])
    output = fastE(mat_times_vec(A,x))
    expected = A0 @ x0
    assert_close(output,expected)

    matT_times_mat = vmap(mat_times_vec,[False,True])
    mat_times_mat  = lambda A,B: matT_times_mat(B.T,A) # vmap doesn't do things in right order
    output = fastE(mat_times_mat(A,B))
    expected = A0 @ B0
    assert_close(output,expected)

def test9():
    "vmap with indexing"
    def f(a,b):
        return a[0]*b

    A = np.random.randn(5,2)
    B = np.random.randn(5,3)
    x = vmap(f)(A,B)
    output = fastE(x)
    expected = A[:,0,None]*B
    assert_close(output,expected)

def test10():
    "vmap with indexing on a thing that is mapped"
    def f(a,b):
        return a[b]
    A = np.random.randn(5,2)
    B = np.random.randint(0,1,size=5)

    x = vmap(f)(A,B)
    output = fastE(x)
    expected = [a[b] for (a,b) in zip(A,B)]
    assert_close(output,expected)

def test11():
    "simple scan that ignores carry and squares input"
    def f(carry,input):
        return 0,input**2
    init = np.array(np.random.randn())
    inputs = np.random.randn(5)
    
    x = scan(f)(init,inputs)
    #x = Scanner(init,inputs,f=f)
    output = fastE(x)
    expected = inputs**2
    assert_close(output,expected)

def test12():
    "scan that implements cumsum"
    def f(carry,input):
        output = carry+input
        return output, output
    init = 0
    inputs = np.random.randn(5)

    x = scan(f)(init,inputs)
    output = fastE(x)
    expected = np.cumsum(inputs)
    assert_close(output,expected)

def test13():
    "cumsum on vectors"
    def f(carry,input):
        print('carry',carry)
        print('input',input)
        return carry+input, carry+input
    init = np.array([0,0])
    inputs = np.random.randn(5,2)

    x = scan(f)(init,inputs)

    output = fastE(x)
    expected = np.cumsum(inputs,axis=0)
    assert_close(output,expected)

# TODO: need multiple inputs to scan

# def test14():
#     "cumsum on indexed parts of vectors"
#     def f(carry,input,index):
#         return carry+input[index], carry+input
#     init = [0,0]
#     inputs = np.random.randn(5,2)
#     indices = np.random.randint(0,1,size=5)

#     x = scan(f)(init,inputs)
#     output = E(x)
#     expected = np.cumsum(inputs,axis=0)
#     assert_close(output,expected)

def test14():
    "2d cumsum"
    
    # first, make a regular cumsum
    def f(carry,input):
        return carry+input, carry+input
    cumsum = lambda inputs: scan(f)(0,inputs)

    # now, cumsum on top of that
    def g(carry,input):
        return carry+cumsum(input), carry+cumsum(input)
    cumsum2 = lambda inputs: scan(g)(np.zeros(inputs.shape[1]),inputs)

    inputs = np.random.randn(5,4)

    x = cumsum2(inputs)
    output = fastE(x)
    expected = np.cumsum(np.cumsum(inputs,axis=0),axis=1)
    assert_close(output,expected)

def test15():
    "scan that implements cumsum on 2nd component"
    def f(carry,input):
        return carry+input[1], carry+input[1]
    init = 0
    inputs = np.random.randn(5,2)

    x = scan(f)(init,inputs)
    output = E(x)
    expected = np.cumsum(inputs[:,1])
    assert_close(output,expected)

def test16():
    "vmap on top of scan, vmap over init but not inputs"

    def f(carry,input):
        return carry+input, carry+input
    def cumsum_start(start,inputs):
        return scan(f)(start,inputs)

    starts = np.random.randn(3)
    inputs = np.random.randn(5)

    x = vmap(cumsum_start,[True,False])(starts,inputs)
    output = fastE(x)
    expected = starts[:,None] + np.cumsum(inputs)[None,:]
    assert_close(output,expected)

def test17():
    "vmap on top of scan, vmap over inputs but not init"

    def f(carry,input):
        return carry+input, carry+input
    def cumsum_start(start,inputs):
        return scan(f)(start,inputs)

    start = np.random.randn()
    inputs = np.random.randn(3,5)

    x = vmap(cumsum_start,[False,True])(start,inputs)
    output = fastE(x)
    expected = start + np.cumsum(inputs,axis=1)
    assert_close(output,expected)


def test18():
    "vmap on top of scan, vmap over both init and inputs"

    def f(carry,input):
        return carry+input, carry+input
    def cumsum_start(start,inputs):
        return scan(f)(start,inputs)

    starts = np.random.randn(3)
    inputs = np.random.randn(3,5)

    x = vmap(cumsum_start,[True,True])(starts,inputs)
    output = E(x)
    expected = starts[:,None] + np.cumsum(inputs,axis=1)
    assert_close(output,expected)

def test19():
    "vmap on top of scan, vmap over implicit input"
    inputs = np.random.randn(5)
    starts = np.random.randn(3)

    def get_cumsum(start):
        def f(carry,input):
            return carry+input, carry+input
        return scan(f)(start,inputs)
    
    x = vmap(get_cumsum)(starts)

    output = E(x)
    expected = starts[:,None] + np.cumsum(inputs)[None,:]
    assert_close(output,expected)

def test20():
    "vmap on top of scan, vmap over implicit input as well as init"
    inputs = np.random.rand(5)
    starts = np.random.rand(3)
    pows   = np.random.rand(3)

    def get_cumsum(start,pow):
        def f(carry,input):
            return carry+input**pow, carry+input**pow
        return scan(f)(start,inputs)
    
    x = vmap(get_cumsum)(starts,pows)

    output = fastE(x)
    expected = starts[:,None] + np.cumsum(inputs[None,:]**pows[:,None],axis=1)
    assert_close(output,expected)

def test21():
    "vmap on top of scan, vmap over implicit input as well as init and explicit input"
    inputs = np.random.rand(3,5)
    starts = np.random.rand(3)
    pows   = np.random.rand(3)

    def get_cumsum(start,pow,input):
        def f(carry,input):
            return carry+input**pow, carry+input**pow
        return scan(f)(start,input)
    
    x = vmap(get_cumsum)(starts,pows,inputs)

    output = E(x)
    expected = starts[:,None] + np.cumsum(inputs**pows[:,None],axis=1)
    assert_close(output,expected)

def test22():
    "Scan with vectors and indexing"
    def f(carry,input):
        output = carry+[1,1.5]
        new_carry = output[0]
        return new_carry,output
    
    inputs = np.random.randn(5)

    x = scan(f)(0,inputs)
    output = fastE(x)
    expected = np.vstack([np.arange(5)+1,np.arange(5)+1.5]).T
    assert_close(output,expected)

#test22()

# TODO: introduce Arrays (we need them!)
# def test23():
#     "Scan that outputs the cumsum and the cumsum of squared values"
#     def f(carry,input):
#         output = carry+[input,input**2]
#         new_carry = output
#         #new_carry = carry+[input,input**2]
#         return new_carry, output
    
#     inputs = np.random.randn(5)

#     x = scan(f)(0,inputs)
#     output = E(x)
#     print('output',output)
#     expected = np.vstack([np.cumsum(inputs), np.cumsum(inputs**2)]).T
#     assert_close(output,expected)

def test23():
    "scan with indexing"
    def f(carry,input):
        output = input[1]
        new_carry = 0
        return new_carry, output
    
    A = np.random.randn(50,2)
    x = scan(f)(0,A)
    output = fastE(x)
    expected = A[:,1]
    assert_close(output,expected)

def test24():
    "scan on top of vmap"

    def f(x):
        return x**2
    
    f2 = vmap(f)

    def g(carry,input):
        return carry+f2(input),t.sin(carry+f2(input))

    inputs = np.random.randn(5,2)

    x = scan(g)(np.zeros(2),inputs)
    output = fastE(x)

    expected = np.sin(np.cumsum(inputs**2,axis=0))

    assert_close(output,expected)
    
def test25():
    "vmap on scan on vmap"

    def f(x):
        return x**2
    
    f2 = vmap(f)

    def g(carry,input):
        return carry+f2(input),t.sin(carry+f2(input))

    def h(input):
        return scan(g)(np.zeros(2),input)

    inputs = np.random.randn(3,5,2)

    x = vmap(h)(inputs)

    output = fastE(x)

    expected = np.sin(np.cumsum(inputs**2,axis=1))

    assert_close(output,expected)

def test26():
    "scan with two parents"
    def f(carry,inputs_a,inputs_b):
        out = carry + inputs_a + inputs_b
        return out,out

    inputs_a = np.random.randn(5)
    inputs_b = np.random.randn(5)

    x = scan(f)(0,inputs_a, inputs_b)

    output = fastE(x)

    expected = np.cumsum(inputs_a) + np.cumsum(inputs_b)

    assert_close(output,expected)

def test27():
    "scan with two parents that happen to be same"
    def f(carry,inputs_a,inputs_b):
        out = carry + inputs_a + inputs_b
        return out,out

    inputs_a = np.random.randn(5)

    x = scan(f)(0,inputs_a, inputs_a)

    output = fastE(x)

    expected = 2*np.cumsum(inputs_a)

    assert_close(output,expected)

#test25()

def test28():
    def f(carry,input):
        return carry+input
    inputs = np.random.randn(5)
    x = recurse(f)(0,inputs)
    output = fastE(x)
    expected = np.cumsum(inputs)
    assert_close(output,expected)

    
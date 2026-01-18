from .base import RVLike, override, makerv, constant
from .indexing import vector_index
from pangolin.ir import print_upstream


def fill_tril(params: RVLike):
    """
    Take a 1D array, fill the lower-triangular entries of a matrix using Pangolin Ops. This does not create a special `Op` but rather reduces to indexing and multiplication.

    Args:
        params: 1D array of length ``N*(N+1)/2``

    Returns:
        2D Lower-triangular matrix with shape ``(N,N)``

    Examples
    --------
    >>> params = constant([1.1,2.2,3.3])
    >>> out = fill_tril(params)
    >>> print_upstream(out)
    shape  | statement
    ------ | ---------
    (3,)   | a = [1.1 2.2 3.3]
    (2, 2) | b = [[0 0] [1 2]]
    (2, 2) | c = index(a,b)
    (2, 2) | d = [[1. 0.] [1. 1.]]
    (2, 2) | e = vmap(vmap(mul, [0, 0], 2), [0, 0], 2)(c,d)
    """

    import numpy as np

    params = makerv(params)
    [M] = params.shape
    N = int((np.sqrt(8 * M + 1) - 1) / 2)  # N*(N+1)/2 == M

    idx_mesh = np.zeros((N, N), dtype=int)

    # Fill the lower triangle with 0, 1, 2, ...
    rows, cols = np.tril_indices(N)
    idx_mesh[rows, cols] = np.arange(M)

    # (3 pre-calc) Create the mask
    mask = np.tril(np.ones((N, N)))

    with override(broadcasting="simple"):
        raw_grid = params[idx_mesh]
        return raw_grid * mask


def extract_tril(L: RVLike):
    """
    Given a square 2D array, return lower-triangular entries as a 1D array. This does not create a special `Op` but instead reduces to multiplication.

    Args:
        L: 2D square array with shape ``(N,N)``

    Returns:
        1D array of length ``N*(N+1)/2``

    Examples
    --------
    >>> L = constant([[1.1,0],[2.2,3.3]])
    >>> l = extract_tril(L)
    >>> print_upstream(l)
    shape  | statement
    ------ | ---------
    (2, 2) | a = [[1.1 0. ] [2.2 3.3]]
    (3,)   | b = [0 1 1]
    (3,)   | c = [0 0 1]
    (3,)   | d = vmap(index, [None, 0, 0], 3)(a,b,c)
    """

    import numpy as np

    L = makerv(L)
    N = L.shape[0]
    M = N * (N + 1) // 2
    rows, cols = np.tril_indices(N)
    with override(broadcasting="simple"):
        l = vector_index(L, rows, cols)
        assert l.shape == (M,)
        return l

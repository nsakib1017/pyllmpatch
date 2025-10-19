# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: numpy-2.2.4/numpy/linalg/_linalg.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-03-16 15:26:29 UTC (1742138789)

"""Lite version of scipy.linalg.\n\nNotes\n-----\nThis module is a lite version of the linalg.py module in SciPy which\ncontains high-level Python interface to the LAPACK library.  The lite\nversion only accesses the following LAPACK functions: dgesv, zgesv,\ndgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,\nzgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.\n"""
__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv', 'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det', 'svd', 'svdvals', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond', 'matrix_rank', 'LinAlgError', 'multi_dot', 'trace', 'diagonal', 'cross', 'outer', 'tensordot', 'matmul', 'matrix_transpose', 'matrix_norm', 'vector_norm', 'vecdot']
import functools
import operator
import warnings
from typing import NamedTuple, Any
from numpy._utils import set_module
from numpy._core import array, asarray, zeros, empty, empty_like, intc, single, double, csingle, cdouble, inexact, complexfloating, newaxis, all, inf, dot, add, multiply, sqrt, sum, isfinite, finfo, errstate, moveaxis, amin, amax, prod, abs, atleast_2d, intp, asanyarray, object_, swapaxes, divide, count_nonzero, isnan, sign, argsort, sort, reciprocal, overrides, diagonal, trace, cross, outer, tensordot, matmul, matrix_transpose, transpose, vecdot
from numpy._globals import _NoValue
from numpy.lib._twodim_base_impl import triu, eye
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from numpy.linalg import _umath_linalg
from numpy._typing import NDArray

class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class QRResult(NamedTuple):
    Q: NDArray[Any]
    R: NDArray[Any]

class SlogdetResult(NamedTuple):
    sign: NDArray[Any]
    logabsdet: NDArray[Any]

class SVDResult(NamedTuple):
    U: NDArray[Any]
    S: NDArray[Any]
    Vh: NDArray[Any]
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy.linalg')
fortran_int = intc

@set_module('numpy.linalg')
class LinAlgError(ValueError):
    """\n    Generic Python-exception-derived object raised by linalg functions.\n\n    General purpose exception class, derived from Python\'s ValueError\n    class, programmatically raised in linalg functions when a Linear\n    Algebra-related condition would prevent further correct execution of the\n    function.\n\n    Parameters\n    ----------\n    None\n\n    Examples\n    --------\n    >>> from numpy import linalg as LA\n    >>> LA.inv(np.zeros((2,2)))\n    Traceback (most recent call last):\n      File \"<stdin>\", line 1, in <module>\n      File \"...linalg.py\", line 350,\n        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))\n      File \"...linalg.py\", line 249,\n        in solve\n        raise LinAlgError(\'Singular matrix\')\n    numpy.linalg.LinAlgError: Singular matrix\n\n    """

def _raise_linalgerror_singular(err, flag):
    raise LinAlgError('Singular matrix')

def _raise_linalgerror_nonposdef(err, flag):
    raise LinAlgError('Matrix is not positive definite')

def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    raise LinAlgError('Eigenvalues did not converge')

def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError('SVD did not converge')

def _raise_linalgerror_lstsq(err, flag):
    raise LinAlgError('SVD did not converge in Linear Least Squares')

def _raise_linalgerror_qr(err, flag):
    raise LinAlgError('Incorrect argument found while performing QR factorization')

def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, '__array_wrap__', new.__array_wrap__)
    return (new, wrap)

def isComplexType(t):
    return issubclass(t, complexfloating)
_real_types_map = {single: single, double: double, csingle: single, cdouble: double}
_complex_types_map = {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}

def _realType(t, default=double):
    return _real_types_map.get(t, default)

def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

def _commonType(*arrays):
    result_type = single
    is_complex = False
    for a in arrays:
        type_ = a.dtype.type
        if issubclass(type_, inexact):
            if isComplexType(type_):
                is_complex = True
            rt = _realType(type_, default=None)
            if rt is double:
                result_type = double
            else:  # inserted
                if rt is None:
                    raise TypeError('array type %s is unsupported in linalg' % (a.dtype.name,))
        else:  # inserted
            result_type = double
    else:  # inserted
        if is_complex:
            result_type = _complex_types_map[result_type]
            return (cdouble, result_type)
        return (double, result_type)

def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ['=', '|']:
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:  # inserted
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    return ret

def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim!= 2:
            raise LinAlgError('%d-dimensional array given. Array must be two-dimensional' % a.ndim)

def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be at least two-dimensional' % a.ndim)

def _assert_stacked_square(*arrays):
    for a in arrays:
        m, n = a.shape[(-2):]
        if m!= n:
            raise LinAlgError('Last 2 dimensions of the array must be square')

def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError('Array must not contain infs or NaNs')

def _is_empty_2d(arr):
    return arr.size == 0 and prod(arr.shape[(-2):]) == 0

def transpose(a):
    """\n    Transpose each matrix in a stack of matrices.\n\n    Unlike np.transpose, this only swaps the last two axes, rather than all of\n    them\n\n    Parameters\n    ----------\n    a : (...,M,N) array_like\n\n    Returns\n    -------\n    aT : (...,N,M) ndarray\n    """  # inserted
    return swapaxes(a, (-1), (-2))

def _tensorsolve_dispatcher(a, b, axes=None):
    return (a, b)

@array_function_dispatch(_tensorsolve_dispatcher)
def tensorsolve(a, b, axes=None):
    """\n    Solve the tensor equation ``a x = b`` for x.\n\n    It is assumed that all indices of `x` are summed over in the product,\n    together with the rightmost indices of `a`, as is done in, for example,\n    ``tensordot(a, x, axes=x.ndim)``.\n\n    Parameters\n    ----------\n    a : array_like\n        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals\n        the shape of that sub-tensor of `a` consisting of the appropriate\n        number of its rightmost indices, and must be such that\n        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be\n        \'square\').\n    b : array_like\n        Right-hand tensor, which can be of any shape.\n    axes : tuple of ints, optional\n        Axes in `a` to reorder to the right, before inversion.\n        If None (default), no reordering is done.\n\n    Returns\n    -------\n    x : ndarray, shape Q\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is singular or not \'square\' (in the above sense).\n\n    See Also\n    --------\n    numpy.tensordot, tensorinv, numpy.einsum\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> a = np.eye(2*3*4)\n    >>> a.shape = (2*3, 4, 2, 3, 4)\n    >>> rng = np.random.default_rng()\n    >>> b = rng.normal(size=(2*3, 4))\n    >>> x = np.linalg.tensorsolve(a, b)\n    >>> x.shape\n    (2, 3, 4)\n    >>> np.allclose(np.tensordot(a, x, axes=3), b)\n    True\n\n    """  # inserted
    a, wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim
    if axes is not None:
        allaxes = list(range(0, an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)
    oldshape = a.shape[-(an - b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k
    if a.size!= prod ** 2:
        raise LinAlgError('Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])')
    a = a.reshape(prod, prod)
    b = b.ravel()
    res = wrap(solve(a, b))
    res.shape = oldshape
    return res

def _solve_dispatcher(a, b):
    return (a, b)

@array_function_dispatch(_solve_dispatcher)
def solve(a, b):
    """\n    Solve a linear matrix equation, or system of linear scalar equations.\n\n    Computes the \"exact\" solution, `x`, of the well-determined, i.e., full\n    rank, linear matrix equation `ax = b`.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Coefficient matrix.\n    b : {(M,), (..., M, K)}, array_like\n        Ordinate or \"dependent variable\" values.\n\n    Returns\n    -------\n    x : {(..., M,), (..., M, K)} ndarray\n        Solution to the system a x = b.  Returned shape is (..., M) if b is\n        shape (M,) and (..., M, K) if b is (..., M, K), where the \"...\" part is\n        broadcasted between a and b.\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is singular or not square.\n\n    See Also\n    --------\n    scipy.linalg.solve : Similar function in SciPy.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The solutions are computed using LAPACK routine ``_gesv``.\n\n    `a` must be square and of full-rank, i.e., all rows (or, equivalently,\n    columns) must be linearly independent; if either is not true, use\n    `lstsq` for the least-squares best \"solution\" of the\n    system/equation.\n\n    .. versionchanged:: 2.0\n\n       The b array is only treated as a shape (M,) column vector if it is\n       exactly 1-dimensional. In all other instances it is treated as a stack\n       of (M, K) matrices. Previously b would be treated as a stack of (M,)\n       vectors if b.ndim was equal to a.ndim - 1.\n\n    References\n    ----------\n    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,\n           FL, Academic Press, Inc., 1980, pg. 22.\n\n    Examples\n    --------\n    Solve the system of equations:\n    ``x0 + 2 * x1 = 1`` and\n    ``3 * x0 + 5 * x1 = 2``:\n\n    >>> import numpy as np\n    >>> a = np.array([[1, 2], [3, 5]])\n    >>> b = np.array([1, 2])\n    >>> x = np.linalg.solve(a, b)\n    >>> x\n    array([-1.,  1.])\n\n    Check that the solution is correct:\n\n    >>> np.allclose(np.dot(a, x), b)\n    True\n\n    """  # inserted
    a, _ = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    b, wrap = _makearray(b)
    t, result_t = _commonType(a, b)
    if b.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:  # inserted
        gufunc = _umath_linalg.solve
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with errstate(call=_raise_linalgerror_singular, invalid='call', over='ignore', divide='ignore', under='ignore'):
        r = gufunc(a, b, signature=signature)
    return wrap(r.astype(result_t, copy=False))

def _tensorinv_dispatcher(a, ind=None):
    return (a,)

@array_function_dispatch(_tensorinv_dispatcher)
def tensorinv(a, ind=2):
    """\n    Compute the \'inverse\' of an N-dimensional array.\n\n    The result is an inverse for `a` relative to the tensordot operation\n    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,\n    ``tensordot(tensorinv(a), a, ind)`` is the \"identity\" tensor for the\n    tensordot operation.\n\n    Parameters\n    ----------\n    a : array_like\n        Tensor to \'invert\'. Its shape must be \'square\', i. e.,\n        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.\n    ind : int, optional\n        Number of first indices that are involved in the inverse sum.\n        Must be a positive integer, default is 2.\n\n    Returns\n    -------\n    b : ndarray\n        `a`\'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is singular or not \'square\' (in the above sense).\n\n    See Also\n    --------\n    numpy.tensordot, tensorsolve\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> a = np.eye(4*6)\n    >>> a.shape = (4, 6, 8, 3)\n    >>> ainv = np.linalg.tensorinv(a, ind=2)\n    >>> ainv.shape\n    (8, 3, 4, 6)\n    >>> rng = np.random.default_rng()\n    >>> b = rng.normal(size=(4, 6))\n    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))\n    True\n\n    >>> a = np.eye(4*6)\n    >>> a.shape = (24, 8, 3)\n    >>> ainv = np.linalg.tensorinv(a, ind=1)\n    >>> ainv.shape\n    (8, 3, 24)\n    >>> rng = np.random.default_rng()\n    >>> b = rng.normal(size=24)\n    >>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))\n    True\n\n    """  # inserted
    a = asarray(a)
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:  # inserted
        raise ValueError('Invalid ind argument.')
    a = a.reshape(prod, (-1))
    ia = inv(a)
    return ia.reshape(*invshape)

def _unary_dispatcher(a):
    return (a,)

@array_function_dispatch(_unary_dispatcher)
def inv(a):
    """\n    Compute the inverse of a matrix.\n\n    Given a square matrix `a`, return the matrix `ainv` satisfying\n    ``a @ ainv = ainv @ a = eye(a.shape[0])``.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Matrix to be inverted.\n\n    Returns\n    -------\n    ainv : (..., M, M) ndarray or matrix\n        Inverse of the matrix `a`.\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is not square or inversion fails.\n\n    See Also\n    --------\n    scipy.linalg.inv : Similar function in SciPy.\n    numpy.linalg.cond : Compute the condition number of a matrix.\n    numpy.linalg.svd : Compute the singular value decomposition of a matrix.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    If `a` is detected to be singular, a `LinAlgError` is raised. If `a` is\n    ill-conditioned, a `LinAlgError` may or may not be raised, and results may\n    be inaccurate due to floating-point errors.\n\n    References\n    ----------\n    .. [1] Wikipedia, \"Condition number\",\n           https://en.wikipedia.org/wiki/Condition_number\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy.linalg import inv\n    >>> a = np.array([[1., 2.], [3., 4.]])\n    >>> ainv = inv(a)\n    >>> np.allclose(a @ ainv, np.eye(2))\n    True\n    >>> np.allclose(ainv @ a, np.eye(2))\n    True\n\n    If a is a matrix object, then the return value is a matrix as well:\n\n    >>> ainv = inv(np.matrix(a))\n    >>> ainv\n    matrix([[-2. ,  1. ],\n            [ 1.5, -0.5]])\n\n    Inverses of several matrices can be computed at once:\n\n    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])\n    >>> inv(a)\n    array([[[-2.  ,  1.  ],\n            [ 1.5 , -0.5 ]],\n           [[-1.25,  0.75],\n            [ 0.75, -0.25]]])\n\n    If a matrix is close to singular, the computed inverse may not satisfy\n    ``a @ ainv = ainv @ a = eye(a.shape[0])`` even if a `LinAlgError`\n    is not raised:\n\n    >>> a = np.array([[2,4,6],[2,0,2],[6,8,14]])\n    >>> inv(a)  # No errors raised\n    array([[-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],\n       [-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],\n       [ 1.12589991e+15,  5.62949953e+14, -5.62949953e+14]])\n    >>> a @ inv(a)\n    array([[ 0.   , -0.5  ,  0.   ],  # may vary\n           [-0.5  ,  0.625,  0.25 ],\n           [ 0.   ,  0.   ,  1.   ]])\n\n    To detect ill-conditioned matrices, you can use `numpy.linalg.cond` to\n    compute its *condition number* [1]_. The larger the condition number, the\n    more ill-conditioned the matrix is. As a rule of thumb, if the condition\n    number ``cond(a) = 10**k``, then you may lose up to ``k`` digits of\n    accuracy on top of what would be lost to the numerical method due to loss\n    of precision from arithmetic methods.\n\n    >>> from numpy.linalg import cond\n    >>> cond(a)\n    np.float64(8.659885634118668e+17)  # may vary\n\n    It is also possible to detect ill-conditioning by inspecting the matrix\'s\n    singular values directly. The ratio between the largest and the smallest\n    singular value is the condition number:\n\n    >>> from numpy.linalg import svd\n    >>> sigma = svd(a, compute_uv=False)  # Do not compute singular vectors\n    >>> sigma.max()/sigma.min()\n    8.659885634118668e+17  # may vary\n\n    """  # inserted
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_singular, invalid='call', over='ignore', divide='ignore', under='ignore'):
        ainv = _umath_linalg.inv(a, signature=signature)
    return wrap(ainv.astype(result_t, copy=False))

def _matrix_power_dispatcher(a, n):
    return (a,)

@array_function_dispatch(_matrix_power_dispatcher)
def matrix_power(a, n):
    """\n    Raise a square matrix to the (integer) power `n`.\n\n    For positive integers `n`, the power is computed by repeated matrix\n    squarings and matrix multiplications. If ``n == 0``, the identity matrix\n    of the same shape as M is returned. If ``n < 0``, the inverse\n    is computed and then raised to the ``abs(n)``.\n\n    .. note:: Stacks of object matrices are not currently supported.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Matrix to be \"powered\".\n    n : int\n        The exponent can be any integer or long integer, positive,\n        negative, or zero.\n\n    Returns\n    -------\n    a**n : (..., M, M) ndarray or matrix object\n        The return value is the same shape and type as `M`;\n        if the exponent is positive or zero then the type of the\n        elements is the same as those of `M`. If the exponent is\n        negative the elements are floating-point.\n\n    Raises\n    ------\n    LinAlgError\n        For matrices that are not square or that (for negative powers) cannot\n        be inverted numerically.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy.linalg import matrix_power\n    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit\n    >>> matrix_power(i, 3) # should = -i\n    array([[ 0, -1],\n           [ 1,  0]])\n    >>> matrix_power(i, 0)\n    array([[1, 0],\n           [0, 1]])\n    >>> matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements\n    array([[ 0.,  1.],\n           [-1.,  0.]])\n\n    Somewhat more sophisticated example\n\n    >>> q = np.zeros((4, 4))\n    >>> q[0:2, 0:2] = -i\n    >>> q[2:4, 2:4] = i\n    >>> q # one of the three quaternion units not equal to 1\n    array([[ 0., -1.,  0.,  0.],\n           [ 1.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  1.],\n           [ 0.,  0., -1.,  0.]])\n    >>> matrix_power(q, 2) # = -np.eye(4)\n    array([[-1.,  0.,  0.,  0.],\n           [ 0., -1.,  0.,  0.],\n           [ 0.,  0., -1.,  0.],\n           [ 0.,  0.,  0., -1.]])\n\n    """  # inserted
    a = asanyarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    try:
        n = operator.index(n)
    except TypeError as e:
        pass  # postinserted
    else:  # inserted
        if a.dtype!= object:
            fmatmul = matmul
        else:  # inserted
            if a.ndim == 2:
                fmatmul = dot
            else:  # inserted
                raise NotImplementedError('matrix_power not supported for stacks of object arrays')
        if n == 0:
            a = empty_like(a)
            a[...] = eye(a.shape[(-2)], dtype=a.dtype)
            return a
        if n < 0:
            a = inv(a)
            n = abs(n)
        if n == 1:
            return a
        if n == 2:
            return fmatmul(a, a)
        if n == 3:
            return fmatmul(fmatmul(a, a), a)
        z = result = None
        while n > 0:
            z = a if z is None else fmatmul(z, z)
            n, bit = divmod(n, 2)
            if bit:
                result = z if result is None else fmatmul(result, z)
        return result
        raise TypeError('exponent must be an integer') from e

def _cholesky_dispatcher(a, /, *, upper=None):
    return (a,)

@array_function_dispatch(_cholesky_dispatcher)
def cholesky(a, /, *, upper=False):
    """\n    Cholesky decomposition.\n\n    Return the lower or upper Cholesky decomposition, ``L * L.H`` or\n    ``U.H * U``, of the square matrix ``a``, where ``L`` is lower-triangular,\n    ``U`` is upper-triangular, and ``.H`` is the conjugate transpose operator\n    (which is the ordinary transpose if ``a`` is real-valued). ``a`` must be\n    Hermitian (symmetric if real-valued) and positive-definite. No checking is\n    performed to verify whether ``a`` is Hermitian or not. In addition, only\n    the lower or upper-triangular and diagonal elements of ``a`` are used.\n    Only ``L`` or ``U`` is actually returned.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Hermitian (symmetric if all elements are real), positive-definite\n        input matrix.\n    upper : bool\n        If ``True``, the result must be the upper-triangular Cholesky factor.\n        If ``False``, the result must be the lower-triangular Cholesky factor.\n        Default: ``False``.\n\n    Returns\n    -------\n    L : (..., M, M) array_like\n        Lower or upper-triangular Cholesky factor of `a`. Returns a matrix\n        object if `a` is a matrix object.\n\n    Raises\n    ------\n    LinAlgError\n       If the decomposition fails, for example, if `a` is not\n       positive-definite.\n\n    See Also\n    --------\n    scipy.linalg.cholesky : Similar function in SciPy.\n    scipy.linalg.cholesky_banded : Cholesky decompose a banded Hermitian\n                                   positive-definite matrix.\n    scipy.linalg.cho_factor : Cholesky decomposition of a matrix, to use in\n                              `scipy.linalg.cho_solve`.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The Cholesky decomposition is often used as a fast way of solving\n\n    .. math:: A \\mathbf{x} = \\mathbf{b}\n\n    (when `A` is both Hermitian/symmetric and positive-definite).\n\n    First, we solve for :math:`\\mathbf{y}` in\n\n    .. math:: L \\mathbf{y} = \\mathbf{b},\n\n    and then for :math:`\\mathbf{x}` in\n\n    .. math:: L^{H} \\mathbf{x} = \\mathbf{y}.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> A = np.array([[1,-2j],[2j,5]])\n    >>> A\n    array([[ 1.+0.j, -0.-2.j],\n           [ 0.+2.j,  5.+0.j]])\n    >>> L = np.linalg.cholesky(A)\n    >>> L\n    array([[1.+0.j, 0.+0.j],\n           [0.+2.j, 1.+0.j]])\n    >>> np.dot(L, L.T.conj()) # verify that L * L.H = A\n    array([[1.+0.j, 0.-2.j],\n           [0.+2.j, 5.+0.j]])\n    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?\n    >>> np.linalg.cholesky(A) # an ndarray object is returned\n    array([[1.+0.j, 0.+0.j],\n           [0.+2.j, 1.+0.j]])\n    >>> # But a matrix object is returned if A is a matrix object\n    >>> np.linalg.cholesky(np.matrix(A))\n    matrix([[ 1.+0.j,  0.+0.j],\n            [ 0.+2.j,  1.+0.j]])\n    >>> # The upper-triangular Cholesky factor can also be obtained.\n    >>> np.linalg.cholesky(A, upper=True)\n    array([[1.-0.j, 0.-2.j],\n           [0.-0.j, 1.-0.j]])\n\n    """  # inserted
    gufunc = _umath_linalg.cholesky_up if upper else _umath_linalg.cholesky_lo
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_nonposdef, invalid='call', over='ignore', divide='ignore', under='ignore'):
        r = gufunc(a, signature=signature)
    return wrap(r.astype(result_t, copy=False))

def _outer_dispatcher(x1, x2):
    return (x1, x2)

@array_function_dispatch(_outer_dispatcher)
def outer(x1, x2, /):
    """\n    Compute the outer product of two vectors.\n\n    This function is Array API compatible. Compared to ``np.outer``\n    it accepts 1-dimensional inputs only.\n\n    Parameters\n    ----------\n    x1 : (M,) array_like\n        One-dimensional input array of size ``N``.\n        Must have a numeric data type.\n    x2 : (N,) array_like\n        One-dimensional input array of size ``M``.\n        Must have a numeric data type.\n\n    Returns\n    -------\n    out : (M, N) ndarray\n        ``out[i, j] = a[i] * b[j]``\n\n    See also\n    --------\n    outer\n\n    Examples\n    --------\n    Make a (*very* coarse) grid for computing a Mandelbrot set:\n\n    >>> rl = np.linalg.outer(np.ones((5,)), np.linspace(-2, 2, 5))\n    >>> rl\n    array([[-2., -1.,  0.,  1.,  2.],\n           [-2., -1.,  0.,  1.,  2.],\n           [-2., -1.,  0.,  1.,  2.],\n           [-2., -1.,  0.,  1.,  2.],\n           [-2., -1.,  0.,  1.,  2.]])\n    >>> im = np.linalg.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))\n    >>> im\n    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],\n           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],\n           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],\n           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],\n           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])\n    >>> grid = rl + im\n    >>> grid\n    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],\n           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],\n           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],\n           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],\n           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])\n\n    An example using a \"vector\" of letters:\n\n    >>> x = np.array([\'a\', \'b\', \'c\'], dtype=object)\n    >>> np.linalg.outer(x, [1, 2, 3])\n    array([[\'a\', \'aa\', \'aaa\'],\n           [\'b\', \'bb\', \'bbb\'],\n           [\'c\', \'cc\', \'ccc\']], dtype=object)\n\n    """  # inserted
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    if x1.ndim!= 1 or x2.ndim!= 1:
        raise ValueError(f'Input arrays must be one-dimensional, but they are x1.ndim={x1.ndim!r} and x2.ndim={x2.ndim!r}.')
    return _core_outer(x1, x2, out=None)

def _qr_dispatcher(a, mode=None):
    return (a,)

@array_function_dispatch(_qr_dispatcher)
def qr(a, mode='reduced'):
    """\n    Compute the qr factorization of a matrix.\n\n    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is\n    upper-triangular.\n\n    Parameters\n    ----------\n    a : array_like, shape (..., M, N)\n        An array-like object with the dimensionality of at least 2.\n    mode : {\'reduced\', \'complete\', \'r\', \'raw\'}, optional, default: \'reduced\'\n        If K = min(M, N), then\n\n        * \'reduced\'  : returns Q, R with dimensions (..., M, K), (..., K, N)\n        * \'complete\' : returns Q, R with dimensions (..., M, M), (..., M, N)\n        * \'r\'        : returns R only with dimensions (..., K, N)\n        * \'raw\'      : returns h, tau with dimensions (..., N, M), (..., K,)\n\n        The options \'reduced\', \'complete, and \'raw\' are new in numpy 1.8,\n        see the notes for more information. The default is \'reduced\', and to\n        maintain backward compatibility with earlier versions of numpy both\n        it and the old default \'full\' can be omitted. Note that array h\n        returned in \'raw\' mode is transposed for calling Fortran. The\n        \'economic\' mode is deprecated.  The modes \'full\' and \'economic\' may\n        be passed using only the first letter for backwards compatibility,\n        but all others must be spelled out. See the Notes for more\n        explanation.\n\n\n    Returns\n    -------\n    When mode is \'reduced\' or \'complete\', the result will be a namedtuple with\n    the attributes `Q` and `R`.\n\n    Q : ndarray of float or complex, optional\n        A matrix with orthonormal columns. When mode = \'complete\' the\n        result is an orthogonal/unitary matrix depending on whether or not\n        a is real/complex. The determinant may be either +/- 1 in that\n        case. In case the number of dimensions in the input array is\n        greater than 2 then a stack of the matrices with above properties\n        is returned.\n    R : ndarray of float or complex, optional\n        The upper-triangular matrix or a stack of upper-triangular\n        matrices if the number of dimensions in the input array is greater\n        than 2.\n    (h, tau) : ndarrays of np.double or np.cdouble, optional\n        The array h contains the Householder reflectors that generate q\n        along with r. The tau array contains scaling factors for the\n        reflectors. In the deprecated  \'economic\' mode only h is returned.\n\n    Raises\n    ------\n    LinAlgError\n        If factoring fails.\n\n    See Also\n    --------\n    scipy.linalg.qr : Similar function in SciPy.\n    scipy.linalg.rq : Compute RQ decomposition of a matrix.\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines ``dgeqrf``, ``zgeqrf``,\n    ``dorgqr``, and ``zungqr``.\n\n    For more information on the qr factorization, see for example:\n    https://en.wikipedia.org/wiki/QR_factorization\n\n    Subclasses of `ndarray` are preserved except for the \'raw\' mode. So if\n    `a` is of type `matrix`, all the return values will be matrices too.\n\n    New \'reduced\', \'complete\', and \'raw\' options for mode were added in\n    NumPy 1.8.0 and the old option \'full\' was made an alias of \'reduced\'.  In\n    addition the options \'full\' and \'economic\' were deprecated.  Because\n    \'full\' was the previous default and \'reduced\' is the new default,\n    backward compatibility can be maintained by letting `mode` default.\n    The \'raw\' option was added so that LAPACK routines that can multiply\n    arrays by q using the Householder reflectors can be used. Note that in\n    this case the returned arrays are of type np.double or np.cdouble and\n    the h array is transposed to be FORTRAN compatible.  No routines using\n    the \'raw\' return are currently exposed by numpy, but some are available\n    in lapack_lite and just await the necessary work.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> a = rng.normal(size=(9, 6))\n    >>> Q, R = np.linalg.qr(a)\n    >>> np.allclose(a, np.dot(Q, R))  # a does equal QR\n    True\n    >>> R2 = np.linalg.qr(a, mode=\'r\')\n    >>> np.allclose(R, R2)  # mode=\'r\' returns the same R as mode=\'full\'\n    True\n    >>> a = np.random.normal(size=(3, 2, 2)) # Stack of 2 x 2 matrices as input\n    >>> Q, R = np.linalg.qr(a)\n    >>> Q.shape\n    (3, 2, 2)\n    >>> R.shape\n    (3, 2, 2)\n    >>> np.allclose(a, np.matmul(Q, R))\n    True\n\n    Example illustrating a common use of `qr`: solving of least squares\n    problems\n\n    What are the least-squares-best `m` and `y0` in ``y = y0 + mx`` for\n    the following data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points\n    and you\'ll see that it should be y0 = 0, m = 1.)  The answer is provided\n    by solving the over-determined matrix equation ``Ax = b``, where::\n\n      A = array([[0, 1], [1, 1], [1, 1], [2, 1]])\n      x = array([[y0], [m]])\n      b = array([[1], [0], [2], [1]])\n\n    If A = QR such that Q is orthonormal (which is always possible via\n    Gram-Schmidt), then ``x = inv(R) * (Q.T) * b``.  (In numpy practice,\n    however, we simply use `lstsq`.)\n\n    >>> A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])\n    >>> A\n    array([[0, 1],\n           [1, 1],\n           [1, 1],\n           [2, 1]])\n    >>> b = np.array([1, 2, 2, 3])\n    >>> Q, R = np.linalg.qr(A)\n    >>> p = np.dot(Q.T, b)\n    >>> np.dot(np.linalg.inv(R), p)\n    array([  1.,   1.])\n\n    """  # inserted
    if mode not in ['reduced', 'complete', 'r', 'raw']:
        if mode in ['f', 'full']:
            msg = 'The \'full\' option is deprecated in favor of \'reduced\'.\nFor backward compatibility let mode default.'
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = 'reduced'
        else:  # inserted
            if mode in ['e', 'economic']:
                msg = 'The \'economic\' option is deprecated.'
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                mode = 'economic'
            else:  # inserted
                raise ValueError(f'Unrecognized mode \'{mode}\'')
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    m, n = a.shape[(-2):]
    t, result_t = _commonType(a)
    a = a.astype(t, copy=True)
    a = _to_native_byte_order(a)
    mn = min(m, n)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_qr, invalid='call', over='ignore', divide='ignore', under='ignore'):
        tau = _umath_linalg.qr_r_raw(a, signature=signature)
    if mode == 'r':
        r = triu(a[..., :mn, :])
        r = r.astype(result_t, copy=False)
        return wrap(r)
    if mode == 'raw':
        q = transpose(a)
        q = q.astype(result_t, copy=False)
        tau = tau.astype(result_t, copy=False)
        return (wrap(q), tau)
    if mode == 'economic':
        a = a.astype(result_t, copy=False)
        return wrap(a)
    if mode == 'complete' and m > n:
        mc = m
        gufunc = _umath_linalg.qr_complete
    else:  # inserted
        mc = mn
        gufunc = _umath_linalg.qr_reduced
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with errstate(call=_raise_linalgerror_qr, invalid='call', over='ignore', divide='ignore', under='ignore'):
        q = gufunc(a, tau, signature=signature)
    r = triu(a[..., :mc, :])
    q = q.astype(result_t, copy=False)
    r = r.astype(result_t, copy=False)
    return QRResult(wrap(q), wrap(r))

@array_function_dispatch(_unary_dispatcher)
def eigvals(a):
    """\n    Compute the eigenvalues of a general matrix.\n\n    Main difference between `eigvals` and `eig`: the eigenvectors aren\'t\n    returned.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        A complex- or real-valued matrix whose eigenvalues will be computed.\n\n    Returns\n    -------\n    w : (..., M,) ndarray\n        The eigenvalues, each repeated according to its multiplicity.\n        They are not necessarily ordered, nor are they necessarily\n        real for real matrices.\n\n    Raises\n    ------\n    LinAlgError\n        If the eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eig : eigenvalues and right eigenvectors of general arrays\n    eigvalsh : eigenvalues of real symmetric or complex Hermitian\n               (conjugate symmetric) arrays.\n    eigh : eigenvalues and eigenvectors of real symmetric or complex\n           Hermitian (conjugate symmetric) arrays.\n    scipy.linalg.eigvals : Similar function in SciPy.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    This is implemented using the ``_geev`` LAPACK routines which compute\n    the eigenvalues and eigenvectors of general square arrays.\n\n    Examples\n    --------\n    Illustration, using the fact that the eigenvalues of a diagonal matrix\n    are its diagonal elements, that multiplying a matrix on the left\n    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose\n    of `Q`), preserves the eigenvalues of the \"middle\" matrix. In other words,\n    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as\n    ``A``:\n\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n    >>> x = np.random.random()\n    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])\n    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])\n    (1.0, 1.0, 0.0)\n\n    Now multiply a diagonal matrix by ``Q`` on one side and\n    by ``Q.T`` on the other:\n\n    >>> D = np.diag((-1,1))\n    >>> LA.eigvals(D)\n    array([-1.,  1.])\n    >>> A = np.dot(Q, D)\n    >>> A = np.dot(A, Q.T)\n    >>> LA.eigvals(A)\n    array([ 1., -1.]) # random\n\n    """  # inserted
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->D'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
        w = _umath_linalg.eigvals(a, signature=signature)
    if not isComplexType(t):
        if all(w.imag == 0):
            w = w.real
            result_t = _realType(result_t)
        else:  # inserted
            result_t = _complexType(result_t)
    return w.astype(result_t, copy=False)

def _eigvalsh_dispatcher(a, UPLO=None):
    return (a,)

@array_function_dispatch(_eigvalsh_dispatcher)
def eigvalsh(a, UPLO='L'):
    """\n    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.\n\n    Main difference from eigh: the eigenvectors are not computed.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        A complex- or real-valued matrix whose eigenvalues are to be\n        computed.\n    UPLO : {\'L\', \'U\'}, optional\n        Specifies whether the calculation is done with the lower triangular\n        part of `a` (\'L\', default) or the upper triangular part (\'U\').\n        Irrespective of this value only the real parts of the diagonal will\n        be considered in the computation to preserve the notion of a Hermitian\n        matrix. It therefore follows that the imaginary part of the diagonal\n        will always be treated as zero.\n\n    Returns\n    -------\n    w : (..., M,) ndarray\n        The eigenvalues in ascending order, each repeated according to\n        its multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If the eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigh : eigenvalues and eigenvectors of real symmetric or complex Hermitian\n           (conjugate symmetric) arrays.\n    eigvals : eigenvalues of general real or complex arrays.\n    eig : eigenvalues and right eigenvectors of general real or complex\n          arrays.\n    scipy.linalg.eigvalsh : Similar function in SciPy.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The eigenvalues are computed using LAPACK routines ``_syevd``, ``_heevd``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n    >>> a = np.array([[1, -2j], [2j, 5]])\n    >>> LA.eigvalsh(a)\n    array([ 0.17157288,  5.82842712]) # may vary\n\n    >>> # demonstrate the treatment of the imaginary part of the diagonal\n    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])\n    >>> a\n    array([[5.+2.j, 9.-2.j],\n           [0.+2.j, 2.-1.j]])\n    >>> # with UPLO=\'L\' this is numerically equivalent to using LA.eigvals()\n    >>> # with:\n    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])\n    >>> b\n    array([[5.+0.j, 0.-2.j],\n           [0.+2.j, 2.+0.j]])\n    >>> wa = LA.eigvalsh(a)\n    >>> wb = LA.eigvals(b)\n    >>> wa; wb\n    array([1., 6.])\n    array([6.+0.j, 1.+0.j])\n\n    """  # inserted
    UPLO = UPLO.upper()
    if UPLO not in ['L', 'U']:
        raise ValueError('UPLO argument must be \'L\' or \'U\'')
    if UPLO == 'L':
        gufunc = _umath_linalg.eigvalsh_lo
    else:  # inserted
        gufunc = _umath_linalg.eigvalsh_up
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->d' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
        w = gufunc(a, signature=signature)
    return w.astype(_realType(result_t), copy=False)

def _convertarray(a):
    t, result_t = _commonType(a)
    a = a.astype(t).T.copy()
    return (a, t, result_t)

@array_function_dispatch(_unary_dispatcher)
def eig(a):
    """\n    Compute the eigenvalues and right eigenvectors of a square array.\n\n    Parameters\n    ----------\n    a : (..., M, M) array\n        Matrices for which the eigenvalues and right eigenvectors will\n        be computed\n\n    Returns\n    -------\n    A namedtuple with the following attributes:\n\n    eigenvalues : (..., M) array\n        The eigenvalues, each repeated according to its multiplicity.\n        The eigenvalues are not necessarily ordered. The resulting\n        array will be of complex type, unless the imaginary part is\n        zero in which case it will be cast to a real type. When `a`\n        is real the resulting eigenvalues will be real (0 imaginary\n        part) or occur in conjugate pairs\n\n    eigenvectors : (..., M, M) array\n        The normalized (unit \"length\") eigenvectors, such that the\n        column ``eigenvectors[:,i]`` is the eigenvector corresponding to the\n        eigenvalue ``eigenvalues[i]``.\n\n    Raises\n    ------\n    LinAlgError\n        If the eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvals : eigenvalues of a non-symmetric array.\n    eigh : eigenvalues and eigenvectors of a real symmetric or complex\n           Hermitian (conjugate symmetric) array.\n    eigvalsh : eigenvalues of a real symmetric or complex Hermitian\n               (conjugate symmetric) array.\n    scipy.linalg.eig : Similar function in SciPy that also solves the\n                       generalized eigenvalue problem.\n    scipy.linalg.schur : Best choice for unitary and other non-Hermitian\n                         normal matrices.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    This is implemented using the ``_geev`` LAPACK routines which compute\n    the eigenvalues and eigenvectors of general square arrays.\n\n    The number `w` is an eigenvalue of `a` if there exists a vector `v` such\n    that ``a @ v = w * v``. Thus, the arrays `a`, `eigenvalues`, and\n    `eigenvectors` satisfy the equations ``a @ eigenvectors[:,i] =\n    eigenvalues[i] * eigenvectors[:,i]`` for :math:`i \\in \\{0,...,M-1\\}`.\n\n    The array `eigenvectors` may not be of maximum rank, that is, some of the\n    columns may be linearly dependent, although round-off error may obscure\n    that fact. If the eigenvalues are all different, then theoretically the\n    eigenvectors are linearly independent and `a` can be diagonalized by a\n    similarity transformation using `eigenvectors`, i.e, ``inv(eigenvectors) @\n    a @ eigenvectors`` is diagonal.\n\n    For non-Hermitian normal matrices the SciPy function `scipy.linalg.schur`\n    is preferred because the matrix `eigenvectors` is guaranteed to be\n    unitary, which is not the case when using `eig`. The Schur factorization\n    produces an upper triangular matrix rather than a diagonal matrix, but for\n    normal matrices only the diagonal of the upper triangular matrix is\n    needed, the rest is roundoff error.\n\n    Finally, it is emphasized that `eigenvectors` consists of the *right* (as\n    in right-hand side) eigenvectors of `a`. A vector `y` satisfying ``y.T @ a\n    = z * y.T`` for some number `z` is called a *left* eigenvector of `a`,\n    and, in general, the left and right eigenvectors of a matrix are not\n    necessarily the (perhaps conjugate) transposes of each other.\n\n    References\n    ----------\n    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,\n    Academic Press, Inc., 1980, Various pp.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n\n    (Almost) trivial example with real eigenvalues and eigenvectors.\n\n    >>> eigenvalues, eigenvectors = LA.eig(np.diag((1, 2, 3)))\n    >>> eigenvalues\n    array([1., 2., 3.])\n    >>> eigenvectors\n    array([[1., 0., 0.],\n           [0., 1., 0.],\n           [0., 0., 1.]])\n\n    Real matrix possessing complex eigenvalues and eigenvectors;\n    note that the eigenvalues are complex conjugates of each other.\n\n    >>> eigenvalues, eigenvectors = LA.eig(np.array([[1, -1], [1, 1]]))\n    >>> eigenvalues\n    array([1.+1.j, 1.-1.j])\n    >>> eigenvectors\n    array([[0.70710678+0.j        , 0.70710678-0.j        ],\n           [0.        -0.70710678j, 0.        +0.70710678j]])\n\n    Complex-valued matrix with real eigenvalues (but complex-valued\n    eigenvectors); note that ``a.conj().T == a``, i.e., `a` is Hermitian.\n\n    >>> a = np.array([[1, 1j], [-1j, 1]])\n    >>> eigenvalues, eigenvectors = LA.eig(a)\n    >>> eigenvalues\n    array([2.+0.j, 0.+0.j])\n    >>> eigenvectors\n    array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary\n           [ 0.70710678+0.j        , -0.        +0.70710678j]])\n\n    Be careful about round-off error!\n\n    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])\n    >>> # Theor. eigenvalues are 1 +/- 1e-9\n    >>> eigenvalues, eigenvectors = LA.eig(a)\n    >>> eigenvalues\n    array([1., 1.])\n    >>> eigenvectors\n    array([[1., 0.],\n           [0., 1.]])\n\n    """  # inserted
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)
    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
        w, vt = _umath_linalg.eig(a, signature=signature)
    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:  # inserted
        result_t = _complexType(result_t)
    vt = vt.astype(result_t, copy=False)
    return EigResult(w.astype(result_t, copy=False), wrap(vt))

@array_function_dispatch(_eigvalsh_dispatcher)
def eigh(a, UPLO='L'):
    """\n    Return the eigenvalues and eigenvectors of a complex Hermitian\n    (conjugate symmetric) or a real symmetric matrix.\n\n    Returns two objects, a 1-D array containing the eigenvalues of `a`, and\n    a 2-D square array or matrix (depending on the input type) of the\n    corresponding eigenvectors (in columns).\n\n    Parameters\n    ----------\n    a : (..., M, M) array\n        Hermitian or real symmetric matrices whose eigenvalues and\n        eigenvectors are to be computed.\n    UPLO : {\'L\', \'U\'}, optional\n        Specifies whether the calculation is done with the lower triangular\n        part of `a` (\'L\', default) or the upper triangular part (\'U\').\n        Irrespective of this value only the real parts of the diagonal will\n        be considered in the computation to preserve the notion of a Hermitian\n        matrix. It therefore follows that the imaginary part of the diagonal\n        will always be treated as zero.\n\n    Returns\n    -------\n    A namedtuple with the following attributes:\n\n    eigenvalues : (..., M) ndarray\n        The eigenvalues in ascending order, each repeated according to\n        its multiplicity.\n    eigenvectors : {(..., M, M) ndarray, (..., M, M) matrix}\n        The column ``eigenvectors[:, i]`` is the normalized eigenvector\n        corresponding to the eigenvalue ``eigenvalues[i]``.  Will return a\n        matrix object if `a` is a matrix object.\n\n    Raises\n    ------\n    LinAlgError\n        If the eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvalsh : eigenvalues of real symmetric or complex Hermitian\n               (conjugate symmetric) arrays.\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays.\n    eigvals : eigenvalues of non-symmetric arrays.\n    scipy.linalg.eigh : Similar function in SciPy (but also solves the\n                        generalized eigenvalue problem).\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``,\n    ``_heevd``.\n\n    The eigenvalues of real symmetric or complex Hermitian matrices are always\n    real. [1]_ The array `eigenvalues` of (column) eigenvectors is unitary and\n    `a`, `eigenvalues`, and `eigenvectors` satisfy the equations ``dot(a,\n    eigenvectors[:, i]) = eigenvalues[i] * eigenvectors[:, i]``.\n\n    References\n    ----------\n    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,\n           FL, Academic Press, Inc., 1980, pg. 222.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n    >>> a = np.array([[1, -2j], [2j, 5]])\n    >>> a\n    array([[ 1.+0.j, -0.-2.j],\n           [ 0.+2.j,  5.+0.j]])\n    >>> eigenvalues, eigenvectors = LA.eigh(a)\n    >>> eigenvalues\n    array([0.17157288, 5.82842712])\n    >>> eigenvectors\n    array([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary\n           [ 0.        +0.38268343j,  0.        -0.92387953j]])\n\n    >>> (np.dot(a, eigenvectors[:, 0]) -\n    ... eigenvalues[0] * eigenvectors[:, 0])  # verify 1st eigenval/vec pair\n    array([5.55111512e-17+0.0000000e+00j, 0.00000000e+00+1.2490009e-16j])\n    >>> (np.dot(a, eigenvectors[:, 1]) -\n    ... eigenvalues[1] * eigenvectors[:, 1])  # verify 2nd eigenval/vec pair\n    array([0.+0.j, 0.+0.j])\n\n    >>> A = np.matrix(a) # what happens if input is a matrix object\n    >>> A\n    matrix([[ 1.+0.j, -0.-2.j],\n            [ 0.+2.j,  5.+0.j]])\n    >>> eigenvalues, eigenvectors = LA.eigh(A)\n    >>> eigenvalues\n    array([0.17157288, 5.82842712])\n    >>> eigenvectors\n    matrix([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary\n            [ 0.        +0.38268343j,  0.        -0.92387953j]])\n\n    >>> # demonstrate the treatment of the imaginary part of the diagonal\n    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])\n    >>> a\n    array([[5.+2.j, 9.-2.j],\n           [0.+2.j, 2.-1.j]])\n    >>> # with UPLO=\'L\' this is numerically equivalent to using LA.eig() with:\n    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])\n    >>> b\n    array([[5.+0.j, 0.-2.j],\n           [0.+2.j, 2.+0.j]])\n    >>> wa, va = LA.eigh(a)\n    >>> wb, vb = LA.eig(b)\n    >>> wa\n    array([1., 6.])\n    >>> wb\n    array([6.+0.j, 1.+0.j])\n    >>> va\n    array([[-0.4472136 +0.j        , -0.89442719+0.j        ], # may vary\n           [ 0.        +0.89442719j,  0.        -0.4472136j ]])\n    >>> vb\n    array([[ 0.89442719+0.j       , -0.        +0.4472136j],\n           [-0.        +0.4472136j,  0.89442719+0.j       ]])\n\n    """  # inserted
    UPLO = UPLO.upper()
    if UPLO not in ['L', 'U']:
        raise ValueError('UPLO argument must be \'L\' or \'U\'')
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:  # inserted
        gufunc = _umath_linalg.eigh_up
    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
        w, vt = gufunc(a, signature=signature)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return EighResult(w, wrap(vt))

def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    return (a,)

@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """\n    Singular Value Decomposition.\n\n    When `a` is a 2D array, and ``full_matrices=False``, then it is\n    factorized as ``u @ np.diag(s) @ vh = (u * s) @ vh``, where\n    `u` and the Hermitian transpose of `vh` are 2D arrays with\n    orthonormal columns and `s` is a 1D array of `a`\'s singular\n    values. When `a` is higher-dimensional, SVD is applied in\n    stacked mode as explained below.\n\n    Parameters\n    ----------\n    a : (..., M, N) array_like\n        A real or complex array with ``a.ndim >= 2``.\n    full_matrices : bool, optional\n        If True (default), `u` and `vh` have the shapes ``(..., M, M)`` and\n        ``(..., N, N)``, respectively.  Otherwise, the shapes are\n        ``(..., M, K)`` and ``(..., K, N)``, respectively, where\n        ``K = min(M, N)``.\n    compute_uv : bool, optional\n        Whether or not to compute `u` and `vh` in addition to `s`.  True\n        by default.\n    hermitian : bool, optional\n        If True, `a` is assumed to be Hermitian (symmetric if real-valued),\n        enabling a more efficient method for finding singular values.\n        Defaults to False.\n\n    Returns\n    -------\n    When `compute_uv` is True, the result is a namedtuple with the following\n    attribute names:\n\n    U : { (..., M, M), (..., M, K) } array\n        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same\n        size as those of the input `a`. The size of the last two dimensions\n        depends on the value of `full_matrices`. Only returned when\n        `compute_uv` is True.\n    S : (..., K) array\n        Vector(s) with the singular values, within each vector sorted in\n        descending order. The first ``a.ndim - 2`` dimensions have the same\n        size as those of the input `a`.\n    Vh : { (..., N, N), (..., K, N) } array\n        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same\n        size as those of the input `a`. The size of the last two dimensions\n        depends on the value of `full_matrices`. Only returned when\n        `compute_uv` is True.\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    See Also\n    --------\n    scipy.linalg.svd : Similar function in SciPy.\n    scipy.linalg.svdvals : Compute singular values of a matrix.\n\n    Notes\n    -----\n    The decomposition is performed using LAPACK routine ``_gesdd``.\n\n    SVD is usually described for the factorization of a 2D matrix :math:`A`.\n    The higher-dimensional case will be discussed below. In the 2D case, SVD is\n    written as :math:`A = U S V^H`, where :math:`A = a`, :math:`U= u`,\n    :math:`S= \\mathtt{np.diag}(s)` and :math:`V^H = vh`. The 1D array `s`\n    contains the singular values of `a` and `u` and `vh` are unitary. The rows\n    of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are\n    the eigenvectors of :math:`A A^H`. In both cases the corresponding\n    (possibly non-zero) eigenvalues are given by ``s**2``.\n\n    If `a` has more than two dimensions, then broadcasting rules apply, as\n    explained in :ref:`routines.linalg-broadcasting`. This means that SVD is\n    working in \"stacked\" mode: it iterates over all indices of the first\n    ``a.ndim - 2`` dimensions and for each combination SVD is applied to the\n    last two indices. The matrix `a` can be reconstructed from the\n    decomposition with either ``(u * s[..., None, :]) @ vh`` or\n    ``u @ (s[..., None] * vh)``. (The ``@`` operator can be replaced by the\n    function ``np.matmul`` for python versions below 3.5.)\n\n    If `a` is a ``matrix`` object (as opposed to an ``ndarray``), then so are\n    all the return values.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> a = rng.normal(size=(9, 6)) + 1j*rng.normal(size=(9, 6))\n    >>> b = rng.normal(size=(2, 7, 8, 3)) + 1j*rng.normal(size=(2, 7, 8, 3))\n\n\n    Reconstruction based on full SVD, 2D case:\n\n    >>> U, S, Vh = np.linalg.svd(a, full_matrices=True)\n    >>> U.shape, S.shape, Vh.shape\n    ((9, 9), (6,), (6, 6))\n    >>> np.allclose(a, np.dot(U[:, :6] * S, Vh))\n    True\n    >>> smat = np.zeros((9, 6), dtype=complex)\n    >>> smat[:6, :6] = np.diag(S)\n    >>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))\n    True\n\n    Reconstruction based on reduced SVD, 2D case:\n\n    >>> U, S, Vh = np.linalg.svd(a, full_matrices=False)\n    >>> U.shape, S.shape, Vh.shape\n    ((9, 6), (6,), (6, 6))\n    >>> np.allclose(a, np.dot(U * S, Vh))\n    True\n    >>> smat = np.diag(S)\n    >>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))\n    True\n\n    Reconstruction based on full SVD, 4D case:\n\n    >>> U, S, Vh = np.linalg.svd(b, full_matrices=True)\n    >>> U.shape, S.shape, Vh.shape\n    ((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))\n    >>> np.allclose(b, np.matmul(U[..., :3] * S[..., None, :], Vh))\n    True\n    >>> np.allclose(b, np.matmul(U[..., :3], S[..., None] * Vh))\n    True\n\n    Reconstruction based on reduced SVD, 4D case:\n\n    >>> U, S, Vh = np.linalg.svd(b, full_matrices=False)\n    >>> U.shape, S.shape, Vh.shape\n    ((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))\n    >>> np.allclose(b, np.matmul(U * S[..., None, :], Vh))\n    True\n    >>> np.allclose(b, np.matmul(U, S[..., None] * Vh))\n    True\n\n    """  # inserted
    import numpy as _nx
    a, wrap = _makearray(a)
    if hermitian:
        if compute_uv:
            s, u = eigh(a)
            sgn = sign(s)
            s = abs(s)
            sidx = argsort(s)[..., ::(-1)]
            sgn = _nx.take_along_axis(sgn, sidx, axis=(-1))
            s = _nx.take_along_axis(s, sidx, axis=(-1))
            u = _nx.take_along_axis(u, sidx[..., None, :], axis=(-1))
            vt = transpose(u * sgn[..., None, :]).conjugate()
            return SVDResult(wrap(u), s, wrap(vt))
        s = eigvalsh(a)
        s = abs(s)
        return sort(s)[..., ::(-1)]
    _assert_stacked_2d(a)
    t, result_t = _commonType(a)
    m, n = a.shape[(-2):]
    if compute_uv:
        if full_matrices:
            gufunc = _umath_linalg.svd_f
        else:  # inserted
            gufunc = _umath_linalg.svd_s
        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'
        with errstate(call=_raise_linalgerror_svd_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
            u, s, vh = gufunc(a, signature=signature)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return SVDResult(wrap(u), s, wrap(vh))
    signature = 'D->d' if isComplexType(t) else 'd->d'
    with errstate(call=_raise_linalgerror_svd_nonconvergence, invalid='call', over='ignore', divide='ignore', under='ignore'):
        s = _umath_linalg.svd(a, signature=signature)
    s = s.astype(_realType(result_t), copy=False)
    return s

def _svdvals_dispatcher(x):
    return (x,)

@array_function_dispatch(_svdvals_dispatcher)
def svdvals(x, /):
    """\n    Returns the singular values of a matrix (or a stack of matrices) ``x``.\n    When x is a stack of matrices, the function will compute the singular\n    values for each matrix in the stack.\n\n    This function is Array API compatible.\n\n    Calling ``np.svdvals(x)`` to get singular values is the same as\n    ``np.svd(x, compute_uv=False, hermitian=False)``.\n\n    Parameters\n    ----------\n    x : (..., M, N) array_like\n        Input array having shape (..., M, N) and whose last two\n        dimensions form matrices on which to perform singular value\n        decomposition. Should have a floating-point data type.\n\n    Returns\n    -------\n    out : ndarray\n        An array with shape (..., K) that contains the vector(s)\n        of singular values of length K, where K = min(M, N).\n\n    See Also\n    --------\n    scipy.linalg.svdvals : Compute singular values of a matrix.\n\n    Examples\n    --------\n\n    >>> np.linalg.svdvals([[1, 2, 3, 4, 5],\n    ...                    [1, 4, 9, 16, 25],\n    ...                    [1, 8, 27, 64, 125]])\n    array([146.68862757,   5.57510612,   0.60393245])\n\n    Determine the rank of a matrix using singular values:\n\n    >>> s = np.linalg.svdvals([[1, 2, 3],\n    ...                        [2, 4, 6],\n    ...                        [-1, 1, -1]]); s\n    array([8.38434191e+00, 1.64402274e+00, 2.31534378e-16])\n    >>> np.count_nonzero(s > 1e-10)  # Matrix of rank 2\n    2\n\n    """  # inserted
    return svd(x, compute_uv=False, hermitian=False)

def _cond_dispatcher(x, p=None):
    return (x,)

@array_function_dispatch(_cond_dispatcher)
def cond(x, p=None):
    """\n    Compute the condition number of a matrix.\n\n    This function is capable of returning the condition number using\n    one of seven different norms, depending on the value of `p` (see\n    Parameters below).\n\n    Parameters\n    ----------\n    x : (..., M, N) array_like\n        The matrix whose condition number is sought.\n    p : {None, 1, -1, 2, -2, inf, -inf, \'fro\'}, optional\n        Order of the norm used in the condition number computation:\n\n        =====  ============================\n        p      norm for matrices\n        =====  ============================\n        None   2-norm, computed directly using the ``SVD``\n        \'fro\'  Frobenius norm\n        inf    max(sum(abs(x), axis=1))\n        -inf   min(sum(abs(x), axis=1))\n        1      max(sum(abs(x), axis=0))\n        -1     min(sum(abs(x), axis=0))\n        2      2-norm (largest sing. value)\n        -2     smallest singular value\n        =====  ============================\n\n        inf means the `numpy.inf` object, and the Frobenius norm is\n        the root-of-sum-of-squares norm.\n\n    Returns\n    -------\n    c : {float, inf}\n        The condition number of the matrix. May be infinite.\n\n    See Also\n    --------\n    numpy.linalg.norm\n\n    Notes\n    -----\n    The condition number of `x` is defined as the norm of `x` times the\n    norm of the inverse of `x` [1]_; the norm can be the usual L2-norm\n    (root-of-sum-of-squares) or one of a number of other matrix norms.\n\n    References\n    ----------\n    .. [1] G. Strang, *Linear Algebra and Its Applications*, Orlando, FL,\n           Academic Press, Inc., 1980, pg. 285.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])\n    >>> a\n    array([[ 1,  0, -1],\n           [ 0,  1,  0],\n           [ 1,  0,  1]])\n    >>> LA.cond(a)\n    1.4142135623730951\n    >>> LA.cond(a, \'fro\')\n    3.1622776601683795\n    >>> LA.cond(a, np.inf)\n    2.0\n    >>> LA.cond(a, -np.inf)\n    1.0\n    >>> LA.cond(a, 1)\n    2.0\n    >>> LA.cond(a, -1)\n    1.0\n    >>> LA.cond(a, 2)\n    1.4142135623730951\n    >>> LA.cond(a, -2)\n    0.70710678118654746 # may vary\n    >>> (min(LA.svd(a, compute_uv=False)) *\n    ... min(LA.svd(LA.inv(a), compute_uv=False)))\n    0.70710678118654746 # may vary\n\n    """  # inserted
    x = asarray(x)
    if _is_empty_2d(x):
        raise LinAlgError('cond is not defined on empty arrays')
    if p is None or p == 2 or p == (-2):
        s = svd(x, compute_uv=False)
        with errstate(all='ignore'):
            if p == (-2):
                r = s[..., (-1)] / s[..., 0]
            else:  # inserted
                r = s[..., 0] / s[..., (-1)]
    else:  # inserted
        _assert_stacked_2d(x)
        _assert_stacked_square(x)
        t, result_t = _commonType(x)
        signature = 'D->D' if isComplexType(t) else 'd->d'
        with errstate(all='ignore'):
            invx = _umath_linalg.inv(x, signature=signature)
            r = norm(x, p, axis=((-2), (-1))) * norm(invx, p, axis=((-2), (-1)))
        r = r.astype(result_t, copy=False)
    r = asarray(r)
    nan_mask = isnan(r)
    if nan_mask.any():
        nan_mask &= ~isnan(x).any(axis=((-2), (-1)))
        if r.ndim > 0:
            r[nan_mask] = inf
        else:  # inserted
            if nan_mask:
                r[()] = inf
    if r.ndim == 0:
        r = r[()]
    return r

def _matrix_rank_dispatcher(A, tol=None, hermitian=None, *, rtol=None):
    return (A,)

@array_function_dispatch(_matrix_rank_dispatcher)
def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """\n    Return matrix rank of array using SVD method\n\n    Rank of the array is the number of singular values of the array that are\n    greater than `tol`.\n\n    Parameters\n    ----------\n    A : {(M,), (..., M, N)} array_like\n        Input vector or stack of matrices.\n    tol : (...) array_like, float, optional\n        Threshold below which SVD values are considered zero. If `tol` is\n        None, and ``S`` is an array with singular values for `M`, and\n        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is\n        set to ``S.max() * max(M, N) * eps``.\n    hermitian : bool, optional\n        If True, `A` is assumed to be Hermitian (symmetric if real-valued),\n        enabling a more efficient method for finding singular values.\n        Defaults to False.\n    rtol : (...) array_like, float, optional\n        Parameter for the relative tolerance component. Only ``tol`` or\n        ``rtol`` can be set at a time. Defaults to ``max(M, N) * eps``.\n\n        .. versionadded:: 2.0.0\n\n    Returns\n    -------\n    rank : (...) array_like\n        Rank of A.\n\n    Notes\n    -----\n    The default threshold to detect rank deficiency is a test on the magnitude\n    of the singular values of `A`.  By default, we identify singular values\n    less than ``S.max() * max(M, N) * eps`` as indicating rank deficiency\n    (with the symbols defined above). This is the algorithm MATLAB uses [1].\n    It also appears in *Numerical recipes* in the discussion of SVD solutions\n    for linear least squares [2].\n\n    This default threshold is designed to detect rank deficiency accounting\n    for the numerical errors of the SVD computation. Imagine that there\n    is a column in `A` that is an exact (in floating point) linear combination\n    of other columns in `A`. Computing the SVD on `A` will not produce\n    a singular value exactly equal to 0 in general: any difference of\n    the smallest SVD value from 0 will be caused by numerical imprecision\n    in the calculation of the SVD. Our threshold for small SVD values takes\n    this numerical imprecision into account, and the default threshold will\n    detect such numerical rank deficiency. The threshold may declare a matrix\n    `A` rank deficient even if the linear combination of some columns of `A`\n    is not exactly equal to another column of `A` but only numerically very\n    close to another column of `A`.\n\n    We chose our default threshold because it is in wide use. Other thresholds\n    are possible.  For example, elsewhere in the 2007 edition of *Numerical\n    recipes* there is an alternative threshold of ``S.max() *\n    np.finfo(A.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe\n    this threshold as being based on \"expected roundoff error\" (p 71).\n\n    The thresholds above deal with floating point roundoff error in the\n    calculation of the SVD.  However, you may have more information about\n    the sources of error in `A` that would make you consider other tolerance\n    values to detect *effective* rank deficiency. The most useful measure\n    of the tolerance depends on the operations you intend to use on your\n    matrix. For example, if your data come from uncertain measurements with\n    uncertainties greater than floating point epsilon, choosing a tolerance\n    near that uncertainty may be preferable. The tolerance may be absolute\n    if the uncertainties are absolute rather than relative.\n\n    References\n    ----------\n    .. [1] MATLAB reference documentation, \"Rank\"\n           https://www.mathworks.com/help/techdoc/ref/rank.html\n    .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,\n           \"Numerical Recipes (3rd edition)\", Cambridge University Press, 2007,\n           page 795.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from numpy.linalg import matrix_rank\n    >>> matrix_rank(np.eye(4)) # Full rank matrix\n    4\n    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix\n    >>> matrix_rank(I)\n    3\n    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0\n    1\n    >>> matrix_rank(np.zeros((4,)))\n    0\n    """  # inserted
    if rtol is not None and tol is not None:
        raise ValueError('`tol` and `rtol` can\'t be both set.')
    A = asarray(A)
    if A.ndim < 2:
        return int(not all(A == 0))
    S = svd(A, compute_uv=False, hermitian=hermitian)
    if tol is None:
        if rtol is None:
            rtol = max(A.shape[(-2):]) * finfo(S.dtype).eps
        else:  # inserted
            rtol = asarray(rtol)[..., newaxis]
        tol = S.max(axis=(-1), keepdims=True) * rtol
    else:  # inserted
        tol = asarray(tol)[..., newaxis]
    return count_nonzero(S > tol, axis=(-1))

def _pinv_dispatcher(a, rcond=None, hermitian=None, *, rtol=None):
    return (a,)

@array_function_dispatch(_pinv_dispatcher)
def pinv(a, rcond=None, hermitian=False, *, rtol=_NoValue):
    """\n    Compute the (Moore-Penrose) pseudo-inverse of a matrix.\n\n    Calculate the generalized inverse of a matrix using its\n    singular-value decomposition (SVD) and including all\n    *large* singular values.\n\n    Parameters\n    ----------\n    a : (..., M, N) array_like\n        Matrix or stack of matrices to be pseudo-inverted.\n    rcond : (...) array_like of float, optional\n        Cutoff for small singular values.\n        Singular values less than or equal to\n        ``rcond * largest_singular_value`` are set to zero.\n        Broadcasts against the stack of matrices. Default: ``1e-15``.\n    hermitian : bool, optional\n        If True, `a` is assumed to be Hermitian (symmetric if real-valued),\n        enabling a more efficient method for finding singular values.\n        Defaults to False.\n    rtol : (...) array_like of float, optional\n        Same as `rcond`, but it\'s an Array API compatible parameter name.\n        Only `rcond` or `rtol` can be set at a time. If none of them are\n        provided then NumPy\'s ``1e-15`` default is used. If ``rtol=None``\n        is passed then the API standard default is used.\n\n        .. versionadded:: 2.0.0\n\n    Returns\n    -------\n    B : (..., N, M) ndarray\n        The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so\n        is `B`.\n\n    Raises\n    ------\n    LinAlgError\n        If the SVD computation does not converge.\n\n    See Also\n    --------\n    scipy.linalg.pinv : Similar function in SciPy.\n    scipy.linalg.pinvh : Compute the (Moore-Penrose) pseudo-inverse of a\n                         Hermitian matrix.\n\n    Notes\n    -----\n    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is\n    defined as: \"the matrix that \'solves\' [the least-squares problem]\n    :math:`Ax = b`,\" i.e., if :math:`\\bar{x}` is said solution, then\n    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.\n\n    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular\n    value decomposition of A, then\n    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are\n    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting\n    of A\'s so-called singular values, (followed, typically, by\n    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix\n    consisting of the reciprocals of A\'s singular values\n    (again, followed by zeros). [1]_\n\n    References\n    ----------\n    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,\n           FL, Academic Press, Inc., 1980, pp. 139-142.\n\n    Examples\n    --------\n    The following example checks that ``a * a+ * a == a`` and\n    ``a+ * a * a+ == a+``:\n\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> a = rng.normal(size=(9, 6))\n    >>> B = np.linalg.pinv(a)\n    >>> np.allclose(a, np.dot(a, np.dot(B, a)))\n    True\n    >>> np.allclose(B, np.dot(B, np.dot(a, B)))\n    True\n\n    """  # inserted
    a, wrap = _makearray(a)
    if rcond is None:
        if rtol is _NoValue:
            rcond = 1e-15
        else:  # inserted
            if rtol is None:
                rcond = max(a.shape[(-2):]) * finfo(a.dtype).eps
            else:  # inserted
                rcond = rtol
    else:  # inserted
        if rtol is not _NoValue:
            raise ValueError('`rtol` and `rcond` can\'t be both set.')
        if False:
            pass  # postinserted
    rcond = asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[(-2):]
        res = empty(a.shape[:(-2)] + (n, m), dtype=a.dtype)
        return wrap(res)
    a = a.conjugate()
    u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)
    cutoff = rcond[..., newaxis] * amax(s, axis=(-1), keepdims=True)
    large = s > cutoff
    s = divide(1, s, where=large, out=s)
    s[~large] = 0
    res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
    return wrap(res)

@array_function_dispatch(_unary_dispatcher)
def slogdet(a):
    """\n    Compute the sign and (natural) logarithm of the determinant of an array.\n\n    If an array has a very small or very large determinant, then a call to\n    `det` may overflow or underflow. This routine is more robust against such\n    issues, because it computes the logarithm of the determinant rather than\n    the determinant itself.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Input array, has to be a square 2-D array.\n\n    Returns\n    -------\n    A namedtuple with the following attributes:\n\n    sign : (...) array_like\n        A number representing the sign of the determinant. For a real matrix,\n        this is 1, 0, or -1. For a complex matrix, this is a complex number\n        with absolute value 1 (i.e., it is on the unit circle), or else 0.\n    logabsdet : (...) array_like\n        The natural log of the absolute value of the determinant.\n\n    If the determinant is zero, then `sign` will be 0 and `logabsdet`\n    will be -inf. In all cases, the determinant is equal to\n    ``sign * np.exp(logabsdet)``.\n\n    See Also\n    --------\n    det\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The determinant is computed via LU factorization using the LAPACK\n    routine ``z/dgetrf``.\n\n    Examples\n    --------\n    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:\n\n    >>> import numpy as np\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> (sign, logabsdet) = np.linalg.slogdet(a)\n    >>> (sign, logabsdet)\n    (-1, 0.69314718055994529) # may vary\n    >>> sign * np.exp(logabsdet)\n    -2.0\n\n    Computing log-determinants for a stack of matrices:\n\n    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])\n    >>> a.shape\n    (3, 2, 2)\n    >>> sign, logabsdet = np.linalg.slogdet(a)\n    >>> (sign, logabsdet)\n    (array([-1., -1., -1.]), array([ 0.69314718,  1.09861229,  2.07944154]))\n    >>> sign * np.exp(logabsdet)\n    array([-2., -3., -8.])\n\n    This routine succeeds where ordinary `det` does not:\n\n    >>> np.linalg.det(np.eye(500) * 0.1)\n    0.0\n    >>> np.linalg.slogdet(np.eye(500) * 0.1)\n    (1, -1151.2925464970228)\n\n    """  # inserted
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    real_t = _realType(result_t)
    signature = 'D->Dd' if isComplexType(t) else 'd->dd'
    sign, logdet = _umath_linalg.slogdet(a, signature=signature)
    sign = sign.astype(result_t, copy=False)
    logdet = logdet.astype(real_t, copy=False)
    return SlogdetResult(sign, logdet)

@array_function_dispatch(_unary_dispatcher)
def det(a):
    """\n    Compute the determinant of an array.\n\n    Parameters\n    ----------\n    a : (..., M, M) array_like\n        Input array to compute determinants for.\n\n    Returns\n    -------\n    det : (...) array_like\n        Determinant of `a`.\n\n    See Also\n    --------\n    slogdet : Another way to represent the determinant, more suitable\n      for large matrices where underflow/overflow may occur.\n    scipy.linalg.det : Similar function in SciPy.\n\n    Notes\n    -----\n    Broadcasting rules apply, see the `numpy.linalg` documentation for\n    details.\n\n    The determinant is computed via LU factorization using the LAPACK\n    routine ``z/dgetrf``.\n\n    Examples\n    --------\n    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:\n\n    >>> import numpy as np\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> np.linalg.det(a)\n    -2.0 # may vary\n\n    Computing determinants for a stack of matrices:\n\n    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])\n    >>> a.shape\n    (3, 2, 2)\n    >>> np.linalg.det(a)\n    array([-2., -3., -8.])\n\n    """  # inserted
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    r = _umath_linalg.det(a, signature=signature)
    r = r.astype(result_t, copy=False)
    return r

def _lstsq_dispatcher(a, b, rcond=None):
    return (a, b)

@array_function_dispatch(_lstsq_dispatcher)
def lstsq(a, b, rcond=None):
    """\n    Return the least-squares solution to a linear matrix equation.\n\n    Computes the vector `x` that approximately solves the equation\n    ``a @ x = b``. The equation may be under-, well-, or over-determined\n    (i.e., the number of linearly independent rows of `a` can be less than,\n    equal to, or greater than its number of linearly independent columns).\n    If `a` is square and of full rank, then `x` (but for round-off error)\n    is the \"exact\" solution of the equation. Else, `x` minimizes the\n    Euclidean 2-norm :math:`||b - ax||`. If there are multiple minimizing\n    solutions, the one with the smallest 2-norm :math:`||x||` is returned.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        \"Coefficient\" matrix.\n    b : {(M,), (M, K)} array_like\n        Ordinate or \"dependent variable\" values. If `b` is two-dimensional,\n        the least-squares solution is calculated for each of the `K` columns\n        of `b`.\n    rcond : float, optional\n        Cut-off ratio for small singular values of `a`.\n        For the purposes of rank determination, singular values are treated\n        as zero if they are smaller than `rcond` times the largest singular\n        value of `a`.\n        The default uses the machine precision times ``max(M, N)``.  Passing\n        ``-1`` will use machine precision.\n\n        .. versionchanged:: 2.0\n            Previously, the default was ``-1``, but a warning was given that\n            this would change.\n\n    Returns\n    -------\n    x : {(N,), (N, K)} ndarray\n        Least-squares solution. If `b` is two-dimensional,\n        the solutions are in the `K` columns of `x`.\n    residuals : {(1,), (K,), (0,)} ndarray\n        Sums of squared residuals: Squared Euclidean 2-norm for each column in\n        ``b - a @ x``.\n        If the rank of `a` is < N or M <= N, this is an empty array.\n        If `b` is 1-dimensional, this is a (1,) shape array.\n        Otherwise the shape is (K,).\n    rank : int\n        Rank of matrix `a`.\n    s : (min(M, N),) ndarray\n        Singular values of `a`.\n\n    Raises\n    ------\n    LinAlgError\n        If computation does not converge.\n\n    See Also\n    --------\n    scipy.linalg.lstsq : Similar function in SciPy.\n\n    Notes\n    -----\n    If `b` is a matrix, then all array results are returned as matrices.\n\n    Examples\n    --------\n    Fit a line, ``y = mx + c``, through some noisy data-points:\n\n    >>> import numpy as np\n    >>> x = np.array([0, 1, 2, 3])\n    >>> y = np.array([-1, 0.2, 0.9, 2.1])\n\n    By examining the coefficients, we see that the line should have a\n    gradient of roughly 1 and cut the y-axis at, more or less, -1.\n\n    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``\n    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:\n\n    >>> A = np.vstack([x, np.ones(len(x))]).T\n    >>> A\n    array([[ 0.,  1.],\n           [ 1.,  1.],\n           [ 2.,  1.],\n           [ 3.,  1.]])\n\n    >>> m, c = np.linalg.lstsq(A, y)[0]\n    >>> m, c\n    (1.0 -0.95) # may vary\n\n    Plot the data along with the fitted line:\n\n    >>> import matplotlib.pyplot as plt\n    >>> _ = plt.plot(x, y, \'o\', label=\'Original data\', markersize=10)\n    >>> _ = plt.plot(x, m*x + c, \'r\', label=\'Fitted line\')\n    >>> _ = plt.legend()\n    >>> plt.show()\n\n    """  # inserted
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    _assert_2d(a, b)
    m, n = a.shape[(-2):]
    m2, n_rhs = b.shape[(-2):]
    if m!= m2:
        raise LinAlgError('Incompatible dimensions')
    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)
    if rcond is None:
        rcond = finfo(t).eps * max(n, m)
    signature = 'DDd->Ddid' if isComplexType(t) else 'ddd->ddid'
    if n_rhs == 0:
        b = zeros(b.shape[:(-2)] + (m, n_rhs + 1), dtype=b.dtype)
    with errstate(call=_raise_linalgerror_lstsq, invalid='call', over='ignore', divide='ignore', under='ignore'):
        x, resids, rank, s = _umath_linalg.lstsq(a, b, rcond, signature=signature)
    if m == 0:
        x[...] = 0
    if n_rhs == 0:
        x = x[..., :n_rhs]
        resids = resids[..., :n_rhs]
    if is_1d:
        x = x.squeeze(axis=(-1))
    if rank!= n or m <= n:
        resids = array([], result_real_t)
    s = s.astype(result_real_t, copy=False)
    resids = resids.astype(result_real_t, copy=False)
    x = x.astype(result_t, copy=True)
    return (wrap(x), wrap(resids), rank, s)

def _multi_svd_norm(x, row_axis, col_axis, op):
    """Compute a function of the singular values of the 2-D matrices in `x`.\n\n    This is a private utility function used by `numpy.linalg.norm()`.\n\n    Parameters\n    ----------\n    x : ndarray\n    row_axis, col_axis : int\n        The axes of `x` that hold the 2-D matrices.\n    op : callable\n        This should be either numpy.amin or `numpy.amax` or `numpy.sum`.\n\n    Returns\n    -------\n    result : float or ndarray\n        If `x` is 2-D, the return values is a float.\n        Otherwise, it is an array with ``x.ndim - 2`` dimensions.\n        The return values are either the minimum or maximum or sum of the\n        singular values of the matrices, depending on whether `op`\n        is `numpy.amin` or `numpy.amax` or `numpy.sum`.\n\n    """  # inserted
    y = moveaxis(x, (row_axis, col_axis), ((-2), (-1)))
    result = op(svd(y, compute_uv=False), axis=(-1))
    return result

def _norm_dispatcher(x, ord=None, axis=None, keepdims=None):
    return (x,)

@array_function_dispatch(_norm_dispatcher)
def norm(x, ord=None, axis=None, keepdims=False):
    """\n    Matrix or vector norm.\n\n    This function is able to return one of eight different matrix norms,\n    or one of an infinite number of vector norms (described below), depending\n    on the value of the ``ord`` parameter.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`\n        is None. If both `axis` and `ord` are None, the 2-norm of\n        ``x.ravel`` will be returned.\n    ord : {int, float, inf, -inf, \'fro\', \'nuc\'}, optional\n        Order of the norm (see table under ``Notes`` for what values are\n        supported for matrices and vectors respectively). inf means numpy\'s\n        `inf` object. The default is None.\n    axis : {None, int, 2-tuple of ints}, optional.\n        If `axis` is an integer, it specifies the axis of `x` along which to\n        compute the vector norms.  If `axis` is a 2-tuple, it specifies the\n        axes that hold 2-D matrices, and the matrix norms of these matrices\n        are computed.  If `axis` is None then either a vector norm (when `x`\n        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default\n        is None.\n\n    keepdims : bool, optional\n        If this is set to True, the axes which are normed over are left in the\n        result as dimensions with size one.  With this option the result will\n        broadcast correctly against the original `x`.\n\n    Returns\n    -------\n    n : float or ndarray\n        Norm of the matrix or vector(s).\n\n    See Also\n    --------\n    scipy.linalg.norm : Similar function in SciPy.\n\n    Notes\n    -----\n    For values of ``ord < 1``, the result is, strictly speaking, not a\n    mathematical \'norm\', but it may still be useful for various numerical\n    purposes.\n\n    The following norms can be calculated:\n\n    =====  ============================  ==========================\n    ord    norm for matrices             norm for vectors\n    =====  ============================  ==========================\n    None   Frobenius norm                2-norm\n    \'fro\'  Frobenius norm                --\n    \'nuc\'  nuclear norm                  --\n    inf    max(sum(abs(x), axis=1))      max(abs(x))\n    -inf   min(sum(abs(x), axis=1))      min(abs(x))\n    0      --                            sum(x != 0)\n    1      max(sum(abs(x), axis=0))      as below\n    -1     min(sum(abs(x), axis=0))      as below\n    2      2-norm (largest sing. value)  as below\n    -2     smallest singular value       as below\n    other  --                            sum(abs(x)**ord)**(1./ord)\n    =====  ============================  ==========================\n\n    The Frobenius norm is given by [1]_:\n\n    :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`\n\n    The nuclear norm is the sum of the singular values.\n\n    Both the Frobenius and nuclear norm orders are only defined for\n    matrices and raise a ValueError when ``x.ndim != 2``.\n\n    References\n    ----------\n    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,\n           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from numpy import linalg as LA\n    >>> a = np.arange(9) - 4\n    >>> a\n    array([-4, -3, -2, ...,  2,  3,  4])\n    >>> b = a.reshape((3, 3))\n    >>> b\n    array([[-4, -3, -2],\n           [-1,  0,  1],\n           [ 2,  3,  4]])\n\n    >>> LA.norm(a)\n    7.745966692414834\n    >>> LA.norm(b)\n    7.745966692414834\n    >>> LA.norm(b, \'fro\')\n    7.745966692414834\n    >>> LA.norm(a, np.inf)\n    4.0\n    >>> LA.norm(b, np.inf)\n    9.0\n    >>> LA.norm(a, -np.inf)\n    0.0\n    >>> LA.norm(b, -np.inf)\n    2.0\n\n    >>> LA.norm(a, 1)\n    20.0\n    >>> LA.norm(b, 1)\n    7.0\n    >>> LA.norm(a, -1)\n    -4.6566128774142013e-010\n    >>> LA.norm(b, -1)\n    6.0\n    >>> LA.norm(a, 2)\n    7.745966692414834\n    >>> LA.norm(b, 2)\n    7.3484692283495345\n\n    >>> LA.norm(a, -2)\n    0.0\n    >>> LA.norm(b, -2)\n    1.8570331885190563e-016 # may vary\n    >>> LA.norm(a, 3)\n    5.8480354764257312 # may vary\n    >>> LA.norm(a, -3)\n    0.0\n\n    Using the `axis` argument to compute vector norms:\n\n    >>> c = np.array([[ 1, 2, 3],\n    ...               [-1, 1, 4]])\n    >>> LA.norm(c, axis=0)\n    array([ 1.41421356,  2.23606798,  5.        ])\n    >>> LA.norm(c, axis=1)\n    array([ 3.74165739,  4.24264069])\n    >>> LA.norm(c, ord=1, axis=1)\n    array([ 6.,  6.])\n\n    Using the `axis` argument to compute matrix norms:\n\n    >>> m = np.arange(8).reshape(2,2,2)\n    >>> LA.norm(m, axis=(1,2))\n    array([  3.74165739,  11.22497216])\n    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])\n    (3.7416573867739413, 11.224972160321824)\n\n    """  # inserted
    x = asarray(x)
    if not issubclass(x.dtype.type, (inexact, object_)):
        x = x.astype(float)
    if axis is None:
        ndim = x.ndim
        if ord is None or (ord in ('f', 'fro') and ndim == 2) or (ord == 2 and ndim == 1):
            x = x.ravel(order='K')
            if isComplexType(x.dtype.type):
                x_real = x.real
                x_imag = x.imag
                sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)
            else:  # inserted
                sqnorm = x.dot(x)
            ret = sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    else:  # inserted
        if not isinstance(axis, tuple):
            try:
                axis = int(axis)
            except Exception as e:
                pass  # postinserted
            else:  # inserted
                axis = (axis,)
    if len(axis) == 1:
        if ord == inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        if ord == -inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        if ord == 0:
            return (x!= 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
        if ord == 1:
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        if ord is None or ord == 2:
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        if isinstance(ord, str):
            raise ValueError(f'Invalid norm order \'{ord}\' for vectors')
        absx = abs(x)
        absx **= ord
        ret = add.reduce(absx, axis=axis, keepdims=keepdims)
        ret **= reciprocal(ord, dtype=ret.dtype)
        return ret
    if len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amax)
        else:  # inserted
            if ord == (-2):
                ret = _multi_svd_norm(x, row_axis, col_axis, amin)
            else:  # inserted
                if ord == 1:
                    if col_axis > row_axis:
                        col_axis -= 1
                    ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
                else:  # inserted
                    if ord == inf:
                        if row_axis > col_axis:
                            row_axis -= 1
                        ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
                    else:  # inserted
                        if ord == (-1):
                            if col_axis > row_axis:
                                col_axis -= 1
                            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
                        else:  # inserted
                            if ord == -inf:
                                if row_axis > col_axis:
                                    row_axis -= 1
                                ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
                            else:  # inserted
                                if ord in (None, 'fro', 'f'):
                                    ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
                                else:  # inserted
                                    if ord == 'nuc':
                                        ret = _multi_svd_norm(x, row_axis, col_axis, sum)
                                    else:  # inserted
                                        raise ValueError('Invalid norm order for matrices.')
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    raise ValueError('Improper number of dimensions to norm.')
                raise TypeError('\'axis\' must be None, an integer or a tuple of integers') from e

def _multidot_dispatcher(arrays, *, out=None):
    yield from arrays
    yield out

@array_function_dispatch(_multidot_dispatcher)
def multi_dot(arrays, *, out=None):
    """\n    Compute the dot product of two or more arrays in a single function call,\n    while automatically selecting the fastest evaluation order.\n\n    `multi_dot` chains `numpy.dot` and uses optimal parenthesization\n    of the matrices [1]_ [2]_. Depending on the shapes of the matrices,\n    this can speed up the multiplication a lot.\n\n    If the first argument is 1-D it is treated as a row vector.\n    If the last argument is 1-D it is treated as a column vector.\n    The other arguments must be 2-D.\n\n    Think of `multi_dot` as::\n\n        def multi_dot(arrays): return functools.reduce(np.dot, arrays)\n\n\n    Parameters\n    ----------\n    arrays : sequence of array_like\n        If the first argument is 1-D it is treated as row vector.\n        If the last argument is 1-D it is treated as column vector.\n        The other arguments must be 2-D.\n    out : ndarray, optional\n        Output argument. This must have the exact kind that would be returned\n        if it was not used. In particular, it must have the right type, must be\n        C-contiguous, and its dtype must be the dtype that would be returned\n        for `dot(a, b)`. This is a performance feature. Therefore, if these\n        conditions are not met, an exception is raised, instead of attempting\n        to be flexible.\n\n    Returns\n    -------\n    output : ndarray\n        Returns the dot product of the supplied arrays.\n\n    See Also\n    --------\n    numpy.dot : dot multiplication with two arguments.\n\n    References\n    ----------\n\n    .. [1] Cormen, \"Introduction to Algorithms\", Chapter 15.2, p. 370-378\n    .. [2] https://en.wikipedia.org/wiki/Matrix_chain_multiplication\n\n    Examples\n    --------\n    `multi_dot` allows you to write::\n\n    >>> import numpy as np\n    >>> from numpy.linalg import multi_dot\n    >>> # Prepare some data\n    >>> A = np.random.random((10000, 100))\n    >>> B = np.random.random((100, 1000))\n    >>> C = np.random.random((1000, 5))\n    >>> D = np.random.random((5, 333))\n    >>> # the actual dot multiplication\n    >>> _ = multi_dot([A, B, C, D])\n\n    instead of::\n\n    >>> _ = np.dot(np.dot(np.dot(A, B), C), D)\n    >>> # or\n    >>> _ = A.dot(B).dot(C).dot(D)\n\n    Notes\n    -----\n    The cost for a matrix multiplication can be calculated with the\n    following function::\n\n        def cost(A, B):\n            return A.shape[0] * A.shape[1] * B.shape[1]\n\n    Assume we have three matrices\n    :math:`A_{10x100}, B_{100x5}, C_{5x50}`.\n\n    The costs for the two different parenthesizations are as follows::\n\n        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500\n        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000\n\n    """  # inserted
    n = len(arrays)
    if n < 2:
        raise ValueError('Expecting at least two arrays.')
    if n == 2:
        return dot(arrays[0], arrays[1], out=out)
    arrays = [asanyarray(a) for a in arrays]
    ndim_first, ndim_last = (arrays[0].ndim, arrays[(-1)].ndim)
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])
    if arrays[(-1)].ndim == 1:
        arrays[(-1)] = atleast_2d(arrays[(-1)]).T
    _assert_2d(*arrays)
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:  # inserted
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]
    if ndim_first == 1 or ndim_last == 1:
        return result.ravel()
    return result

def _multi_dot_three(A, B, C, out=None):
    """\n    Find the best order for three arrays and do the multiplication.\n\n    For three arguments `_multi_dot_three` is approximately 15 times faster\n    than `_multi_dot_matrix_chain_order`\n\n    """  # inserted
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)
    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    return dot(A, dot(B, C), out=out)

def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """\n    Return a np.array that encodes the optimal order of multiplications.\n\n    The optimal order array is then used by `_multi_dot()` to do the\n    multiplication.\n\n    Also return the cost matrix if `return_costs` is `True`\n\n    The implementation CLOSELY follows Cormen, \"Introduction to Algorithms\",\n    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.\n\n        cost[i, j] = min([\n            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)\n            for k in range(i, j)])\n\n    """  # inserted
    n = len(arrays)
    p = [a.shape[0] for a in arrays] + [arrays[(-1)].shape[1]]
    m = zeros((n, n), dtype=double)
    s = empty((n, n), dtype=intp)
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k
    return (s, m) if return_costs else s

def _multi_dot(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""  # inserted
    if i == j:
        assert out is None
        return arrays[i]
    else:  # inserted
        return dot(_multi_dot(arrays, order, i, order[i, j]), _multi_dot(arrays, order, order[i, j] + 1, j), out=out)

def _diagonal_dispatcher(x, /, *, offset=None):
    return (x,)

@array_function_dispatch(_diagonal_dispatcher)
def diagonal(x, /, *, offset=0):
    """\n    Returns specified diagonals of a matrix (or a stack of matrices) ``x``.\n\n    This function is Array API compatible, contrary to\n    :py:func:`numpy.diagonal`, the matrix is assumed\n    to be defined by the last two dimensions.\n\n    Parameters\n    ----------\n    x : (...,M,N) array_like\n        Input array having shape (..., M, N) and whose innermost two\n        dimensions form MxN matrices.\n    offset : int, optional\n        Offset specifying the off-diagonal relative to the main diagonal,\n        where::\n\n            * offset = 0: the main diagonal.\n            * offset > 0: off-diagonal above the main diagonal.\n            * offset < 0: off-diagonal below the main diagonal.\n\n    Returns\n    -------\n    out : (...,min(N,M)) ndarray\n        An array containing the diagonals and whose shape is determined by\n        removing the last two dimensions and appending a dimension equal to\n        the size of the resulting diagonals. The returned array must have\n        the same data type as ``x``.\n\n    See Also\n    --------\n    numpy.diagonal\n\n    Examples\n    --------\n    >>> a = np.arange(4).reshape(2, 2); a\n    array([[0, 1],\n           [2, 3]])\n    >>> np.linalg.diagonal(a)\n    array([0, 3])\n\n    A 3-D example:\n\n    >>> a = np.arange(8).reshape(2, 2, 2); a\n    array([[[0, 1],\n            [2, 3]],\n           [[4, 5],\n            [6, 7]]])\n    >>> np.linalg.diagonal(a)\n    array([[0, 3],\n           [4, 7]])\n\n    Diagonals adjacent to the main diagonal can be obtained by using the\n    `offset` argument:\n\n    >>> a = np.arange(9).reshape(3, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n    >>> np.linalg.diagonal(a, offset=1)  # First superdiagonal\n    array([1, 5])\n    >>> np.linalg.diagonal(a, offset=2)  # Second superdiagonal\n    array([2])\n    >>> np.linalg.diagonal(a, offset=-1)  # First subdiagonal\n    array([3, 7])\n    >>> np.linalg.diagonal(a, offset=-2)  # Second subdiagonal\n    array([6])\n\n    The anti-diagonal can be obtained by reversing the order of elements\n    using either `numpy.flipud` or `numpy.fliplr`.\n\n    >>> a = np.arange(9).reshape(3, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n    >>> np.linalg.diagonal(np.fliplr(a))  # Horizontal flip\n    array([2, 4, 6])\n    >>> np.linalg.diagonal(np.flipud(a))  # Vertical flip\n    array([6, 4, 2])\n\n    Note that the order in which the diagonal is retrieved varies depending\n    on the flip function.\n\n    """  # inserted
    return _core_diagonal(x, offset, axis1=(-2), axis2=(-1))

def _trace_dispatcher(x, /, *, offset=None, dtype=None):
    return (x,)

@array_function_dispatch(_trace_dispatcher)
def trace(x, /, *, offset=0, dtype=None):
    """\n    Returns the sum along the specified diagonals of a matrix\n    (or a stack of matrices) ``x``.\n\n    This function is Array API compatible, contrary to\n    :py:func:`numpy.trace`.\n\n    Parameters\n    ----------\n    x : (...,M,N) array_like\n        Input array having shape (..., M, N) and whose innermost two\n        dimensions form MxN matrices.\n    offset : int, optional\n        Offset specifying the off-diagonal relative to the main diagonal,\n        where::\n\n            * offset = 0: the main diagonal.\n            * offset > 0: off-diagonal above the main diagonal.\n            * offset < 0: off-diagonal below the main diagonal.\n\n    dtype : dtype, optional\n        Data type of the returned array.\n\n    Returns\n    -------\n    out : ndarray\n        An array containing the traces and whose shape is determined by\n        removing the last two dimensions and storing the traces in the last\n        array dimension. For example, if x has rank k and shape:\n        (I, J, K, ..., L, M, N), then an output array has rank k-2 and shape:\n        (I, J, K, ..., L) where::\n\n            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])\n\n        The returned array must have a data type as described by the dtype\n        parameter above.\n\n    See Also\n    --------\n    numpy.trace\n\n    Examples\n    --------\n    >>> np.linalg.trace(np.eye(3))\n    3.0\n    >>> a = np.arange(8).reshape((2, 2, 2))\n    >>> np.linalg.trace(a)\n    array([3, 11])\n\n    Trace is computed with the last two axes as the 2-d sub-arrays.\n    This behavior differs from :py:func:`numpy.trace` which uses the first two\n    axes by default.\n\n    >>> a = np.arange(24).reshape((3, 2, 2, 2))\n    >>> np.linalg.trace(a).shape\n    (3, 2)\n\n    Traces adjacent to the main diagonal can be obtained by using the\n    `offset` argument:\n\n    >>> a = np.arange(9).reshape((3, 3)); a\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n    >>> np.linalg.trace(a, offset=1)  # First superdiagonal\n    6\n    >>> np.linalg.trace(a, offset=2)  # Second superdiagonal\n    2\n    >>> np.linalg.trace(a, offset=-1)  # First subdiagonal\n    10\n    >>> np.linalg.trace(a, offset=-2)  # Second subdiagonal\n    6\n\n    """  # inserted
    return _core_trace(x, offset, axis1=(-2), axis2=(-1), dtype=dtype)

def _cross_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2)

@array_function_dispatch(_cross_dispatcher)
def cross(x1, x2, /, *, axis=(-1)):
    """\n    Returns the cross product of 3-element vectors.\n\n    If ``x1`` and/or ``x2`` are multi-dimensional arrays, then\n    the cross-product of each pair of corresponding 3-element vectors\n    is independently computed.\n\n    This function is Array API compatible, contrary to\n    :func:`numpy.cross`.\n\n    Parameters\n    ----------\n    x1 : array_like\n        The first input array.\n    x2 : array_like\n        The second input array. Must be compatible with ``x1`` for all\n        non-compute axes. The size of the axis over which to compute\n        the cross-product must be the same size as the respective axis\n        in ``x1``.\n    axis : int, optional\n        The axis (dimension) of ``x1`` and ``x2`` containing the vectors for\n        which to compute the cross-product. Default: ``-1``.\n\n    Returns\n    -------\n    out : ndarray\n        An array containing the cross products.\n\n    See Also\n    --------\n    numpy.cross\n\n    Examples\n    --------\n    Vector cross-product.\n\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([4, 5, 6])\n    >>> np.linalg.cross(x, y)\n    array([-3,  6, -3])\n\n    Multiple vector cross-products. Note that the direction of the cross\n    product vector is defined by the *right-hand rule*.\n\n    >>> x = np.array([[1,2,3], [4,5,6]])\n    >>> y = np.array([[4,5,6], [1,2,3]])\n    >>> np.linalg.cross(x, y)\n    array([[-3,  6, -3],\n           [ 3, -6,  3]])\n\n    >>> x = np.array([[1, 2], [3, 4], [5, 6]])\n    >>> y = np.array([[4, 5], [6, 1], [2, 3]])\n    >>> np.linalg.cross(x, y, axis=0)\n    array([[-24,  6],\n           [ 18, 24],\n           [-6,  -18]])\n\n    """  # inserted
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    if x1.shape[axis]!= 3 or x2.shape[axis]!= 3:
        raise ValueError(f'Both input arrays must be (arrays of) 3-dimensional vectors, but they are {x1.shape[axis]} and {x2.shape[axis]} dimensional instead.')
    return _core_cross(x1, x2, axis=axis)

def _matmul_dispatcher(x1, x2, /):
    return (x1, x2)

@array_function_dispatch(_matmul_dispatcher)
def matmul(x1, x2, /):
    """\n    Computes the matrix product.\n\n    This function is Array API compatible, contrary to\n    :func:`numpy.matmul`.\n\n    Parameters\n    ----------\n    x1 : array_like\n        The first input array.\n    x2 : array_like\n        The second input array.\n\n    Returns\n    -------\n    out : ndarray\n        The matrix product of the inputs.\n        This is a scalar only when both ``x1``, ``x2`` are 1-d vectors.\n\n    Raises\n    ------\n    ValueError\n        If the last dimension of ``x1`` is not the same size as\n        the second-to-last dimension of ``x2``.\n\n        If a scalar value is passed in.\n\n    See Also\n    --------\n    numpy.matmul\n\n    Examples\n    --------\n    For 2-D arrays it is the matrix product:\n\n    >>> a = np.array([[1, 0],\n    ...               [0, 1]])\n    >>> b = np.array([[4, 1],\n    ...               [2, 2]])\n    >>> np.linalg.matmul(a, b)\n    array([[4, 1],\n           [2, 2]])\n\n    For 2-D mixed with 1-D, the result is the usual.\n\n    >>> a = np.array([[1, 0],\n    ...               [0, 1]])\n    >>> b = np.array([1, 2])\n    >>> np.linalg.matmul(a, b)\n    array([1, 2])\n    >>> np.linalg.matmul(b, a)\n    array([1, 2])\n\n\n    Broadcasting is conventional for stacks of arrays\n\n    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))\n    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))\n    >>> np.linalg.matmul(a,b).shape\n    (2, 2, 2)\n    >>> np.linalg.matmul(a, b)[0, 1, 1]\n    98\n    >>> sum(a[0, 1, :] * b[0 , :, 1])\n    98\n\n    Vector, vector returns the scalar inner product, but neither argument\n    is complex-conjugated:\n\n    >>> np.linalg.matmul([2j, 3j], [2j, 3j])\n    (-13+0j)\n\n    Scalar multiplication raises an error.\n\n    >>> np.linalg.matmul([1,2], 3)\n    Traceback (most recent call last):\n    ...\n    ValueError: matmul: Input operand 1 does not have enough dimensions ...\n\n    """  # inserted
    return _core_matmul(x1, x2)

def _tensordot_dispatcher(x1, x2, /, *, axes=None):
    return (x1, x2)

@array_function_dispatch(_tensordot_dispatcher)
def tensordot(x1, x2, /, *, axes=2):
    return _core_tensordot(x1, x2, axes=axes)
tensordot.__doc__ = _core_tensordot.__doc__

def _matrix_transpose_dispatcher(x):
    return (x,)

@array_function_dispatch(_matrix_transpose_dispatcher)
def matrix_transpose(x, /):
    return _core_matrix_transpose(x)
matrix_transpose.__doc__ = _core_matrix_transpose.__doc__

def _matrix_norm_dispatcher(x, /, *, keepdims=None, ord=None):
    return (x,)

@array_function_dispatch(_matrix_norm_dispatcher)
def matrix_norm(x, /, *, keepdims=False, ord='fro'):
    """\n    Computes the matrix norm of a matrix (or a stack of matrices) ``x``.\n\n    This function is Array API compatible.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array having shape (..., M, N) and whose two innermost\n        dimensions form ``MxN`` matrices.\n    keepdims : bool, optional\n        If this is set to True, the axes which are normed over are left in\n        the result as dimensions with size one. Default: False.\n    ord : {1, -1, 2, -2, inf, -inf, \'fro\', \'nuc\'}, optional\n        The order of the norm. For details see the table under ``Notes``\n        in `numpy.linalg.norm`.\n\n    See Also\n    --------\n    numpy.linalg.norm : Generic norm function\n\n    Examples\n    --------\n    >>> from numpy import linalg as LA\n    >>> a = np.arange(9) - 4\n    >>> a\n    array([-4, -3, -2, ...,  2,  3,  4])\n    >>> b = a.reshape((3, 3))\n    >>> b\n    array([[-4, -3, -2],\n           [-1,  0,  1],\n           [ 2,  3,  4]])\n\n    >>> LA.matrix_norm(b)\n    7.745966692414834\n    >>> LA.matrix_norm(b, ord=\'fro\')\n    7.745966692414834\n    >>> LA.matrix_norm(b, ord=np.inf)\n    9.0\n    >>> LA.matrix_norm(b, ord=-np.inf)\n    2.0\n\n    >>> LA.matrix_norm(b, ord=1)\n    7.0\n    >>> LA.matrix_norm(b, ord=-1)\n    6.0\n    >>> LA.matrix_norm(b, ord=2)\n    7.3484692283495345\n    >>> LA.matrix_norm(b, ord=-2)\n    1.8570331885190563e-016 # may vary\n\n    """  # inserted
    x = asanyarray(x)
    return norm(x, axis=((-2), (-1)), keepdims=keepdims, ord=ord)

def _vector_norm_dispatcher(x, /, *, axis=None, keepdims=None, ord=None):
    return (x,)

@array_function_dispatch(_vector_norm_dispatcher)
def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """\n    Computes the vector norm of a vector (or batch of vectors) ``x``.\n\n    This function is Array API compatible.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    axis : {None, int, 2-tuple of ints}, optional\n        If an integer, ``axis`` specifies the axis (dimension) along which\n        to compute vector norms. If an n-tuple, ``axis`` specifies the axes\n        (dimensions) along which to compute batched vector norms. If ``None``,\n        the vector norm must be computed over all array values (i.e.,\n        equivalent to computing the vector norm of a flattened array).\n        Default: ``None``.\n    keepdims : bool, optional\n        If this is set to True, the axes which are normed over are left in\n        the result as dimensions with size one. Default: False.\n    ord : {int, float, inf, -inf}, optional\n        The order of the norm. For details see the table under ``Notes``\n        in `numpy.linalg.norm`.\n\n    See Also\n    --------\n    numpy.linalg.norm : Generic norm function\n\n    Examples\n    --------\n    >>> from numpy import linalg as LA\n    >>> a = np.arange(9) + 1\n    >>> a\n    array([1, 2, 3, 4, 5, 6, 7, 8, 9])\n    >>> b = a.reshape((3, 3))\n    >>> b\n    array([[1, 2, 3],\n           [4, 5, 6],\n           [7, 8, 9]])\n\n    >>> LA.vector_norm(b)\n    16.881943016134134\n    >>> LA.vector_norm(b, ord=np.inf)\n    9.0\n    >>> LA.vector_norm(b, ord=-np.inf)\n    1.0\n\n    >>> LA.vector_norm(b, ord=0)\n    9.0\n    >>> LA.vector_norm(b, ord=1)\n    45.0\n    >>> LA.vector_norm(b, ord=-1)\n    0.3534857623790153\n    >>> LA.vector_norm(b, ord=2)\n    16.881943016134134\n    >>> LA.vector_norm(b, ord=-2)\n    0.8058837395885292\n\n    """  # inserted
    x = asanyarray(x)
    shape = list(x.shape)
    if axis is None:
        x = x.ravel()
        _axis = 0
    else:  # inserted
        if isinstance(axis, tuple):
            normalized_axis = normalize_axis_tuple(axis, x.ndim)
            rest = tuple((i for i in range(x.ndim) if i not in normalized_axis))
            newshape = axis + rest
            x = _core_transpose(x, newshape).reshape((prod([x.shape[i] for i in axis], dtype=int), *[x.shape[i] for i in rest]))
            _axis = 0
        else:  # inserted
            _axis = axis
    res = norm(x, axis=_axis, ord=ord)
    if keepdims:
        _axis = normalize_axis_tuple(range(len(shape)) if axis is None else axis, len(shape))
        for i in _axis:
            shape[i] = 1
        res = res.reshape(tuple(shape))
    return res

def _vecdot_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2)

@array_function_dispatch(_vecdot_dispatcher)
def vecdot(x1, x2, /, *, axis=(-1)):
    """\n    Computes the vector dot product.\n\n    This function is restricted to arguments compatible with the Array API,\n    contrary to :func:`numpy.vecdot`.\n\n    Let :math:`\\mathbf{a}` be a vector in ``x1`` and :math:`\\mathbf{b}` be\n    a corresponding vector in ``x2``. The dot product is defined as:\n\n    .. math::\n       \\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=0}^{n-1} \\overline{a_i}b_i\n\n    over the dimension specified by ``axis`` and where :math:`\\overline{a_i}`\n    denotes the complex conjugate if :math:`a_i` is complex and the identity\n    otherwise.\n\n    Parameters\n    ----------\n    x1 : array_like\n        First input array.\n    x2 : array_like\n        Second input array.\n    axis : int, optional\n        Axis over which to compute the dot product. Default: ``-1``.\n\n    Returns\n    -------\n    output : ndarray\n        The vector dot product of the input.\n\n    See Also\n    --------\n    numpy.vecdot\n\n    Examples\n    --------\n    Get the projected size along a given normal for an array of vectors.\n\n    >>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])\n    >>> n = np.array([0., 0.6, 0.8])\n    >>> np.linalg.vecdot(v, n)\n    array([ 3.,  8., 10.])\n\n    """  # inserted
    return _core_vecdot(x1, x2, axis=axis)
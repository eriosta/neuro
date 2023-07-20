# first line: 157
def fastica(
    X,
    n_components=None,
    *,
    algorithm="parallel",
    whiten="warn",
    fun="logcosh",
    fun_args=None,
    max_iter=200,
    tol=1e-04,
    w_init=None,
    whiten_solver="svd",
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
):
    """Perform Fast Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default="warn"
        Specify the whitening strategy to use.

        - If 'arbitrary-variance' (default), a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. deprecated:: 1.1
            Starting in v1.3, `whiten='unit-variance'` will be used by default.
            `whiten=True` is deprecated from 1.1 and will raise ValueError in 1.3.
            Use `whiten=arbitrary-variance` instead.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        Maximum number of iterations to perform.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : ndarray of shape (n_components, n_components), default=None
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

        .. versionadded:: 1.2

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_X_mean : bool, default=False
        If True, X_mean is returned too.

    compute_sources : bool, default=True
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    K : ndarray of shape (n_components, n_features) or None
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : ndarray of shape (n_components, n_components)
        The square matrix that unmixes the data after whitening.
        The mixing matrix is the pseudo-inverse of matrix ``W K``
        if K is not None, else it is the inverse of W.

    S : ndarray of shape (n_samples, n_components) or None
        Estimated source matrix.

    X_mean : ndarray of shape (n_features,)
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``
    While FastICA was proposed to estimate as many sources
    as features, it is possible to estimate less by setting
    n_components < n_features. It this case K is not a square matrix
    and the estimated A is the pseudo-inverse of ``W K``.

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, "Fast Independent Component Analysis",
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.
    """
    est = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        fun_args=fun_args,
        max_iter=max_iter,
        tol=tol,
        w_init=w_init,
        whiten_solver=whiten_solver,
        random_state=random_state,
    )
    S = est._fit_transform(X, compute_sources=compute_sources)

    if est._whiten in ["unit-variance", "arbitrary-variance"]:
        K = est.whitening_
        X_mean = est.mean_
    else:
        K = None
        X_mean = None

    returned_values = [K, est._unmixing, S]
    if return_X_mean:
        returned_values.append(X_mean)
    if return_n_iter:
        returned_values.append(est.n_iter_)

    return returned_values

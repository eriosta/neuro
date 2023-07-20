# first line: 738
def dict_learning_online(
    X,
    n_components=2,
    *,
    alpha=1,
    n_iter="deprecated",
    max_iter=None,
    return_code=True,
    dict_init=None,
    callback=None,
    batch_size="warn",
    verbose=False,
    shuffle=True,
    n_jobs=None,
    method="lars",
    iter_offset="deprecated",
    random_state=None,
    return_inner_stats="deprecated",
    inner_stats="deprecated",
    return_n_iter="deprecated",
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
    tol=1e-3,
    max_no_improvement=10,
):
    """Solve a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.
    This is accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    n_components : int or None, default=2
        Number of dictionary atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter.

    n_iter : int, default=100
        Number of mini-batch iterations to perform.

        .. deprecated:: 1.1
           `n_iter` is deprecated in 1.1 and will be removed in 1.4. Use
           `max_iter` instead.

    max_iter : int, default=None
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
        If ``max_iter`` is not None, ``n_iter`` is ignored.

        .. versionadded:: 1.1

    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary for warm restart scenarios.
        If `None`, the initial values for the dictionary are created
        with an SVD decomposition of the data via :func:`~sklearn.utils.randomized_svd`.

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

    batch_size : int, default=3
        The number of samples to take in each batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` will change from 3 to 256 in version 1.3.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    iter_offset : int, default=0
        Number of previous iterations completed on the dictionary used for
        initialization.

        .. deprecated:: 1.1
           `iter_offset` serves internal purpose only and will be removed in 1.3.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_inner_stats : bool, default=False
        Return the inner statistics A (dictionary covariance) and B
        (data approximation). Useful to restart the algorithm in an
        online setting. If `return_inner_stats` is `True`, `return_code` is
        ignored.

        .. deprecated:: 1.1
           `return_inner_stats` serves internal purpose only and will be removed in 1.3.

    inner_stats : tuple of (A, B) ndarrays, default=None
        Inner sufficient statistics that are kept by the algorithm.
        Passing them at initialization is useful in online settings, to
        avoid losing the history of the evolution.
        `A` `(n_components, n_components)` is the dictionary covariance matrix.
        `B` `(n_features, n_components)` is the data approximation matrix.

        .. deprecated:: 1.1
           `inner_stats` serves internal purpose only and will be removed in 1.3.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

        .. deprecated:: 1.1
           `return_n_iter` will be removed in 1.3 and n_iter will always be returned.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform when solving the lasso problem.

        .. versionadded:: 0.22

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps. Used only if `max_iter` is not None.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function. Used only if
        `max_iter` is not None.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Returns
    -------
    code : ndarray of shape (n_samples, n_components),
        The sparse code (only returned if `return_code=True`).

    dictionary : ndarray of shape (n_components, n_features),
        The solutions to the dictionary learning problem.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    See Also
    --------
    dict_learning : Solve a dictionary learning matrix factorization problem.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate, version of the dictionary
        learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    """
    deps = (return_n_iter, return_inner_stats, iter_offset, inner_stats)
    if max_iter is not None and not all(arg == "deprecated" for arg in deps):
        raise ValueError(
            "The following arguments are incompatible with 'max_iter': "
            "return_n_iter, return_inner_stats, iter_offset, inner_stats"
        )

    iter_offset = _check_warn_deprecated(iter_offset, "iter_offset", default=0)
    return_inner_stats = _check_warn_deprecated(
        return_inner_stats,
        "return_inner_stats",
        default=False,
        additional_message="From 1.3 inner_stats will never be returned.",
    )
    inner_stats = _check_warn_deprecated(inner_stats, "inner_stats", default=None)
    return_n_iter = _check_warn_deprecated(
        return_n_iter,
        "return_n_iter",
        default=False,
        additional_message=(
            "From 1.3 'n_iter' will never be returned. Refer to the 'n_iter_' and "
            "'n_steps_' attributes of the MiniBatchDictionaryLearning object instead."
        ),
    )

    if max_iter is not None:
        transform_algorithm = "lasso_" + method

        est = MiniBatchDictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            n_iter=n_iter,
            n_jobs=n_jobs,
            fit_algorithm=method,
            batch_size=batch_size,
            shuffle=shuffle,
            dict_init=dict_init,
            random_state=random_state,
            transform_algorithm=transform_algorithm,
            transform_alpha=alpha,
            positive_code=positive_code,
            positive_dict=positive_dict,
            transform_max_iter=method_max_iter,
            verbose=verbose,
            callback=callback,
            tol=tol,
            max_no_improvement=max_no_improvement,
        ).fit(X)

        if not return_code:
            return est.components_
        else:
            code = est.transform(X)
            return code, est.components_

    # TODO remove the whole old behavior in 1.3
    # Fallback to old behavior

    n_iter = _check_warn_deprecated(
        n_iter, "n_iter", default=100, additional_message="Use 'max_iter' instead."
    )

    if batch_size == "warn":
        warnings.warn(
            "The default value of batch_size will change from 3 to 256 in 1.3.",
            FutureWarning,
        )
        batch_size = 3

    if n_components is None:
        n_components = X.shape[1]

    if method not in ("lars", "cd"):
        raise ValueError("Coding method not supported as a fit algorithm.")

    _check_positive_coding(method, positive_code)

    method = "lasso_" + method

    t0 = time.time()
    n_samples, n_features = X.shape
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init V with SVD of X
    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(X, n_components, random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[
            dictionary,
            np.zeros((n_components - r, dictionary.shape[1]), dtype=dictionary.dtype),
        ]

    if verbose == 1:
        print("[dict_learning]", end=" ")

    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else:
        X_train = X

    X_train = check_array(
        X_train, order="C", dtype=[np.float64, np.float32], copy=False
    )

    # Fortran-order dict better suited for the sparse coding which is the
    # bottleneck of this algorithm.
    dictionary = check_array(dictionary, order="F", dtype=X_train.dtype, copy=False)
    dictionary = np.require(dictionary, requirements="W")

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    # The covariance of the dictionary
    if inner_stats is None:
        A = np.zeros((n_components, n_components), dtype=X_train.dtype)
        # The data approximation
        B = np.zeros((n_features, n_components), dtype=X_train.dtype)
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

    # If n_iter is zero, we need to return zero.
    ii = iter_offset - 1

    for ii, batch in zip(range(iter_offset, iter_offset + n_iter), batches):
        this_X = X_train[batch]
        dt = time.time() - t0
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            if verbose > 10 or ii % ceil(100.0 / verbose) == 0:
                print(
                    "Iteration % 3i (elapsed time: % 3is, % 4.1fmn)" % (ii, dt, dt / 60)
                )

        this_code = sparse_encode(
            this_X,
            dictionary,
            algorithm=method,
            alpha=alpha,
            n_jobs=n_jobs,
            check_input=False,
            positive=positive_code,
            max_iter=method_max_iter,
            verbose=verbose,
        )

        # Update the auxiliary variables
        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else:
            theta = float(batch_size**2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code.T, this_code)
        B *= beta
        B += np.dot(this_X.T, this_code)

        # Update dictionary in place
        _update_dict(
            dictionary,
            this_X,
            this_code,
            A,
            B,
            verbose=verbose,
            random_state=random_state,
            positive=positive_dict,
        )

        # Maybe we need a stopping criteria based on the amount of
        # modification in the dictionary
        if callback is not None:
            callback(locals())

    if return_inner_stats:
        if return_n_iter:
            return dictionary, (A, B), ii - iter_offset + 1
        else:
            return dictionary, (A, B)
    if return_code:
        if verbose > 1:
            print("Learning code...", end=" ")
        elif verbose == 1:
            print("|", end=" ")
        code = sparse_encode(
            X,
            dictionary,
            algorithm=method,
            alpha=alpha,
            n_jobs=n_jobs,
            check_input=False,
            positive=positive_code,
            max_iter=method_max_iter,
            verbose=verbose,
        )
        if verbose > 1:
            dt = time.time() - t0
            print("done (total time: % 3is, % 4.1fmn)" % (dt, dt / 60))
        if return_n_iter:
            return code, dictionary, ii - iter_offset + 1
        else:
            return code, dictionary

    if return_n_iter:
        return dictionary, ii - iter_offset + 1
    else:
        return dictionary

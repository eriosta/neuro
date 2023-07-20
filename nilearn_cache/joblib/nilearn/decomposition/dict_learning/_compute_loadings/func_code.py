# first line: 26
def _compute_loadings(components, data):
    ridge = Ridge(fit_intercept=False, alpha=1e-8)
    ridge.fit(components.T, np.asarray(data.T))
    loadings = ridge.coef_.T

    S = np.sqrt(np.sum(loadings**2, axis=0))
    S[S == 0] = 1
    loadings /= S[np.newaxis, :]
    return loadings

def make_ci95(v):
    'return tuple with 95 percent confidence interval for the value in np.array v'
    n_samples = 10000
    samples = np.random.choice(v, size=n_samples, replace=True)  # draw with replacement
    sorted_samples = np.sort(samples)
    ci = (sorted_samples[int(n_samples * 0.025) - 1], sorted_samples[int(n_samples * 0.975) - 1])
    return ci

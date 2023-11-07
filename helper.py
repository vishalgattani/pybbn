from scipy.stats import binom


def get_binomial_prob(n, p):
    """
    Given number of experiments to be performed and the prior probability of success, returns the list of probabilities of successes for k trials out of total n runs

    Args:
    n (_type_): number of experiments
    p (_type_): probability of success

    Returns:
    _type_: list of probabilities of success for k out of n runs
    """
    return list(binom.pmf(list(range(n + 1)), n, p))


def get_cdf_binomial_prob(n, p):
    """Given number of experiments to be performed and the prior probability of success, returns the list of cumulative probabilities of successes for k trials out of total n runs

    Args:
        total_exp_runs (_type_): number of experiments
        p (_type_): probability of success

    Returns:
        _type_: list of cumulative probabilities of success up till k out of n runs
    """
    return list(binom.cdf(list(range(n + 1)), n, p))

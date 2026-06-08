# Author: Vishal Gattani
# Created: 2024-06-07

from typing import List

from scipy.stats import binom


def get_binomial_prob(n: int, p: float) -> List[float]:
    """Given number of experiments and prior probability of success,
    returns the list of probabilities of successes for k trials out of total n runs.

    Args:
        n: number of experiments
        p: probability of success

    Returns:
        List of probabilities of success for k out of n runs.
    """
    return list(binom.pmf(list(range(n + 1)), n, p))


def get_cdf_binomial_prob(n: int, p: float) -> List[float]:
    """Given number of experiments and prior probability of success,
    returns the list of cumulative probabilities of successes for k trials out of total n runs.

    Args:
        n: number of experiments
        p: probability of success

    Returns:
        List of cumulative probabilities of success up till k out of n runs.
    """
    return list(binom.cdf(list(range(n + 1)), n, p))

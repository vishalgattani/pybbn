"""Tests for pybbn_assurance/helper.py — binomial & CDF math functions."""

import pytest

from pybbn_assurance.helper import get_binomial_prob, get_cdf_binomial_prob


class TestGetBinomialProb:
    """Tests for get_binomial_prob."""

    def test_returns_correct_length(self):
        """List should contain n+1 entries (k=0..n)."""
        result = get_binomial_prob(n=5, p=0.5)
        assert len(result) == 6  # 0, 1, 2, 3, 4, 5

    def test_probabilities_sum_to_one(self):
        """The PMF must sum to 1.0."""
        result = get_binomial_prob(n=10, p=0.3)
        assert sum(result) == pytest.approx(1.0, abs=1e-10)

    def test_single_experiment(self):
        """n=1 should give [1-p, p]."""
        result = get_binomial_prob(n=1, p=0.7)
        assert result[0] == pytest.approx(0.3)
        assert result[1] == pytest.approx(0.7)

    def test_zero_experiments(self):
        """n=0 should return [1.0]."""
        result = get_binomial_prob(n=0, p=0.5)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_certain_success(self):
        """p=1.0 → all probability mass on k=n."""
        result = get_binomial_prob(n=5, p=1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(1.0)

    def test_certain_failure(self):
        """p=0.0 → all probability mass on k=0."""
        result = get_binomial_prob(n=5, p=0.0)
        assert result[0] == pytest.approx(1.0)
        assert result[5] == pytest.approx(0.0)

    def test_fair_coin_symmetry(self):
        """p=0.5 with n=10 should be symmetric around the mean."""
        result = get_binomial_prob(n=10, p=0.5)
        for i in range(len(result)):
            assert result[i] == pytest.approx(result[len(result) - 1 - i], abs=1e-10)

    def test_expected_value(self):
        """E[k] = n*p should hold approximately."""
        n, p = 20, 0.3
        result = get_binomial_prob(n, p)
        expected = sum(k * prob for k, prob in enumerate(result))
        assert expected == pytest.approx(n * p, abs=0.5)

    def test_large_n_runs(self):
        """Should handle larger n without error."""
        result = get_binomial_prob(n=100, p=0.5)
        assert len(result) == 101
        assert sum(result) == pytest.approx(1.0, abs=1e-8)

    def test_all_values_non_negative(self):
        """All probabilities should be >= 0."""
        result = get_binomial_prob(n=10, p=0.25)
        assert all(v >= 0 for v in result)


class TestGetCdfBinomialProb:
    """Tests for get_cdf_binomial_prob."""

    def test_returns_correct_length(self):
        """List should contain n+1 entries."""
        result = get_cdf_binomial_prob(n=5, p=0.5)
        assert len(result) == 6

    def test_cdf_is_monotonically_non_decreasing(self):
        """CDF must never decrease."""
        result = get_cdf_binomial_prob(n=10, p=0.3)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    def test_cdf_ends_at_one(self):
        """Final CDF value must be 1.0."""
        result = get_cdf_binomial_prob(n=7, p=0.4)
        assert result[-1] == pytest.approx(1.0, abs=1e-10)

    def test_cdf_starts_at_pmf_k0(self):
        """CDF[0] should equal PMF[0]."""
        pmf = get_binomial_prob(n=5, p=0.3)
        cdf = get_cdf_binomial_prob(n=5, p=0.3)
        assert cdf[0] == pytest.approx(pmf[0])

    def test_cdf_equals_cumulative_pmf(self):
        """CDF[k] should equal sum of PMF[0..k]."""
        pmf = get_binomial_prob(n=6, p=0.4)
        cdf = get_cdf_binomial_prob(n=6, p=0.4)
        for k in range(len(cdf)):
            expected = sum(pmf[: k + 1])
            assert cdf[k] == pytest.approx(expected, abs=1e-10)

    def test_zero_experiments(self):
        """n=0 should return [1.0]."""
        result = get_cdf_binomial_prob(n=0, p=0.5)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_certain_success_cdf(self):
        """p=1.0 → CDF should be [0,0,...,0,1]."""
        result = get_cdf_binomial_prob(n=5, p=1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(1.0)

    def test_certain_failure_cdf(self):
        """p=0.0 → CDF should be [1,1,...,1]."""
        result = get_cdf_binomial_prob(n=5, p=0.0)
        assert all(v == pytest.approx(1.0) for v in result)

    def test_all_values_between_zero_and_one(self):
        """All CDF values must be in [0, 1]."""
        result = get_cdf_binomial_prob(n=15, p=0.7)
        assert all(0.0 <= v <= 1.0 for v in result)

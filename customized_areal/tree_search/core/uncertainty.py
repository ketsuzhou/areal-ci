"""Bayesian uncertainty estimation for dynamic group size allocation."""

from __future__ import annotations


def compute_query_uncertainty(
    episode_rewards: list[float],
    episode_steps: list[int],
    reward_type: str,
) -> float:
    """Compute step-adjusted Bayesian posterior variance for a query.

    Returns U(q) = posterior_var * mean_steps.
    Returns float('inf') when uncertainty cannot be estimated (0 episodes).

    Binary rewards use a Beta(1,1) posterior.
    Continuous rewards use a Normal-Inverse-Gamma posterior with weak prior.
    """
    n = len(episode_rewards)
    if n == 0:
        return float("inf")

    mean_steps = sum(episode_steps) / n

    if reward_type == "binary":
        s = sum(1 for r in episode_rewards if r > 0)
        alpha = 1 + s
        beta = 1 + n - s
        posterior_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    else:
        # Normal-Inverse-Gamma posterior with weak prior
        mu0 = 0.0
        kappa0 = 1e-3
        alpha0 = 2.0
        beta0 = 1.0

        r_bar = sum(episode_rewards) / n
        ss = sum((r - r_bar) ** 2 for r in episode_rewards)

        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * r_bar) / kappa_n
        alpha_n = alpha0 + n / 2
        beta_n = (
            beta0
            + 0.5 * ss
            + (kappa0 * n * (r_bar - mu0) ** 2) / (2 * kappa_n)
        )

        # Posterior variance of the latent mean
        posterior_var = beta_n / ((alpha_n - 1) * kappa_n)

    return posterior_var * mean_steps


def should_discard_query(episode_rewards: list[float]) -> bool:
    """Return True if all rewards are identical (no learning signal).

    Discards only when len(rewards) >= 2 and all rewards are identical.
    Single-episode queries are never discarded by this rule.
    Empty lists are discarded.
    """
    if len(episode_rewards) == 0:
        return True
    if len(episode_rewards) < 2:
        return False
    first = episode_rewards[0]
    return all(abs(r - first) < 1e-9 for r in episode_rewards)

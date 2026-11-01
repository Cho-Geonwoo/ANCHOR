"""ANCHOR: fixed offline-policy KL and a linearly annealed coefficient."""

import jax
import jax.numpy as jnp
from wsrl.agents.cql import CQLAgent


def gaussian_parameters(distribution):
    """Unwrap a tanh Gaussian; KL is invariant to the shared bijection."""
    for _ in range(8):
        if hasattr(distribution, "loc") and hasattr(distribution, "scale_diag"):
            return distribution.loc, distribution.scale_diag
        if not hasattr(distribution, "distribution"):
            break
        distribution = distribution.distribution
    raise TypeError("ANCHOR requires a diagonal Gaussian policy")


def diagonal_gaussian_kl(loc, scale, reference_loc, reference_scale):
    scale = jnp.clip(scale, 1e-6, 1e6)
    reference_scale = jnp.clip(reference_scale, 1e-6, 1e6)
    return jnp.sum(
        jnp.log(reference_scale / scale)
        + (scale**2 + (loc - reference_loc) ** 2) / (2.0 * reference_scale**2)
        - 0.5,
        axis=-1,
    )


class AnchorAgent(CQLAgent):
    def policy_loss_fn(self, batch, params, rng):
        loss, info = super().policy_loss_fn(batch, params, rng)
        if "kl_reg_weight" not in batch:
            return loss, info  # Ordinary CQL during offline pretraining.
        dist = self.forward_policy(batch["observations"], rng=rng, grad_params=params)
        loc, scale = gaussian_parameters(dist)
        kl = diagonal_gaussian_kl(
            loc,
            scale,
            jax.lax.stop_gradient(batch["reference_policy_loc"]),
            jax.lax.stop_gradient(batch["reference_policy_stds"]),
        ).mean()
        loss = loss + batch["kl_reg_weight"] * kl
        return loss, {
            **info,
            "actor_loss": loss,
            "kl_loss": kl,
            "kl_weight": batch["kl_reg_weight"],
        }

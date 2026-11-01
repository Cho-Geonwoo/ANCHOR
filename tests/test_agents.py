import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from train import make_online_agent
from wsrl.agents import agents
from wsrl.agents.anchor import diagonal_gaussian_kl, gaussian_parameters
from wsrl.agents.cql import CQLAgent
from wsrl.benchmark import agent_config


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.obs = jnp.zeros((8, 5))
        self.actions = jnp.zeros((8, 2))
        self.batch = dict(
            observations=self.obs,
            next_observations=self.obs + 0.1,
            actions=self.actions,
            rewards=jnp.zeros(8),
            masks=jnp.ones(8),
        )

    def create(self, method, env="kitchen-partial-v0"):
        cfg = agent_config(method, env, small=True)
        return (
            agents[method].create(
                jax.random.PRNGKey(1), self.obs, self.actions, encoder_def=None, **cfg
            ),
            cfg,
        )

    def test_kl_direction_and_zero(self):
        a = diagonal_gaussian_kl(
            jnp.zeros((1, 2)), jnp.ones((1, 2)), jnp.zeros((1, 2)), 2 * jnp.ones((1, 2))
        )
        self.assertAlmostEqual(float(a[0]), 2 * (np.log(2) + 0.125 - 0.5), places=6)
        self.assertEqual(
            float(
                diagonal_gaussian_kl(
                    self.actions, self.actions + 1, self.actions, self.actions + 1
                ).sum()
            ),
            0.0,
        )

    def test_zero_weight_equals_base_loss_and_reference_has_no_gradient(self):
        agent, cfg = self.create("anchor")
        loc, scale = gaussian_parameters(agent.forward_policy(self.obs, train=False))
        batch = dict(
            self.batch,
            reference_policy_loc=loc + 1,
            reference_policy_stds=scale,
            kl_reg_weight=jnp.asarray(0.0),
        )
        rng = jax.random.PRNGKey(3)
        loss, _ = agent.policy_loss_fn(batch, agent.state.params, rng)
        base, _ = CQLAgent(agent.state, agent.config).policy_loss_fn(batch, agent.state.params, rng)
        np.testing.assert_allclose(loss, base)
        batch["kl_reg_weight"] = jnp.asarray(1.0)
        ref_gradient = jax.grad(
            lambda ref: agent.policy_loss_fn(
                dict(batch, reference_policy_loc=ref), agent.state.params, rng
            )[0]
        )(loc)
        np.testing.assert_array_equal(ref_gradient, jnp.zeros_like(loc))

    def test_anchor_freezes_reference_and_high_utd_accepts_scalar(self):
        agent, cfg = self.create("anchor")
        frozen_bytes = serialization.to_bytes(agent)
        online = make_online_agent(agent, "anchor", cfg, self.obs, self.actions, 10)
        self.assertEqual(online.config["cql_alpha"], 1.0)
        self.assertEqual(agent.config["cql_alpha"], 5.0)
        loc, scale = gaussian_parameters(agent.forward_policy(self.obs, train=False))
        batch = dict(
            self.batch,
            reference_policy_loc=loc,
            reference_policy_stds=scale,
            kl_reg_weight=jnp.asarray(0.5),
        )
        online, info = online.update_high_utd(batch, utd_ratio=2)
        self.assertEqual(serialization.to_bytes(agent), frozen_bytes)
        self.assertTrue(np.isfinite(float(info["actor"]["actor_loss"])))
        self.assertIn("kl_loss", info["actor"])

    def test_porl_transfers_actor_but_reinitializes_critic(self):
        offline, cfg = self.create("porl", "antmaze-large-play-v2")
        online = make_online_agent(offline, "porl", cfg, self.obs, self.actions, 10)
        self.assertFalse(online.config["use_cql_loss"])
        for a, b in zip(
            jax.tree_util.tree_leaves(offline.state.params["modules_actor"]),
            jax.tree_util.tree_leaves(online.state.params["modules_actor"]),
        ):
            np.testing.assert_array_equal(a, b)
        old = jax.tree_util.tree_leaves(offline.state.params["modules_critic"])
        new = jax.tree_util.tree_leaves(online.state.params["modules_critic"])
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(old, new)))
        batch = dict(self.batch)
        next_actions, _ = online._compute_next_actions(batch, jax.random.PRNGKey(4))
        self.assertEqual(next_actions.shape, (8, 2))
        actor_before = jax.tree_util.tree_leaves(online.state.params["modules_actor"])
        online, _ = online.update_high_utd_critic(batch, utd_ratio=2)
        for before, after in zip(
            actor_before, jax.tree_util.tree_leaves(online.state.params["modules_actor"])
        ):
            np.testing.assert_array_equal(before, after)

    def test_critic_only_step_keeps_existing_actor_momentum_frozen(self):
        agent, _ = self.create("anchor")
        agent, _ = agent.update(self.batch)
        actor = jax.tree_util.tree_leaves(agent.state.params["modules_actor"])
        moments = jax.tree_util.tree_leaves(agent.state.opt_states["actor"])
        agent, _ = agent.update_critic(self.batch)
        for before, after in zip(
            actor, jax.tree_util.tree_leaves(agent.state.params["modules_actor"])
        ):
            np.testing.assert_array_equal(before, after)
        for before, after in zip(
            moments, jax.tree_util.tree_leaves(agent.state.opt_states["actor"])
        ):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()

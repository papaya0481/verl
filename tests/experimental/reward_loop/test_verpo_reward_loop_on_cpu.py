import numpy as np
from omegaconf import OmegaConf

from verl.experimental.reward_loop.reward_loop import aggregate_reward_extra_infos
from verl.experimental.reward_loop.reward_manager.verpo import VeRPORewardManager


def test_aggregate_reward_extra_infos_handles_mixed_schema():
    reward_extra_infos = [
        {"score": 1.0, "traj_reward": 1.0, "reward_valid": True},
        {"score": 0.0, "dense_reward": 0.0, "reward_valid": False},
    ]

    reward_extra_keys, non_tensor_batch = aggregate_reward_extra_infos(reward_extra_infos)

    assert set(reward_extra_keys) == {"score", "traj_reward", "dense_reward", "reward_valid"}
    assert np.allclose(non_tensor_batch["score"], np.array([1.0, 0.0]))
    assert np.allclose(non_tensor_batch["traj_reward"], np.array([1.0, 0.0]))
    assert np.allclose(non_tensor_batch["dense_reward"], np.array([0.0, 0.0]))
    assert non_tensor_batch["reward_valid"].tolist() == [True, False]


def test_verpo_score_raw_failure_returns_zero_signals():
    config = OmegaConf.create({"reward": {"reward_kwargs": {}}})
    manager = VeRPORewardManager(config=config, tokenizer=None, compute_score=None)

    scored = manager._score_raw(None, "")

    assert scored["reward_score"] == 0.0
    assert scored["reward_extra_info"] == {
        "score": 0.0,
        "acc": 0.0,
        "passed": 0,
        "total": 0,
        "pass_rate": 0.0,
        "dense_reward": 0.0,
        "traj_reward": 0.0,
        "outcome_reward": 0.0,
        "efficiency_decay": 0.0,
        "avg_difficulty_weight": 0.0,
        "density_sigma": 0.0,
        "reward_valid": False,
    }


def test_verpo_does_not_apply_overlong_penalty_to_failed_sample():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "max_resp_len": 10,
                    "overlong_buffer_cfg": {
                        "enable": True,
                        "len": 2,
                        "penalty_factor": 1.0,
                        "log": True,
                    }
                }
            }
        }
    )
    manager = VeRPORewardManager(config=config, tokenizer=None, compute_score=None)

    scored = manager._score_raw(None, "")
    penalized = manager._apply_overlong_penalty_to_scored(scored, valid_response_length=15)

    assert penalized["reward_score"] == 0.0
    assert penalized["reward_extra_info"]["traj_reward"] == 0.0
    assert penalized["reward_extra_info"]["dense_reward"] == 0.0
    assert penalized["reward_extra_info"]["score"] == 0.0
    assert penalized["reward_extra_info"]["overlong_reward"] == 0.0
    assert penalized["reward_extra_info"]["overlong"] is False
    assert penalized["reward_extra_info"]["reward_valid"] is False


def test_verpo_applies_overlong_penalty_to_dense_and_traj_rewards():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "max_resp_len": 10,
                    "overlong_buffer_cfg": {
                        "enable": True,
                        "len": 2,
                        "penalty_factor": 1.0,
                        "log": True,
                    },
                }
            }
        }
    )
    manager = VeRPORewardManager(config=config, tokenizer=None, compute_score=None)
    scored = {
        "reward_score": 1.0,
        "reward_extra_info": {
            "score": 1.0,
            "dense_reward": 0.5,
            "traj_reward": 1.0,
            "reward_valid": True,
        },
    }

    penalized = manager._apply_overlong_penalty_to_scored(scored, valid_response_length=10)

    assert penalized["reward_score"] == 0.0
    assert penalized["reward_extra_info"]["score"] == 0.0
    assert penalized["reward_extra_info"]["traj_reward"] == 0.0
    assert penalized["reward_extra_info"]["dense_reward"] == -0.5
    assert penalized["reward_extra_info"]["overlong_reward"] == -1.0
    assert penalized["reward_extra_info"]["overlong"] is True

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "dummy response"


def _build_data(ground_truth: str) -> DataProto:
    prompts = torch.tensor([[11, 12]])
    responses = torch.tensor([[21, 22, 23, 24]])
    attention_mask = torch.ones((1, 6), dtype=torch.long)

    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensors={
            "data_source": ["unit_test"],
            "reward_model": [{"ground_truth": ground_truth}],
            "extra_info": [{}],
        },
    )


def _compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    del data_source, solution_str, extra_info
    if ground_truth == "correct":
        return {"score": 1.0, "acc": 1.0, "all_passed": True}
    return {"score": -1.0, "acc": 0.0, "all_passed": False}


def _build_manager() -> DAPORewardManager:
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "max_resp_len": 4,
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
    return DAPORewardManager(config=config, tokenizer=_DummyTokenizer(), compute_score=_compute_score)


@pytest.mark.asyncio
async def test_dapo_does_not_apply_overlong_penalty_to_failed_sample():
    manager = _build_manager()

    result = await manager.run_single(_build_data("wrong"))

    assert result["reward_score"] == -1.0
    assert result["reward_extra_info"]["score"] == -1.0
    assert result["reward_extra_info"]["overlong_reward"] == 0.0
    assert result["reward_extra_info"]["overlong"] is False


@pytest.mark.asyncio
async def test_dapo_applies_overlong_penalty_to_positive_sample():
    manager = _build_manager()

    result = await manager.run_single(_build_data("correct"))

    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"]["score"] == 1.0
    assert result["reward_extra_info"]["overlong_reward"] == -1.0
    assert result["reward_extra_info"]["overlong"] is True

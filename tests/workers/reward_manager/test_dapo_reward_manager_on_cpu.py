from types import SimpleNamespace

import torch

from verl import DataProto
from verl.workers.reward_manager.dapo import DAPORewardManager


class _DummyTokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(int(token_id)) for token_id in token_ids)


def _build_batch() -> DataProto:
    prompts = torch.tensor(
        [
            [11, 12],
            [21, 22],
        ]
    )
    responses = torch.tensor(
        [
            [31, 32, 33, 34],
            [41, 42, 43, 44],
        ]
    )
    attention_mask = torch.ones((2, 6), dtype=torch.long)

    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensors={
            "data_source": ["unit_test", "unit_test"],
            "reward_model": [
                {"ground_truth": "correct"},
                {"ground_truth": "wrong"},
            ],
            "extra_info": [{}, {}],
        },
    )


def _compute_score(data_source, solution_str, ground_truth, extra_info):
    del data_source, solution_str, extra_info
    if ground_truth == "correct":
        return {"score": 1.0, "acc": 1.0, "all_passed": True}
    return {"score": -1.0, "acc": 0.0, "all_passed": False}


def test_overlong_penalty_only_applies_to_positive_scores():
    manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_compute_score,
        max_resp_len=4,
        overlong_buffer_cfg=SimpleNamespace(enable=True, len=2, penalty_factor=1.0, log=True),
    )

    result = manager(_build_batch(), return_dict=True)

    reward_tensor = result["reward_tensor"]
    reward_extra_info = result["reward_extra_info"]

    assert reward_tensor[0, 3].item() == 0.0
    assert reward_tensor[1, 3].item() == -1.0
    assert reward_extra_info["overlong_reward"] == [-1.0, 0.0]
    assert reward_extra_info["overlong"] == [True, False]
    assert reward_extra_info["acc"] == [1.0, 0.0]

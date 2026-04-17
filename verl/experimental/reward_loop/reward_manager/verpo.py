"""VeRPO reward manager for the experimental reward_loop pipeline.

Inherits the async run_single interface from RewardManagerBase but overrides
the batch-level scoring to compute group-level rho_j per the paper.

The key difference from DAPORewardManager:
- run_single is kept for interface compatibility (used by reward_loop.py).
- run_batch (called by reward_loop when available) executes all N samples for
  a prompt group together, computes group-level rho_j, then scores each sample.

When run_batch is not available in the caller, run_single falls back to the
EMA-based single-sample path in compute_score (backward compatible).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

from scripts.plugins.verl_codegen1_reward import execute_single


def _cfg_get(cfg, key, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return default


@register("verpo")
class VeRPORewardManager(RewardManagerBase):
    """VeRPO reward manager: group-level rho_j, R^turn as weighted sum."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None,
                 reward_model_tokenizer=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score_fn = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score_fn)

        reward_kwargs_cfg = _cfg_get(config.reward, "reward_kwargs", {}) or {}
        self.overlong_buffer_cfg = _cfg_get(reward_kwargs_cfg, "overlong_buffer_cfg", None)
        self.max_resp_len = _cfg_get(reward_kwargs_cfg, "max_resp_len", None)

        # VeRPO-specific kwargs (nested under reward_kwargs.reward_kwargs in yaml)
        inner_rk = _cfg_get(reward_kwargs_cfg, "reward_kwargs", {}) or {}
        self.timeout_sec = int(_cfg_get(inner_rk, "timeout_sec", 4))
        self.memory_limit_mb = int(_cfg_get(inner_rk, "memory_limit_mb", 1024))
        self.difficulty_alpha = float(_cfg_get(inner_rk, "difficulty_alpha", 2.0))
        self.density_eps = float(_cfg_get(inner_rk, "density_eps", 1e-6))
        self.density_sigma_floor = float(_cfg_get(inner_rk, "density_sigma_floor", 1e-3))
        self.efficiency_gamma = float(_cfg_get(inner_rk, "efficiency_gamma", 0.995))
        self.efficiency_mode = str(_cfg_get(inner_rk, "efficiency_mode", "turn_count"))
        self.num_workers = int(_cfg_get(reward_kwargs_cfg, "num_workers", 4))

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None
            assert self.max_resp_len >= self.overlong_buffer_cfg.len
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_response(self, data_item) -> tuple[str, int]:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_str, valid_response_length

    def _apply_overlong_penalty(self, reward: float, valid_response_length: int) -> tuple[float, dict]:
        extra = {}
        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            buf_len = self.overlong_buffer_cfg.len
            expected = self.max_resp_len - buf_len
            exceed = valid_response_length - expected
            penalty = min(-exceed / buf_len * self.overlong_buffer_cfg.penalty_factor, 0)
            reward += penalty
            if self.overlong_buffer_cfg.log:
                extra["overlong_reward"] = penalty
                extra["overlong"] = penalty < 0
        return reward, extra

    def _score_raw(self, raw: dict | None, response_str: str) -> dict:
        """Convert execute_single result into final reward dict."""
        if raw is None:
            return {"reward_score": 0.0, "reward_extra_info": {}}

        # Caller must have already set dense_reward on raw via group computation.
        dense_reward = raw.get("dense_reward", 0.0)

        gamma = self.efficiency_gamma
        if self.efficiency_mode == "turn_count":
            efficiency_decay = gamma ** max(raw["turn_count"], 1)
        else:
            tok = max(len(response_str.split()), 1)
            efficiency_decay = gamma ** max(tok - 1, 0)

        traj_reward = raw["outcome_reward"] * efficiency_decay
        score = traj_reward

        extra = {
            "score": score,
            "acc": raw["pass_rate"],
            "passed": raw["passed"],
            "total": raw["total"],
            "pass_rate": raw["pass_rate"],
            "dense_reward": dense_reward,
            "traj_reward": traj_reward,
            "outcome_reward": raw["outcome_reward"],
            "efficiency_decay": efficiency_decay,
            "avg_difficulty_weight": raw.get("avg_difficulty_weight", 0.0),
            "density_sigma": raw.get("density_sigma", self.density_sigma_floor),
        }
        return {"reward_score": score, "reward_extra_info": extra}

    # ------------------------------------------------------------------
    # run_single: required by RewardManagerBase / reward_loop.py
    # Falls back to single-sample compute_score (EMA-based rho_j).
    # ------------------------------------------------------------------

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1
        data_item = data[0]
        response_str, valid_response_length = await self.loop.run_in_executor(
            None, lambda: self._decode_response(data_item)
        )

        data_source = data_item.non_tensor_batch.get("data_source", "")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})

        if self.is_async_reward_score:
            result = await self.compute_score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score_fn(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = float(result)
            reward_extra_info["acc"] = score

        reward = score
        reward, overlong_extra = self._apply_overlong_penalty(reward, valid_response_length)
        reward_extra_info.update(overlong_extra)
        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    # ------------------------------------------------------------------
    # run_batch: group-aware scoring (paper-aligned).
    # Called by reward_loop.py if it detects this method exists.
    # ------------------------------------------------------------------

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Score all samples in data with group-level rho_j.

        Groups samples by uid (set by the trainer). Within each group,
        collects all passed_flags, computes group-level rho_j, then
        computes R^turn = sum_j w_j' * p_j for each sample.
        """
        n = len(data)

        # Step 1: execute all samples in parallel (thread pool)
        raw_results: list[dict | None] = [None] * n
        response_strs: list[str] = [""] * n
        valid_lengths: list[int] = [0] * n

        def _run(i):
            item = data[i]
            r_str, vlen = self._decode_response(item)
            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            extra = dict(item.non_tensor_batch.get("extra_info") or {})
            raw = execute_single(
                solution_str=r_str,
                ground_truth=gt,
                extra_info=extra,
                timeout_sec=self.timeout_sec,
                memory_limit_mb=self.memory_limit_mb,
            )
            return i, r_str, vlen, raw

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futs = [loop.run_in_executor(pool, _run, i) for i in range(n)]
            for coro in asyncio.as_completed(futs):
                i, r_str, vlen, raw = await coro
                raw_results[i] = raw
                response_strs[i] = r_str
                valid_lengths[i] = vlen

        # Step 2: group by uid
        uids = list(data.non_tensor_batch.get("uid", [str(i) for i in range(n)]))
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)

        # Step 3: compute group-level dense reward, then build output
        outputs: list[dict] = [{"reward_score": 0.0, "reward_extra_info": {}}] * n

        for uid, indices in uid_to_indices.items():
            valid_indices = [i for i in indices if raw_results[i] is not None]
            group_flags = [raw_results[i]["passed_flags"] for i in valid_indices]

            # Vectorize group-level rho / weights once per group
            if group_flags:
                import numpy as _np
                group_arr = _np.array(group_flags, dtype=_np.float64)   # [N, M]
                rho = group_arr.mean(axis=0)                             # [M]
                base_w = _np.exp(-self.difficulty_alpha * rho)           # [M]
                sigma_val = float(max(rho.std() / 2.0, self.density_sigma_floor))
                diff = rho[:, None] - rho[None, :]                       # [M, M]
                density = _np.exp(
                    -(diff ** 2) / (2.0 * sigma_val * sigma_val + self.density_eps)
                ).sum(axis=1)
                norm_w = base_w / (density + self.density_eps)           # [M]
                avg_w = float(base_w.mean())
            else:
                norm_w = avg_w = sigma_val = None

            for i in indices:
                raw = raw_results[i]
                if raw is None:
                    outputs[i] = {"reward_score": 0.0, "reward_extra_info": {}}
                    continue

                if norm_w is not None:
                    q = _np.asarray(raw["passed_flags"], dtype=_np.float64)
                    dense_reward = float(_np.dot(norm_w, q))
                else:
                    dense_reward, avg_w, sigma_val = 0.0, 0.0, self.density_sigma_floor

                raw["dense_reward"] = dense_reward
                raw["avg_difficulty_weight"] = avg_w
                raw["density_sigma"] = sigma_val

                scored = self._score_raw(raw, response_strs[i])
                reward = scored["reward_score"]
                reward, overlong_extra = self._apply_overlong_penalty(reward, valid_lengths[i])
                scored["reward_score"] = reward
                scored["reward_extra_info"].update(overlong_extra)
                outputs[i] = scored

        return outputs

import os

from verl.utils.fs import resolve_peft_adapter_path


def test_resolve_peft_adapter_path_accepts_adapter_dir(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert resolve_peft_adapter_path(os.fspath(adapter_dir)) == os.fspath(adapter_dir)


def test_resolve_peft_adapter_path_accepts_actor_checkpoint_root(tmp_path):
    actor_dir = tmp_path / "global_step_1" / "actor"
    nested_adapter_dir = actor_dir / "lora_adapter"
    nested_adapter_dir.mkdir(parents=True)
    (nested_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert resolve_peft_adapter_path(os.fspath(actor_dir)) == os.fspath(nested_adapter_dir)


def test_resolve_peft_adapter_path_returns_original_when_missing(tmp_path):
    missing_adapter_dir = tmp_path / "missing"

    assert resolve_peft_adapter_path(os.fspath(missing_adapter_dir)) == os.fspath(missing_adapter_dir)

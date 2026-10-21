"""BP-04: dump split, collator and checkpoint round-trip; a tiny end-to-end run when the target is cached."""

import json

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mini_aurora.baseline.data import DumpDataset, collate, make_loader
from mini_aurora.baseline.dump import split_rows
from mini_aurora.models.target import DEFAULT_TARGET

needs_target = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str),
    reason=f"{DEFAULT_TARGET} not in the HF cache",
)


def _item(T, aux=6, hid=2):
    return {"input_ids": torch.arange(T), "loss_mask": torch.ones(T, dtype=torch.long),
            "aux_hidden_states": torch.randn(1, T, aux), "hidden_states": torch.randn(1, T, hid)}


def test_split_rows_keeps_last_rows_out():
    rows = [{"id": i} for i in range(10)]
    s = split_rows(rows, heldout=3)
    assert [r["id"] for r in s["train"]] == list(range(7)) and [r["id"] for r in s["heldout"]] == [7, 8, 9]
    with pytest.raises(ValueError):
        split_rows(rows, heldout=10)


def test_collate_right_pads():
    features, attention_mask = collate([_item(3), _item(5)])
    assert features.input_ids.shape == (2, 5) and features.hidden_states.shape == (2, 5, 6)
    assert attention_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert features.loss_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert features.input_ids[0].tolist() == [0, 1, 2, 0, 0]
    assert torch.equal(features.hidden_states[0, 3:], torch.zeros(2, 6))


def test_dataset_and_loader(tmp_path):
    for i, T in enumerate((4, 2, 3)):
        torch.save(_item(T), tmp_path / f"c{i}.ckpt")
    assert len(DumpDataset(tmp_path)) == 3 and len(DumpDataset(tmp_path, limit=2)) == 2
    batches = list(make_loader(tmp_path, batch_size=2, shuffle=False))
    assert len(batches) == 2 and batches[0][0].input_ids.shape == (2, 4)
    with pytest.raises(FileNotFoundError):
        DumpDataset(tmp_path / "empty")


@needs_target
def test_dump_train_evaluate_smoke(tmp_path):
    """Two conversations through dump -> train (1 step) -> checkpoint -> load -> evaluate, on the local device."""
    from mini_aurora.baseline.checkpoint import latest_checkpoint, load_draft, save_draft
    from mini_aurora.baseline.dump import run_dump
    from mini_aurora.baseline.evaluate import evaluate
    from mini_aurora.baseline.train import train
    from mini_aurora.data.download import write_jsonl
    from mini_aurora.device import get_device, get_dtype
    from mini_aurora.models.draft import build_draft, target_lm_head_weight, wrap_eagle3
    from mini_aurora.models.samples import SAMPLE_CONVERSATIONS
    from mini_aurora.models.target import load_target, load_tokenizer
    from mini_aurora.models.vocab import build_vocab_mapping, save_vocab_mapping

    jsonl = tmp_path / "chat.jsonl"
    write_jsonl(jsonl, [{"id": f"chat-{i:05d}", "domain": "chat", "messages": m} for i, m in enumerate(SAMPLE_CONVERSATIONS[:4])])
    manifest = run_dump(str(jsonl), str(tmp_path / "dump"), heldout=1, limit=None, dtype=torch.bfloat16)
    assert manifest["splits"]["train"]["conversations"] == 3 and manifest["splits"]["heldout"]["conversations"] == 1
    item = torch.load(next((tmp_path / "dump" / "train").glob("*.ckpt")), weights_only=False)
    assert item["aux_hidden_states"].shape[-1] == 3 * item["hidden_states"].shape[-1]
    assert item["loss_mask"][-1] == 0

    target, tokenizer = load_target(), load_tokenizer()
    d2t, t2d = build_vocab_mapping(tokenizer, SAMPLE_CONVERSATIONS, target.model.config.vocab_size)
    vocab = save_vocab_mapping(tmp_path / "vocab.pt", d2t, t2d)
    result = train(str(tmp_path / "dump"), str(tmp_path / "ckpt"), str(vocab), epochs=1, batch_size=1, accumulation=1, log_every=0)
    assert result["config"]["total_steps"] == 3
    assert all(torch.isfinite(torch.tensor(result["epochs"][0]["heldout"]["loss"])))
    assert json.loads((tmp_path / "ckpt" / "eval.json").read_text())["epochs"][0]["step"] == 3

    device = get_device()
    path = latest_checkpoint(tmp_path / "ckpt")
    draft, ckpt = load_draft(target, path, device=device, dtype=get_dtype(device))
    assert ckpt["step"] == 3 and torch.equal(draft.t2d.cpu(), t2d)
    # Round-trip: saving the loaded draft reproduces the same weights.
    path2 = save_draft(draft, tmp_path / "ckpt2", 3)
    a, b = torch.load(path, weights_only=False)["state_dict"], torch.load(path2, weights_only=False)["state_dict"]
    assert a.keys() == b.keys() and all(torch.equal(a[k].float(), b[k].float()) for k in a)
    loader = make_loader(tmp_path / "dump" / "heldout", 1, shuffle=False, feature_dtype=get_dtype(device))
    out = evaluate(wrap_eagle3(draft), target_lm_head_weight(target), loader, device)
    assert out["assistant_tokens"] > 0 and len(out["acc"]) == 3

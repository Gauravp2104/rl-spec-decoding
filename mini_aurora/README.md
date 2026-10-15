# mini_aurora

Scaled-down Aurora reproduction (base paper only, no GRPO) that runs on one Apple Silicon Mac.
Plan: [`docs/base_paper/PLAN.md`](../docs/base_paper/PLAN.md). Conventions: [`docs/base_paper/CONVENTIONS.md`](../docs/base_paper/CONVENTIONS.md).

## [BP-01] Mac environment + device layer

Tracks #4. Branch: `bp/01-env-mac`. Status: **done**.

### Setup (once, from the repo root)

```bash
bash mini_aurora/setup_mac.sh                     # creates .venv (Python 3.12) and installs everything
source .venv/bin/activate && source mini_aurora/env.sh
```

`setup_mac.sh` does:
1. `uv venv --python 3.12 .venv`
2. `uv pip install -r mini_aurora/requirements-mac.txt`: torch (MPS), `transformers==4.57.1` (Aurora's pin), datasets, accelerate, omegaconf, pydantic, numba, wandb (offline), matplotlib, pytest.
3. `uv pip install -e aurora-main --no-deps`: Aurora's model/loss/data code **without** Mooncake, Ray, sglang-router and flash-attn (CUDA/cluster-only).
4. `uv pip install -e . --no-deps`: this package (root `pyproject.toml`).

### Environment (`mini_aurora/env.sh`, also applied by `import mini_aurora`)

| Variable | Value | Why |
|---|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Ops missing on MPS run on CPU instead of erroring |
| `TORCHDYNAMO_DISABLE` | `1` | Eager mode; `torch.compile` / FlexAttention are unreliable on MPS |
| `WANDB_MODE` | `offline` | `aurora.utils.logging` imports wandb; no network logging |
| `TOKENIZERS_PARALLELISM` | `false` | Silences fork warnings from HF tokenizers |

`import mini_aurora` sets these with `setdefault`, so a value already exported wins.

### Device layer (`mini_aurora/device.py`)

- `get_device(prefer=None)`: `prefer` or `$MINI_AURORA_DEVICE` if set, else cuda → mps → cpu. Use this everywhere instead of `.cuda()`.
- `get_dtype(device)`: bf16 on CUDA/MPS, fp32 on CPU.
- `empty_cache(device)`, `memory_allocated_gb(device)`: for the memory measurements in BP-03.

Models built on top of Aurora must use `attention_backend="sdpa"` (BP-03).

### Verification

```bash
pytest -q                                                                  # mini_aurora/tests: 3 passed
(cd aurora-main && pytest -q tests/test_parse.py tests/test_vocab_mapping.py)   # 17 passed
```

Checked on an M5 Pro (macOS 26.3.1) with torch 2.14.1, transformers 4.57.1, MPS available.

Safe to import on Mac: `aurora.models.*` (including `aurora.models.target.eagle3_target_model`) and `aurora.data`.
Do **not** import modules that import Ray or Mooncake at the top: `aurora.ray`, `aurora.controller`, `aurora.transfer.mooncake`,
`aurora.inference.engine`, `aurora.utils.misc`, `aurora.train_entry`.
The other Aurora tests (`test_mooncake_master.py`, `test_data_fetcher.py`, ...) are expected to fail on a Mac.

### Done when

- [x] `aurora.models.draft.llama3_eagle` and `aurora.models.eagle3` import on Mac
- [x] `pytest tests/test_parse.py tests/test_vocab_mapping.py` passes

## Layout

| Path | Sub-task |
|---|---|
| `device.py`, `env.sh`, `setup_mac.sh`, `requirements-mac.txt`, `tests/` | BP-01 |
| `data/` | BP-02 |
| `models/` | BP-03 |
| `baseline/` | BP-04 |
| `server/` | BP-05 |
| `online/` | BP-06 |
| `eval/` | BP-07 |
| `scale/` | BP-09 |

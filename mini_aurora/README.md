# [BP-01] Mac environment + device layer

Tracks #4. Branch: `bp/01-env-mac`.

- `uv venv --python 3.12`; `requirements-mac.txt` (torch, `transformers==4.57.1`, datasets, accelerate, omegaconf, numba, matplotlib, pytest).
- `pip install -e aurora-main --no-deps` (skips Mooncake / Ray / sglang-router).
- `mini_aurora/device.py::get_device()` → cuda → mps → cpu.
- Env: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `TORCHDYNAMO_DISABLE=1`.
- **Done when:** `aurora.models.draft.llama3_eagle` and `aurora.models.eagle3` import on Mac; `pytest tests/test_parse.py tests/test_vocab_mapping.py` passes.

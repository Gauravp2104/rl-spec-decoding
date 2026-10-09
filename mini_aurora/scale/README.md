# [BP-09] (Optional) CUDA portability / scale check

Tracks #20. Branch: `bp/09-cuda-scale-check`.

- Same `mini_aurora` code on Colab/cluster CUDA with Qwen3-1.7B/4B and FlexAttention re-enabled.
- If 3 CUDA GPUs are available, run Aurora's `examples/qwen3-4b-external-*` once to calibrate mini results against the real system.

# Source before running anything in mini_aurora:  source mini_aurora/env.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1   # unsupported MPS ops fall back to CPU instead of erroring
export TORCHDYNAMO_DISABLE=1           # eager mode; torch.compile / FlexAttention are not reliable on MPS
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

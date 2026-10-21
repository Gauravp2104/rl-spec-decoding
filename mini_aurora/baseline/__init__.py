"""BP-04: offline static EAGLE-3 baseline.

Aurora's offline path at laptop scale: run the frozen target over the chat pool once and store the EAGLE-3
features (``dump``), train the draft on that dump for a few epochs (``train``), save the checkpoint
(``checkpoint``) and score it on held-out chat conversations (``evaluate``). The checkpoint is both the
*static* baseline and the warm start for with-draft online training (BP-06).

Needs Aurora, i.e. torch >= 2.3: Apple Silicon, CUDA, or the CPU container in ``mini_aurora/docker``.
"""

DUMP_DIR = "outputs/baseline/hidden_states"
CKPT_DIR = "outputs/baseline/checkpoints"
HELDOUT = 100            # last rows of the chat pool, never trained on
DUMP_DTYPE = "bfloat16"  # storage dtype for the features (~2 bytes per value)

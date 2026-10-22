"""BP-02: public data subsets, request streams and the cached draft vocab mapping.

Pure-Python modules (``download``, ``streams``) run anywhere; ``vocab`` needs Aurora, i.e. torch >= 2.3
(Apple Silicon, CUDA, or the Linux CPU container in ``mini_aurora/docker``).
"""

DOMAINS = ("chat", "code", "math")
SEED = 0
PER_DOMAIN = 1000
MAX_TOKENS = 1024
MAX_TURNS = 6
DATA_DIR = "data"
STREAMS = ("mixed", "ordered", "domain_shift")
# Domain-shift stream: a long chat run first so the drafter settles, then two shifts.
SHIFT_SEGMENTS = (("chat", 600), ("code", 300), ("math", 300))

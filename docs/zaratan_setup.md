# Zaratan setup

Team instructions for running `mini_aurora` on UMD's Zaratan HPC cluster. Access comes through the MSAI-635 course allocation (`msai635-class`).

1. Log in (MFA is required unless you are on the UMD VPN):
   ```bash
   ssh <umd-directory-id>@login.zaratan.umd.edu
   ```
2. Check the allocation and pick a work directory. Home is 10 GB but fast; the course scratch (`~/scratch.msai635`) is big but slow for the many small files of a Python install, and it has been over its file-count quota. The runs so far used home for the environment and scratch only for large outputs:
   ```bash
   sbalance                                     # SUs left on msai635-class
   mkdir -p ~/rl-work && cd ~/rl-work
   git clone https://github.com/Gauravp2104/rl-spec-decoding.git repo && cd repo
   ```
3. Create the environment (once, on the login node; about 20 minutes). Aurora needs Python 3.12 and the cluster module is 3.10, so the script fetches a standalone CPython with `uv`, installs torch 2.6 (cu124), downloads the ShareGPT / code / math data, builds the 24K vocab and caches Qwen3-0.6B and 1.7B. Everything goes under `~/rl-work` (about 8 GB):
   ```bash
   SCRATCH=$HOME/rl-work bash mini_aurora/zaratan/setup_env.sh
   ```
4. Run the test suite on an A100 MIG slice (7 SU/hour, no queue) and check the log:
   ```bash
   sbatch mini_aurora/zaratan/tests.sbatch
   squeue -u $USER
   cat slurm-bp09-tests-<jobid>.out       # expect "68 passed"
   ```
5. Submit jobs. Each script in `mini_aurora/zaratan/` has its smoke and full commands in its header; walltime (`-t`) is the SU cap:
   ```bash
   sbatch --export=ALL,SCRATCH=$HOME/rl-work mini_aurora/zaratan/bp09_1p7b.sbatch   # example: 1.7B smoke run, ≤ 20 SU
   bash mini_aurora/zaratan/submit_bp08.sh                                           # the E1–E4 arms
   squeue -u $USER
   sacct -j <jobid> --format=JobID,Elapsed,AllocTRES%40,State -X   # SUs charged = elapsed A100 hours × 48
   ```

GPU cost: 1 A100 hour = 48 SU, 1 MIG slice hour = 7 SU, 1 H100 hour = 144 SU. The full-A100 queue is first-come for this account and has waited hours; MIG slices start at once.

# Base Paper Implementation — Branch, PR and Issue Conventions

## Branch model

```
main
 └── base-paper                      integration branch (umbrella PR → main)
      ├── bp/01-env-mac              one branch per sub-task (sub-PR → base-paper)
      ├── bp/02-data-streams
      └── ...
```

- **Never commit directly to `main` or `base-paper`.** All work lands through PRs.
- Sub-task PRs target **`base-paper`**, not `main`.
- The umbrella PR (`base-paper` → `main`) is merged last, with a **merge commit** (not squash) to keep sub-PR history.

## Feature branch naming

| Kind | Pattern | Example |
|---|---|---|
| Sub-task branch (one per issue) | `bp/<NN>-<slug>` | `bp/05-spec-server` |
| Follow-up / extra work for a sub-task | `bp/<NN>-<slug>--<topic>` | `bp/05-spec-server--kv-cache` |
| Fix on integration branch | `bp/fix-<slug>` | `bp/fix-mps-nan-loss` |
| Experiment-only branch | `bp/exp-<id>-<slug>` | `bp/exp-e2-domain-shift` |
| Later GRPO project (not this one) | `grpo/<NN>-<slug>` | `grpo/01-reward-module` |

Rules: lowercase, hyphen-separated, `<NN>` is the two-digit sub-task number, `<slug>` ≤ 4 words.
Use `--` (double hyphen) for follow-ups — Git cannot hold both `bp/05-spec-server` and
`bp/05-spec-server/kv-cache` as branches.

Follow-up branches target their parent sub-task branch while it is open, otherwise `base-paper`.

## Titles

| Item | Format | Example |
|---|---|---|
| Issue | `[BP-NN] <Title>` | `[BP-05] Mini speculative-decoding server` |
| PR | `[BP-NN] <Title>` | `[BP-05] Mini speculative-decoding server` |
| Commit | `[BP-NN] <imperative summary>` | `[BP-05] Add greedy acceptance rule` |

## Labels and project

- Every issue/PR: `base-paper`; add `mac-repro` if it must run on Apple Silicon.
- Every issue/PR is added to the **Base paper implementation** GitHub Project.
- Board status: `Todo` → `In Progress` → `Done`.
- Sub-task issues are **sub-issues** of the tracking issue `[BP-0] Base paper implementation`.

## PR workflow

1. Open as **draft** early; mark ready when the issue's *Done when* criteria pass.
2. PR body: `Closes #<issue>` and a short test/evidence section (logs, plots, memory numbers).
   (`Closes` fires when the umbrella merges into `main`; close the issue manually if you want it closed earlier.)
3. One teammate review before merging into `base-paper`.
4. After each merge into `base-paper`, rebase open sub-task branches:
   `git fetch && git rebase origin/base-paper && git push --force-with-lease`.

Merge order: **BP-01 → (BP-02, BP-03) → (BP-04, BP-05) → BP-06 → BP-07 → BP-08 → BP-09**.

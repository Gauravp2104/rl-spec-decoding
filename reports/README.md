# Reports

| Report | Source | PDF |
|---|---|---|
| Project proposal | Overleaf (not in the repo) | [`proposal/MSAI635_project_proposal_Group2.pdf`](proposal/MSAI635_project_proposal_Group2.pdf) |
| Interim report | [`interim/interim_report.tex`](interim/interim_report.tex) | [`interim/interim_report.pdf`](interim/interim_report.pdf) |

Build the interim report (needs a TeX distribution with TikZ; it also compiles as-is on Overleaf):

```bash
cd reports/interim && latexmk -pdf interim_report.tex
```

Numbers in the interim report come from `docs/base_paper/PLAN.md`, `mini_aurora/*/README.md` and
`docs/concepts/` on the `base-paper` branch (BP-01 to BP-05 merged).

# Monograph index

> **Language note.** The pre-projeto was submitted in PT-BR (`\usepackage[brazil]{babel}`).
> CIn/UFPE accepts TGs in either Portuguese or English. This scaffold is written in
> English to match the working materials (`thesis/phase*.md`, all reports). If the
> final submission must be in PT-BR, translate per chapter — content/structure don't change.

## Chapter map

| # | File | Status | Primary source materials | Approx. pages |
| --- | --- | --- | --- | ---: |
| Front matter | `09_abstract.md` | scaffold | — | 2 |
| 1 | `01_introduction.md` | scaffold | pre-projeto §1, `tema1.txt`, advisor feedback | 4–6 |
| 2 | `02_background.md` | scaffold | proposal §2–3, Gil Tene refs, queueing texts | 8–12 |
| 3 | `03_related_work.md` | scaffold | `referencias.bib`, claim-check research | 4–6 |
| 4 | `04_methodology.md` | scaffold | `PLAN.md`, `phase3.md`, `phase3_cpu_mechanism_plan.md` | 6–10 |
| 5 | `05_implementation.md` | scaffold | `phase1.md`, `phase2.md`, code | 8–12 |
| 6 | `06_results.md` | scaffold | all 4 reports + CSVs + figures | 10–14 |
| 7 | `07_discussion.md` | scaffold | mechanism finding, caveats, lit verification | 4–6 |
| 8 | `08_conclusion.md` | scaffold | — | 2–3 |
| Back matter | `references.md` | placeholder | `referencias.bib` | — |

Total target: ~50–80 pages depending on figure density.

## How to write

Each scaffold uses these markers:
- `<<TODO: ...>>` — prose to write, with a one-line prompt
- `<<SOURCE: file>>` — where the raw material already exists; copy + edit prose
- `<<FIG: results/figures/X.png>>` — figure to include (already on disk)
- `<<CITE: bibkey>>` — placeholder for a reference (key in `pre_projeto/referencias.bib`)

The scaffolds contain the **outline + the prose hooks** — the contribution is structural, not content. Fill in the `<<TODO>>` blocks with prose; the rest (figures, tables, references) is already pointed at the right artifact.

## Data and figures (already on disk)

| Asset | Path |
| --- | --- |
| 219-cell sweep | `project/3-bench/results/summary.csv` |
| Bootstrap CIs | `project/3-bench/results/ci.csv` |
| Effect sizes | `project/3-bench/results/effect_sizes.csv` |
| Latency budget | `project/3-bench/results/budget.csv` |
| CPU mechanism (after 2 am) | `project/3-bench/results/cpu_mechanism.csv` |
| L1 floor | `project/3-bench/results/layer1/L1_summary.json` |
| L0 noise floor | `project/3-bench/results/layer0/L0_summary.json` |
| All figures | `project/3-bench/results/figures/*.png` |

## Outstanding data gaps (small)

1. FastAPI no-keepalive follow-up — quantifies worker-axis dormancy (~15 min bench).
2. L1 literature corroboration — research agent is running; will land in this folder.

# Live Dota 2 win probability (Vexil Capital take-home)

A per-minute probability that Radiant wins a professional Dota 2 match, using only what is known at that minute, compared with a gold-lead baseline.

## What is in this repository

| File | What it is |
|---|---|
| `vexil_win_probability.ipynb` | The whole project in one notebook, committed with all outputs: data pull, per-minute table, leakage checks, split, models, evaluation, calibration, prediction table, comments |
| `REPORT.md` | The report (4 pages) |
| `figures/calibration_by_phase.png` | The figure used in the report |
| `predictions_heldout.csv` | Held-out predictions: `match_id, minute, p_model, p_baseline, radiant_win` (the notebook rewrites it) |
| `requirements.txt` | Pinned package versions |

## Run it on a laptop

Tested with Python 3.13 on Linux.

```bash
git clone https://github.com/kientt247-collab/vexilcap.git
cd vexilcap
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then either open the notebook and choose "Run all":

```bash
pip install jupyterlab             # only if you want the Jupyter interface
jupyter lab vexil_win_probability.ipynb
```

or run it headless, which writes a new executed copy:

```bash
jupyter nbconvert --to notebook --execute vexil_win_probability.ipynb \
    --output executed.ipynb --ExecutePreprocessor.timeout=3600
```

## Run it on Google Colab

Open `vexil_win_probability.ipynb` in Colab (File > Open notebook > GitHub, or upload it) and choose Runtime > Run all. The first cell installs the same pinned versions when it detects Colab.

## What to expect

- No API key and no local data file are needed. The notebook pulls 500 matches from the two OpenDota endpoints of the brief (`/proMatches`, `/matches/{match_id}`).
- Runtime: about 13 to 16 minutes from scratch. The pull makes about 512 calls at 55 per minute (about 9.5 minutes), plus waiting when OpenDota answers 429. The rest takes about 3 minutes.
- API responses are cached in `./cache`. A second run in the same folder makes no HTTP call and takes about 3 minutes.
- The same numbers on every run:
  - the 500 match ids are pinned by a SHA-256, and the notebook stops if the pull finds a different list;
  - all random steps use fixed seeds;
  - the split is checked against fixed hashes;
  - the A1 model is checked against its written specification;
  - the last cell checks every number quoted in the comments.
- Files written by a run:
  - `cache/`: the API responses;
  - `matches_index.csv`: the 500 matches with match_id, start_time and patch;
  - `features_per_minute.csv`: the per-minute table;
  - `predictions_heldout.csv`: the held-out predictions.

## Time spent

About 28 hours, including AI-assisted work (also stated in the notebook's setup section).

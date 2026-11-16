# More than a Hack 2027, Phase 1: Ball Possession

Team FC Data-lona's entry for the Initial Challenge. It labels every frame of the Barcelona vs
Athletic Club tactical video as H, A, D or X.

All the training and tuning happens on Metrica Sports' public sample games
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)). The competition
match is only used as input to models that are already trained. We didn't hand label anything,
didn't use the video, and didn't train or tune on the ground truth.

The write-up is in [TECHNICAL.md](TECHNICAL.md).

## Getting the submission file

You need Python 3.12 or 3.13.

```bash
pip install -r requirements.txt
python run.py --data /path/to/bundle --team "FC Data-lona"
```

This writes `outputs/FC Data-lona.csv` and a short log next to it.

`--data` is the folder from the organisers that has `tactical/` in it. We only read three files
from it: `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and
`BAR-ATH_tactical_smart_tagging.xml`.

The trained model is already in `models/`, so there's no training step. It runs in about a
minute, and you get the same file every time.

On Windows you can also run `RUN_ME_WINDOWS.bat "FC Data-lona" "C:\path\to\bundle"`. It sets up
a virtual environment, runs everything and checks the file with the organisers' validator.

## Retraining

If you want to rebuild the model yourself:

```bash
git clone https://github.com/metrica-sports/sample-data.git
pip install -r requirements-train.txt
python train.py --sample-data ./sample-data
```

## What's where

- `run.py` runs the trained model on the competition match and writes the CSV
- `train.py` trains the classifiers and picks the decoder settings on the sample games
- `possession/game.py` loads the data and turns sample-game events into frame labels
- `possession/degrade.py` adds noise to the sample games so they look like automatic tracking
- `possession/features.py` builds the per-frame features
- `possession/models.py` has the two classifiers
- `possession/decode.py` has the Viterbi decoder
- `possession/official_metrics.py` is the organisers' scoring code, copied as is (MIT licence)
- `models/` holds the trained model, the decoder settings and the training log

## Data

The match data is only licensed for the competition, so it's not in this repo, and
`.gitignore` keeps it (and any generated CSVs) out. Sample game data is from Metrica Sports.

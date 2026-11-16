# More than a Hack 2027 — Phase 1: Ball Possession

Labels every frame of the tactical video of FC Barcelona vs Athletic Club as `H`, `A`, `D` or `X`.

**Compliance in one line:** every model and every tuned parameter comes from **external data
only**, namely Metrica Sports' open sample games
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)). The competition
game is only ever *read as input* by fixed models. No hand labels, no human-produced data about
the game, no use of the video or the ground truth for training, fitting or calibration, and no
upload of the provided data to any AI service.

Full write-up: [TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission

Python 3.12 or 3.13.

```bash
pip install -r requirements.txt
python run.py --data /path/to/unpacked/bundle --team <TeamName>
# -> outputs/<TeamName>.csv  and  outputs/<TeamName>_run_log.txt
```

`--data` is the folder that contains `tactical/`. Only these files are read:
`BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_smart_tagging.xml`.

`run.py` loads the committed `models/model.joblib`, so it needs no training step. It takes
about 1–2 minutes and is deterministic: the same data and the same pinned versions produce the
same file.

On Windows: `RUN_ME_WINDOWS.bat <TeamName> "C:\path\to\bundle"` sets up a virtualenv, runs the
pipeline, then validates the file with the organisers' `validate_submission.py`.

## Retrain from scratch (external data only)

```bash
git clone https://github.com/metrica-sports/sample-data.git
pip install -r requirements-train.txt
python train.py --sample-data ./sample-data
# -> models/model.joblib, models/params.json, models/training_report.txt
```

## Layout

| path | what |
|---|---|
| `run.py` | apply the fixed models to the competition game, write the CSV |
| `train.py` | train classifiers and tune the decoder on the external sample games |
| `possession/game.py` | loaders (competition ATD, Metrica sample games), event → label derivation |
| `possession/degrade.py` | makes clean external tracking look like automatic tracking |
| `possession/features.py` | ball cleaning, distances, team shape, dead-ball cues |
| `possession/models.py` | the two gradient-boosted frame classifiers |
| `possession/decode.py` | three-state Viterbi decoder |
| `possession/official_metrics.py` | the organisers' scorer, vendored unchanged (MIT), used in tuning |
| `models/` | trained models, tuned decoder parameters, training report |

## Data

The match data is licensed for the competition and is **not** in this repository. `.gitignore`
excludes it, along with every generated CSV. Metrica sample data © Metrica Sports, used with
attribution as their repository requests.

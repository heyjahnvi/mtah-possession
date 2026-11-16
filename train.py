"""Trains and tunes everything on the Metrica Sports sample games. Nothing from the
competition match is used here.

    git clone https://github.com/metrica-sports/sample-data.git
    pip install -r requirements-train.txt
    python train.py --sample-data ./sample-data

What it does:
  1. loads the three sample games and turns their event data into H/A/D labels per frame
  2. adds noise so they look like automatic tracking (possession/degrade.py), with 2 seeds
  3. builds features, plus a copy with home and away swapped
  4. trains on two games and predicts the third, for each of the three games
  5. picks the decoder settings that give the best spellF1 on those held-out predictions
     (using the organisers' own scoring code)
  6. retrains on all three games and saves models/model.joblib
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from possession import features as F
from possession import models as M
from possession import decode as V
from possession import official_metrics as pm
from possession.degrade import degrade
from possession.game import load_sample_game

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1)
GAMES = (1, 2, 3)


def swap_labels(lab):
    out = lab.copy()
    out[lab == "H"] = "A"
    out[lab == "A"] = "H"
    return out


def score(labels_gt, labels_pred):
    gt = pm.label_codes(labels_gt)
    pr = pm.label_codes(labels_pred)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pr, runs)
    sp = pm.spell_scores(gt, filled, runs)
    mf = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pr)))
    return sp["mean_f1"], mf, sp["n_gt"], sp["n_pred"]


def baseline(game, lab, window=50):
    """Organisers' baseline (nearest player + majority vote), just to compare against."""
    hm = game.half_mask()
    ball, _ = F.clean_ball(game.ball, hm)
    d = np.hypot(game.players[:, :, 0] - ball[:, None, 0], game.players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    near = game.team[np.argmin(d, axis=1)]
    code = np.where(np.isfinite(d.min(1)), near, -1)
    h = np.concatenate([[0], np.cumsum(code == 0)])
    a = np.concatenate([[0], np.cumsum(code == 1)])
    i = np.arange(len(code))
    lo, hi = np.maximum(0, i - window), np.minimum(len(code), i + window + 1)
    hh, aa = h[hi] - h[lo], a[hi] - a[lo]
    out = np.where(hh + aa == 0, "D", np.where(hh >= aa, "H", "A"))
    out = np.where(hm > 0, out, "X")
    return score(lab, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample-data", default=os.path.join(HERE, "sample-data"))
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = []

    def log(msg):
        print(msg, flush=True)
        report.append(msg)

    t0 = time.time()
    data = {}
    for g in GAMES:
        game, lab = load_sample_game(args.sample_data, g)
        hm = game.half_mask()
        lab = np.where(hm > 0, lab, "X")
        s = lab[hm > 0]
        log("{}: {} frames, D share {:.3f}, {} label segments".format(
            game.name, game.n, np.mean(s == "D"), int(np.sum(s[1:] != s[:-1]))))
        samples, evalset = [], None
        for seed in SEEDS:
            dg = degrade(game, seed=100 * g + seed)
            for swap in (False, True):
                Xp, pn, Xd, dn = F.build(dg, swap=swap)
                samples.append((Xp, Xd, swap_labels(lab) if swap else lab, hm))
                if seed == SEEDS[0] and not swap:
                    evalset = (Xp, Xd, lab, hm, dg)
        data[g] = (samples, evalset)
        log("  features built ({:.0f}s)".format(time.time() - t0))

    # --- leave-one-game-out predictions -------------------------------------------------------
    oof = {}
    for g in GAMES:
        train = [s for h in GAMES if h != g for s in data[h][0]]
        mods = M.fit(train, seed=0)
        Xp, Xd, lab, hm, dg = data[g][1]
        oof[g] = M.predict(mods, Xp, Xd) + (lab, dg)
        ph, pd_ = oof[g][0], oof[g][1]
        alive = (hm > 0) & (lab != "D") & (lab != "X")
        graded = (hm > 0) & (lab != "X")
        log("  held-out {}: frame accuracy team {:.3f} (alive frames), dead {:.3f}".format(
            dg.name, np.mean((ph[alive] > 0.5) == (lab[alive] == "H")),
            np.mean((pd_[graded] > 0.5) == (lab[graded] == "D"))))

    def evaluate(params):
        res = []
        for g in GAMES:
            ph, pd_, lab, dg = oof[g]
            pred = V.decode(ph, pd_, dg.halves, dg.n, params)
            res.append(score(lab, pred))
        return res

    # --- tune the decoder on held-out predictions (coordinate descent over a grid) -------------
    grid = {
        "p_switch": [2, 4, 6, 9, 13, 18, 25, 35],
        "p_dead": [2, 4, 6, 9, 13, 18, 25, 35],
        "ha_scale": [0.25, 0.5, 1.0, 2.0],
        "d_scale": [0.25, 0.5, 1.0, 2.0],
        "d_bias": [-1.5, -0.75, -0.25, 0.0, 0.25, 0.75, 1.5],
    }
    params = {"p_switch": 9, "p_dead": 9, "ha_scale": 1.0, "d_scale": 1.0, "d_bias": 0.0}
    cache = {}

    def objective(p):
        key = tuple(sorted(p.items()))
        if key not in cache:
            cache[key] = float(np.mean([r[0] for r in evaluate(p)]))
        return cache[key]

    best = objective(params)
    log("tuning decoder; start spellF1 {:.4f}".format(best))
    for rnd in range(3):
        improved = False
        for k, values in grid.items():
            for v in values:
                trial = dict(params, **{k: v})
                s = objective(trial)
                if s > best + 1e-6:
                    best, params, improved = s, trial, True
        log("  round {}: spellF1 {:.4f} with {}".format(rnd + 1, best, params))
        if not improved:
            break

    log("held-out results with tuned decoder (degraded external games):")
    for g, r in zip(GAMES, evaluate(params)):
        b = baseline(oof[g][3], oof[g][2])
        log("  Sample_Game_{}: spellF1 {:.4f} macroF1 {:.4f} (possessions true {} / predicted {})"
            "   | organisers' baseline here: spellF1 {:.4f} macroF1 {:.4f}".format(g, r[0], r[1], r[2], r[3], b[0], b[1]))

    # --- final models on all external games -------------------------------------------------------
    import joblib
    import sklearn
    allsamples = [s for g in GAMES for s in data[g][0]]
    mods = M.fit(allsamples, seed=0)
    joblib.dump({"models": mods, "params": params, "pos_features": pn, "dead_features": dn,
                 "sklearn": sklearn.__version__}, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "params.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    log("saved models/model.joblib ({:.0f}s total)".format(time.time() - t0))
    with open(os.path.join(args.out, "training_report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

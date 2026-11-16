"""Builds the submission file from the match data.

    pip install -r requirements.txt
    python run.py --data C:/path/to/bundle --team "FC Data-lona"

--data is the folder from the organisers that has tactical/ inside it.
The models and decoder settings in models/ come from train.py, which only uses the Metrica
sample games. Nothing here is fitted to the competition match, it just applies what was
trained. Same data in, same file out.
"""

import argparse
import os
import time

import numpy as np

from possession import features as F
from possession import models as M
from possession import decode as V
from possession.game import load_competition_game

HERE = os.path.dirname(os.path.abspath(__file__))

# Two small rules that use Smart Tagging. We picked these values by hand and did not tune
# them, because the sample games have no Smart Tagging to tune them on.
SETPIECE_DEAD_PRIOR = 1.0   # push towards "dead" in the 2 s before a tagged set piece
SETPIECE_LEAD = 50
PRE_KICKOFF_DEAD = 50.0     # before the kickoff the ball is dead, so force it


def kickoffs(game, ball_speed):
    """For each half, the first frame inside the 'Kick Off (Start)' tag where the ball moves
    faster than 3 m/s."""
    out = {}
    for t in game.tags:
        if t["code"] == "SET PIECES" and t["type"] == "kick off (start)":
            for h, a, b in game.halves:
                if a - 60 * 25 <= t["start"] <= b and h not in out:
                    k = max(t["start"], a)
                    win = np.arange(k, min(max(t["end"], k) + 15 * 25, b))
                    mv = win[np.nan_to_num(ball_speed[win], nan=0.0) > 3.0]
                    out[h] = int(mv[0]) if len(mv) else k
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(HERE, "data"))
    ap.add_argument("--team", default="submission")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--model", default=os.path.join(HERE, "models", "model.joblib"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.data, "tactical")):
        raise SystemExit("Can't find a tactical/ folder in {}. Point --data at the folder you downloaded.".format(args.data))

    import joblib
    os.makedirs(args.out, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    bundle = joblib.load(args.model)
    params = bundle["params"]
    log("decoder parameters (tuned on external games): {}".format(params))

    game = load_competition_game(args.data, args.cache or None)
    hm = game.half_mask()
    graded = hm > 0
    log("teams: H = {}, A = {}; {} track slots; {} smart-tagging instances ({:.0f}s)".format(
        game.team_names[0], game.team_names[1], len(game.team), len(game.tags), time.time() - t0))

    Xp, pn, Xd, dn = F.build(game)
    assert pn == bundle["pos_features"] and dn == bundle["dead_features"], "feature mismatch with model"
    p_home, p_dead = M.predict(bundle["models"], Xp, Xd)

    ball, bspeed = F.clean_ball(game.ball, hm)
    extra = np.zeros(game.n)
    for t in game.tags:
        if t["code"] == "SET PIECES":
            extra[max(t["start"] - SETPIECE_LEAD, 0):max(t["start"], 0)] += SETPIECE_DEAD_PRIOR
    for h, k in kickoffs(game, bspeed).items():
        a = [x for x in game.halves if x[0] == h][0][1]
        extra[a:k] += PRE_KICKOFF_DEAD
        log("half {}: kickoff detected at video frame {} ({} graded frames before it set dead)".format(h, k, k - a))

    labels = V.decode(p_home, p_dead, game.halves, game.n, params, extra_dead=extra)

    s = labels[graded]
    n_poss = 0
    for _, a, b in game.halves:
        seg = labels[a:b + 1]
        alive = seg != "D"
        n_poss += int(np.sum(alive & np.concatenate([[True], seg[1:] != seg[:-1]])))
    log("summary (graded frames): H {:.3f}  A {:.3f}  D {:.3f}; {} possessions; "
        "ball seen on {:.1f}% after bridging".format(np.mean(s == "H"), np.mean(s == "A"), np.mean(s == "D"),
                                                    n_poss, 100 * np.mean(~np.isnan(ball[graded, 0]))))
    path = os.path.join(args.out, args.team + ".csv")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    log("wrote {} ({:.0f}s)".format(path, time.time() - t0))
    with open(os.path.join(args.out, args.team + "_run_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()

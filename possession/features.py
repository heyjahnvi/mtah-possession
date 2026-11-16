"""Per-frame features. Same code for the sample games and the competition match, and nothing
in here is learned.

We build two matrices per game:
  possession features: used to decide which team has the ball (oriented to the home side)
  dead features: used to decide if the ball is in play
"""

import warnings

import numpy as np

FPS = 25
BALL_GAP_INTERP = 12     # bridge ball gaps up to 0.5 s
BALL_SPEED_CAP = 40.0    # m/s, teleport filter
OOB_MARGIN = 0.75        # metres beyond a line
PLAYER_SPEED_CAP = 12.0  # m/s; faster is an identity switch in the tracker


def rolling_mean(x, w):
    """Centred NaN-aware moving average over +/- w frames along axis 0."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    xs = np.where(valid, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    cs = np.concatenate([pad, np.cumsum(xs, axis=0)])
    cn = np.concatenate([pad, np.cumsum(valid, axis=0)])
    n = len(x)
    lo = np.clip(np.arange(n) - w, 0, n)
    hi = np.clip(np.arange(n) + w + 1, 0, n)
    c = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)


def _fill_dir(values, valid, forward=True):
    """Carry the last (or next) valid value across gaps; also return the age in frames."""
    n = len(values)
    idx = np.where(valid, np.arange(n), -1 if forward else n)
    if forward:
        idx = np.maximum.accumulate(idx)
        ok = idx >= 0
    else:
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        ok = idx < n
    out = np.full(n, np.nan)
    age = np.full(n, np.nan)
    out[ok] = values[idx[ok]]
    age[ok] = np.abs(np.arange(n)[ok] - idx[ok])
    return out, age


def clean_ball(ball, halves_mask):
    xy = ball.astype(np.float64).copy()
    seen = np.where(~np.isnan(xy[:, 0]))[0]
    if len(seen) > 1:
        d = np.hypot(np.diff(xy[seen, 0]), np.diff(xy[seen, 1]))
        bad = np.where(d / (np.diff(seen) / FPS) > BALL_SPEED_CAP)[0] + 1
        xy[seen[bad]] = np.nan
    seen = np.where(~np.isnan(xy[:, 0]))[0]
    for a, b in zip(seen[:-1], seen[1:]):
        gap = b - a - 1
        if 0 < gap <= BALL_GAP_INTERP and halves_mask[a] and halves_mask[a] == halves_mask[b]:
            t = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            xy[a + 1:b] = xy[a] + t * (xy[b] - xy[a])
    vel = np.full_like(xy, np.nan)
    vel[2:-2] = (xy[4:] - xy[:-4]) / (4.0 / FPS)
    return xy, np.hypot(vel[:, 0], vel[:, 1])


def player_speed(players):
    v = np.full(players.shape[:2], np.nan, dtype=np.float32)
    d = players[5:] - players[:-5]
    v[5:] = np.hypot(d[..., 0], d[..., 1]) / (5.0 / FPS)
    v[v > PLAYER_SPEED_CAP] = np.nan
    return v


def build(game, swap=False):
    """Returns (X_pos, pos_names, X_dead, dead_names).
    swap=True swaps home and away. We use it in training to double the data and keep the
    possession model symmetric."""
    L, W = game.pitch
    team = (1 - game.team) if swap else game.team
    hm = game.half_mask()
    n = game.n
    P = game.players.astype(np.float64)
    ball, bspeed = clean_ball(game.ball, hm)
    seen = ~np.isnan(ball[:, 0])

    # attack direction of 'home' per half, from mean positions
    direction = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for h, a, b in game.halves:
            hx = np.nanmean(P[a:b + 1][:, team == 0, 0])
            direction[a:b + 1] = 1.0 if hx < L / 2 else -1.0

        x = (P[:, :, 0] - L / 2) * direction[:, None]
        y = P[:, :, 1]
        bx = (ball[:, 0] - L / 2) * direction
        by = ball[:, 1]
        spd = player_speed(game.players).astype(np.float64)
        vx = np.full_like(x, np.nan)
        vx[5:] = (x[5:] - x[:-5]) / (5.0 / FPS)
        vx[np.abs(vx) > PLAYER_SPEED_CAP] = np.nan

        dist = np.hypot(P[:, :, 0] - ball[:, None, 0], P[:, :, 1] - ball[:, None, 1])
        f = {}
        for t, nm in ((0, "H"), (1, "A")):
            s = team == t
            dt = np.where(np.isnan(dist[:, s]), np.inf, dist[:, s])
            dmin = dt.min(axis=1)
            f["d" + nm] = np.where(np.isfinite(dmin), dmin, np.nan)
            f["n3" + nm] = np.sum(dt < 3.0, axis=1).astype(float)
            f["n8" + nm] = np.sum(dt < 8.0, axis=1).astype(float)
            xt, yt = x[:, s], y[:, s]
            f["cx" + nm] = np.nanmean(xt, 1)
            f["sx" + nm] = np.nanstd(xt, 1)
            f["sy" + nm] = np.nanstd(yt, 1)
            f["xmax" + nm] = np.nanmax(xt, 1)
            f["xmin" + nm] = np.nanmin(xt, 1)
            f["spd" + nm] = np.nanmean(spd[:, s], 1)
            f["vx" + nm] = np.nanmean(vx[:, s], 1)
            f["cnt" + nm] = np.sum(~np.isnan(xt), 1).astype(float)
        med_spd = np.nanmedian(spd, 1)
        max_spd = np.nanmax(spd, 1)
        n_tracked = np.sum(~np.isnan(P[:, :, 0]), 1).astype(float)

    margin = f["dA"] - f["dH"]
    ff_m, ff_age = _fill_dir(margin, ~np.isnan(margin), True)
    bf_m, bf_age = _fill_dir(margin, ~np.isnan(margin), False)
    oob = seen & ((ball[:, 0] < -OOB_MARGIN) | (ball[:, 0] > L + OOB_MARGIN) |
                  (ball[:, 1] < -OOB_MARGIN) | (ball[:, 1] > W + OOB_MARGIN))
    _, since_seen = _fill_dir(np.zeros(n), seen, True)
    _, until_seen = _fill_dir(np.zeros(n), seen, False)
    still = (np.nan_to_num(bspeed, nan=99) < 0.5).astype(float)
    dmin = np.fmin(f["dH"], f["dA"])
    edge = np.fmin(np.fmin(ball[:, 0], L - ball[:, 0]), np.fmin(ball[:, 1], W - ball[:, 1]))

    shape = {}
    for k in ("cx", "sx", "sy", "xmax", "xmin", "spd", "vx", "cnt"):
        shape[k + "H"] = rolling_mean(f[k + "H"], 25)
        shape[k + "A"] = rolling_mean(f[k + "A"], 25)
        shape["d_" + k] = shape[k + "H"] - shape[k + "A"]

    pos = {
        "margin": margin, "dH": f["dH"], "dA": f["dA"],
        "n3H": np.where(seen, f["n3H"], np.nan), "n3A": np.where(seen, f["n3A"], np.nan),
        "n8_diff": np.where(seen, f["n8H"] - f["n8A"], np.nan),
        "ball_speed": bspeed, "ball_x": bx,
        "margin_r6": rolling_mean(margin, 6), "margin_r25": rolling_mean(margin, 25),
        "ff_margin": ff_m, "ff_age": ff_age, "bf_margin": bf_m, "bf_age": bf_age,
    }
    pos.update(shape)
    dead = {
        "oob": oob.astype(float), "oob_r25": rolling_mean(oob.astype(float), 25),
        "oob_r75": rolling_mean(oob.astype(float), 75),
        "ball_speed": bspeed, "ball_speed_r12": rolling_mean(bspeed, 12),
        "still_r25": rolling_mean(np.where(seen, still, np.nan), 25),
        "edge": edge, "ball_x_abs": np.abs(bx), "dmin": dmin,
        "since_seen": since_seen, "until_seen": until_seen,
        "med_spd": med_spd, "med_spd_r25": rolling_mean(med_spd, 25),
        "med_spd_r75": rolling_mean(med_spd, 75), "med_spd_r150": rolling_mean(med_spd, 150),
        "max_spd_r25": rolling_mean(max_spd, 25), "n_tracked": n_tracked,
        "sx_all": shape["sxH"] + shape["sxA"], "sy_all": shape["syH"] + shape["syA"],
        "spd_diff_abs": np.abs(shape["d_spd"]),
    }
    Xp = np.column_stack(list(pos.values())).astype(np.float32)
    Xd = np.column_stack(list(dead.values())).astype(np.float32)
    return Xp, list(pos), Xd, list(dead)

"""Make a clean external game look like the competition's automatic tracking.

The Metrica sample games carry corrected tracking: every player always tracked, and the ball
present on ~90% of live frames but absent on almost every dead-ball frame. A model trained on that
would learn "no ball = dead ball", which is false for automatic tracking, where the ball is missing
on 41% of graded frames for reasons that have nothing to do with the state of play (DATA.md).

So before training we:
  1. fill every ball gap inside a half by linear interpolation (the ball travels from where it
     went out to where play restarts), which removes the visibility leak;
  2. drop the ball in bursts with a two-state Markov chain, to the published 41% missing rate;
  3. drop players in bursts to the published ~20.5 of 22 tracked (about 7% missing);
  4. add small position noise.

Only figures published in the organisers' documentation are used, nothing measured on the game.
"""

import numpy as np

BALL_MISSING = 0.41         # DATA.md: ball absent on 41% of graded frames
BALL_MEAN_GAP = 40          # frames; assumed burst length (1.6 s)
PLAYER_MISSING = 0.07       # DATA.md: 20.5 of 22 objects tracked on average
PLAYER_MEAN_GAP = 50
BALL_NOISE = 0.3            # metres
PLAYER_NOISE = 0.15


def _fill_gaps(xy, halves):
    xy = xy.copy()
    for _, a, b in halves:
        seg = xy[a:b + 1]
        ok = ~np.isnan(seg[:, 0])
        if ok.sum() < 2:
            continue
        idx = np.arange(len(seg))
        for k in range(2):
            seg[:, k] = np.interp(idx, idx[ok], seg[ok, k])
        xy[a:b + 1] = seg
    return xy


def _burst_mask(n, missing, mean_gap, rng):
    p_out = missing / ((1 - missing) * mean_gap)
    p_back = 1.0 / mean_gap
    u = rng.random(n)
    m = np.zeros(n, dtype=bool)
    state = rng.random() < missing
    for i in range(n):
        if state:
            state = u[i] >= p_back
        else:
            state = u[i] < p_out
        m[i] = state
    return m


def _smooth_noise(shape, sigma, rng, w=12):
    """Low-frequency noise: tracking error drifts, it does not jump every frame."""
    from .features import rolling_mean
    z = rng.normal(0, sigma * np.sqrt(2 * w + 1), shape)
    return rolling_mean(z.reshape(shape[0], -1), w).reshape(shape).astype(np.float32)


def degrade(game, seed):
    from .game import Game
    rng = np.random.default_rng(seed)
    n = game.n
    ball = _fill_gaps(game.ball, game.halves)
    ball = ball + _smooth_noise(ball.shape, BALL_NOISE, rng)
    ball[_burst_mask(n, BALL_MISSING, BALL_MEAN_GAP, rng)] = np.nan

    players = game.players + _smooth_noise(game.players.shape, PLAYER_NOISE, rng)
    for s in range(players.shape[1]):
        players[_burst_mask(n, PLAYER_MISSING, PLAYER_MEAN_GAP, rng), s] = np.nan
    return Game(game.name + "_deg{}".format(seed), players, game.team, ball.astype(np.float32),
                game.halves, game.pitch, [], game.team_names)

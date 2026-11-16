"""Add noise to the clean sample games so they look more like the automatic tracking we get
for the competition match.

The Metrica sample games use corrected tracking. All players are always there, and the ball
is missing almost only when it is dead. If we trained on that directly, the model would learn
"no ball means dead ball", which is not true for automatic tracking (DATA.md says the ball is
missing on 41% of graded frames).

So before training we:
  1. fill the ball gaps inside each half by interpolation,
  2. remove the ball again in random bursts until 41% of frames are missing,
  3. remove players in bursts (about 20.5 of 22 tracked, also from DATA.md),
  4. add a bit of slow-moving position noise.

All the numbers come from the organisers' docs, nothing is measured on the match itself.
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
    """Smoothed noise. Tracking errors drift slowly, they don't jump every frame."""
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

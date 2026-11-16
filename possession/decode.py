"""Viterbi decoder over three states: H, A and D.

Each change of state costs a penalty, so a team only takes the ball over when the model
favours it for long enough. That is how we deal with touches, deflections and duels: a few
frames of the other team near the ball are not enough to pay for a switch.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def logit(p, clip=4.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def viterbi(e_ha, e_d, p_switch, p_dead):
    n = len(e_ha)
    sh = (0.5 * e_ha).tolist()
    sa = (-0.5 * e_ha).tolist()
    sd = e_d.tolist()
    back = np.empty((n, 3), dtype=np.int8)
    vh, va, vd = sh[0], sa[0], sd[0]
    for i in range(1, n):
        if vh >= va - p_switch:
            bh, kh = vh, 0
        else:
            bh, kh = va - p_switch, 1
        if vd - p_dead > bh:
            bh, kh = vd - p_dead, 2
        if va >= vh - p_switch:
            ba, ka = va, 1
        else:
            ba, ka = vh - p_switch, 0
        if vd - p_dead > ba:
            ba, ka = vd - p_dead, 2
        if vh >= va:
            al, kal = vh, 0
        else:
            al, kal = va, 1
        if vd >= al - p_dead:
            bd, kd = vd, 2
        else:
            bd, kd = al - p_dead, kal
        back[i, 0] = kh
        back[i, 1] = ka
        back[i, 2] = kd
        vh, va, vd = bh + sh[i], ba + sa[i], bd + sd[i]
    path = np.empty(n, dtype=np.int8)
    last = int(np.argmax([vh, va, vd]))
    path[-1] = last
    for i in range(n - 1, 0, -1):
        last = back[i, last]
        path[i - 1] = last
    return path


def decode(p_home, p_dead, halves, n, params, extra_dead=None):
    """p_home is P(home | alive) per frame, p_dead is P(dead) per frame.
    Returns one letter per frame, X outside the halves."""
    e_ha = params["ha_scale"] * logit(p_home)
    e_d = params["d_scale"] * logit(p_dead) + params["d_bias"]
    if extra_dead is not None:
        e_d = e_d + extra_dead
    out = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        out[a:b + 1] = LETTERS[viterbi(e_ha[a:b + 1], e_d[a:b + 1], params["p_switch"], params["p_dead"])]
    return out

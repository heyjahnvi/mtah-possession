"""The two frame classifiers. Trained only on external games (see train.py)."""

import numpy as np

THIN = 3  # train on every 3rd frame; neighbours are near-duplicates


def _clf(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def fit(samples, seed=0):
    """samples: list of (Xp, Xd, labels, half_mask). Returns (possession_model, dead_model)."""
    Xp_tr, yp_tr, Xd_tr, yd_tr = [], [], [], []
    for Xp, Xd, lab, hm in samples:
        g = (hm > 0) & (lab != "X")
        idx = np.where(g)[0][::THIN]
        Xd_tr.append(Xd[idx])
        yd_tr.append((lab[idx] == "D").astype(int))
        alive = idx[lab[idx] != "D"]
        Xp_tr.append(Xp[alive])
        yp_tr.append((lab[alive] == "H").astype(int))
    pm = _clf(seed).fit(np.vstack(Xp_tr), np.concatenate(yp_tr))
    dm = _clf(seed).fit(np.vstack(Xd_tr), np.concatenate(yd_tr))
    return pm, dm


def predict(models, Xp, Xd):
    pm, dm = models
    return pm.predict_proba(Xp)[:, 1], dm.predict_proba(Xd)[:, 1]

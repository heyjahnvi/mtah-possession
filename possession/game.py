"""Loading match data into arrays, for the competition match and for the sample games.

A Game holds:
    players : (N, S, 2) float32 positions in metres, NaN when a track slot is not seen
    team    : (S,) int8, 0 for home and 1 for away
    ball    : (N, 2) float32 in metres, NaN when the ball is not seen
    halves  : list of (half, first_frame, last_frame), inclusive, the graded frames
    pitch   : (length, width) in metres
    tags    : Smart Tagging instances (competition match only)
"""

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

HOME, AWAY = 0, 1
FPS = 25


@dataclass
class Game:
    name: str
    players: np.ndarray
    team: np.ndarray
    ball: np.ndarray
    halves: list
    pitch: tuple
    tags: list = field(default_factory=list)
    team_names: tuple = ("home", "away")

    @property
    def n(self):
        return len(self.ball)

    def half_mask(self):
        m = np.zeros(self.n, dtype=np.int8)
        for h, a, b in self.halves:
            m[a:b + 1] = h
        return m


# ------------------------------------------------------------------ competition game (ATD) ------
BARCA_N_FRAMES = 199319
# Graded halves, published by the organisers in DATA.md as match boundaries (not labels).
BARCA_HALVES = [(1, 11185, 90651), (2, 110808, 190197)]
BARCA_PITCH = (105.3, 68.0)


def _epts_team_of_slot(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}".format(meta_path))
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([HOME if s == teams[0][0] else AWAY for s in slots], dtype=np.int8)
    return team, (teams[0][1], teams[1][1])


def _parse_xy(group):
    if not group or group[0] == "N":
        return np.nan, np.nan
    a = group.split(",")
    try:
        return float(a[0]), float(a[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def parse_epts_raw(raw_path, n_frames, n_slots, pitch, first_frame_id=1):
    """'<frame>:<x,y[,s]>;...:<x,y[,s]>' -> players (N,S,2), ball (N,2) in metres."""
    L, W = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - first_frame_id
            if not 0 <= f < n_frames:
                continue
            for k, g in enumerate(parts[1].split(";")[:n_slots]):
                players[f, k] = _parse_xy(g)
            ball[f] = _parse_xy(parts[2])
    players[..., 0] *= L
    players[..., 1] *= W
    ball[:, 0] *= L
    ball[:, 1] *= W
    return players, ball


def load_smart_tagging(path, home_name, away_name):
    if not os.path.isfile(path):
        return []

    def norm(s):
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        t = norm(tags.get("Team"))
        team = None
        if t and t != "na":
            team = HOME if "barcelona" in t or t[:5] == norm(home_name)[:5] else AWAY
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "team": team,
            "type": (tags.get("Type") or "").strip().lower(),
        })
    return out


def load_competition_game(data_dir, cache_dir=None):
    stem = os.path.join(data_dir, "tactical", "BAR-ATH_tactical")
    team, names = _epts_team_of_slot(stem + "_atd.xml")
    cache = os.path.join(cache_dir, "bar_ath_atd.npz") if cache_dir else None
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        players, ball = z["players"], z["ball"]
    else:
        players, ball = parse_epts_raw(stem + "_atd.txt", BARCA_N_FRAMES, len(team), BARCA_PITCH)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache, players=players, ball=ball)
    tags = load_smart_tagging(stem + "_smart_tagging.xml", names[0], names[1])
    return Game("BAR-ATH", players, team, ball, list(BARCA_HALVES), BARCA_PITCH, tags, names)


# ------------------------------------------------------- external: Metrica open sample games ----
SAMPLE_PITCH = (105.0, 68.0)


def _load_sample_csv(folder, g):
    import pandas as pd
    arrays, teams = [], []
    ball = None
    period = None
    for side, t in (("Home", HOME), ("Away", AWAY)):
        df = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(g, side)),
                         skiprows=2)
        cols = list(df.columns)
        pcols = [c for c in cols if c.startswith("Player")]
        for c in pcols:
            i = cols.index(c)
            arrays.append(df.iloc[:, [i, i + 1]].to_numpy(np.float32))
            teams.append(t)
        bi = cols.index("Ball")
        ball = df.iloc[:, [bi, bi + 1]].to_numpy(np.float32)
        period = df["Period"].to_numpy()
    players = np.stack(arrays, axis=1)
    return players, np.array(teams, dtype=np.int8), ball, period


def _halves_from_period(period):
    halves = []
    for h in (1, 2):
        idx = np.where(period == h)[0]
        halves.append((h, int(idx[0]), int(idx[-1])))
    return halves


def _events_csv(folder, g):
    import pandas as pd
    e = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(g)))
    out = []
    for r in e.itertuples(index=False):
        out.append({"team": HOME if r.Team == "Home" else AWAY, "type": str(r.Type),
                    "sub": str(r.Subtype) if isinstance(r.Subtype, str) else "",
                    "start": int(r[4]) - 1, "end": int(r[6]) - 1})
    return out


def _events_json(path, home_id):
    d = json.load(open(path))["data"]
    out = []
    for e in d:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        out.append({"team": HOME if e["team"]["id"] == home_id else AWAY, "type": e["type"]["name"],
                    "sub": sub or "", "start": int(e["start"]["frame"]) - 1,
                    "end": int(e["end"]["frame"] or e["start"]["frame"]) - 1})
    return out


def labels_from_events(events, n, halves):
    """Frame labels for a sample game, built from its event data.

    We follow the competition definition as closely as we can. A team has the ball from its
    first controlled action (pass, carry, recovery, shot or restart) until the other team gets
    one. The ball is dead from when it crosses a line, a foul is given or a goal goes in, until
    the next restart. Challenges, deflections and lost balls don't change possession by
    themselves, only the opponent's recovery does."""
    lab = np.full(n, "X", dtype="<U1")
    changes = []  # (frame, priority, state)
    for e in events:
        t, s, en, sub = e["type"], e["start"], e["end"], e["sub"].upper()
        team = "H" if e["team"] == HOME else "A"
        if t in ("SET PIECE", "PASS", "CARRY", "RECOVERY"):
            changes.append((s, 1, team))
        elif t == "SHOT":
            changes.append((s, 1, team))
            if "GOAL" in sub or sub.endswith("OUT"):
                changes.append((max(en, s), 2, "D"))
        elif t == "BALL OUT":
            changes.append((max(en, s), 2, "D"))
        elif t == "FAULT RECEIVED":
            changes.append((s, 2, "D"))
    changes.sort()
    for h, a, b in halves:
        state = "D"
        cur = a
        for f, _, st in (c for c in changes if a <= c[0] <= b):
            # if a live event shows up while the ball is dead, treat it as a restart
            lab[cur:f] = state
            state, cur = st, f
        lab[cur:b + 1] = state
    return lab


def load_sample_game(sample_root, g):
    """Returns (Game, labels). sample_root is a clone of github.com/metrica-sports/sample-data."""
    folder = os.path.join(sample_root, "data", "Sample_Game_{}".format(g))
    L, W = SAMPLE_PITCH
    if g in (1, 2):
        players, team, ball, period = _load_sample_csv(folder, g)
        halves = _halves_from_period(period)
        events = _events_csv(folder, g)
    else:
        from kloppy import metrica
        ds = metrica.load_tracking_epts(meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
                                        raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"),
                                        coordinates="metrica")
        df = ds.to_df()
        home = ds.metadata.teams[0]
        cols, teams = [], []
        for tm, t in ((ds.metadata.teams[0], HOME), (ds.metadata.teams[1], AWAY)):
            for p in tm.players:
                if p.player_id + "_x" in df.columns:
                    cols.append(df[[p.player_id + "_x", p.player_id + "_y"]].to_numpy(np.float32))
                    teams.append(t)
        players = np.stack(cols, axis=1).copy()
        team = np.array(teams, dtype=np.int8)
        ball = df[["ball_x", "ball_y"]].to_numpy(np.float32).copy()
        period = df["period_id"].to_numpy()
        halves = _halves_from_period(period)
        events = _events_json(os.path.join(folder, "Sample_Game_3_events.json"), home.team_id)
    players[..., 0] *= L
    players[..., 1] *= W
    ball[:, 0] *= L
    ball[:, 1] *= W
    game = Game("Sample_Game_{}".format(g), players, team, ball, halves, SAMPLE_PITCH)
    return game, labels_from_events(events, game.n, halves)

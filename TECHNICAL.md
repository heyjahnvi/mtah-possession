# Technical document — More than a Hack 2027, Phase 1

**Team:** _<TeamName>_ · **Repository:** _<public URL>_ · **Submission file:** `<TeamName>.csv`

## 1. Approach

Possession is a property of *intervals*, and the ranking metric (spellF1) scores whole
possessions. So the system has two stages:

1. **Two frame classifiers** give, for every frame, `P(home | ball alive)` and `P(ball dead)`.
2. **A three-state Viterbi decoder** (H, A, D) turns those probabilities into a label sequence
   for each half. Every change of state has a cost, so a team only takes over the ball when its
   evidence outlasts the incumbent's for long enough to pay that cost.

The decoder is our working definition of **controlled possession**. A touch, a deflection or a
lost aerial duel gives a few frames of contrary evidence and cannot pay for a switch; sustained
control can. We chose it over smoothing, because a smoothing window trades flicker against late
transitions (the organisers' four baselines show exactly that), while switching costs keep
transitions sharp and still remove flicker.

### Compliance with the rules clarification

| rule | how we meet it |
|---|---|
| train/retrain only on external data | both classifiers are trained only on Metrica's open sample games 1–3 |
| tune parameters only on external data | all five decoder parameters are tuned on those games (§1.4) |
| no hand labels or human-produced data on the provided game | none; the external games' labels come from *their own* event data |
| no checking output against the video | the video is never opened by code or by us |
| no uploading provided data to AI services | the data never left the team's own machine |

The competition game is only *read as input* by models that are already fixed. Two structural
priors from Smart Tagging are applied with constants set by reasoning, not tuned (§1.5).

### 1.1 External training data

We used the three public **Metrica Sports sample games**, which include tracking and event data
for both teams at 25 fps. Per-frame labels are derived from their event data by an algorithm that
follows the competition definition:
- A team owns the ball from its first controlled action (pass, carry, recovery, shot, restart)
  until the opponent's.
- The ball is dead from the moment it crosses a line (`BALL OUT` end frame, or a shot going out),
  a foul is given or a goal is scored, until the next restart.
- Challenges, a lost ball and deflections do not change possession until the opponent recovers
  the ball.

The resulting labels have a dead share of 27–37% and 400–460 segments per game.

**Making the external games look like automatic tracking** (`possession/degrade.py`). The sample
tracking is corrected: every player is always tracked, and the ball is missing almost exactly
when it is dead. A model trained on that would learn "no ball = dead ball". We therefore:
1. Fill every ball gap inside a half by interpolation, which removes that leak.
2. Drop the ball again in random bursts (two-state Markov chain, mean gap 1.6 s) to the **41%
   missing rate published in DATA.md**.
3. Drop players in bursts to the published ~20.5 of 22 tracked.
4. Add low-frequency position noise (0.3 m for the ball, 0.15 m for players).

Each game is degraded with two seeds and also mirrored (home ↔ away swapped, labels swapped),
which keeps the possession model symmetric.

### 1.2 Features (identical code for external and competition games)

- **Ball:** teleport filter (> 40 m/s), linear bridging of gaps ≤ 0.5 s, and speed from a central
  difference.
- **Possession features** (38, oriented to the home side's attacking direction, which is detected
  per half from mean positions):
  - each team's nearest-player distance to the ball and their margin, smoothed at 0.25 s and 1 s;
  - players of each team within 3 m and 8 m;
  - ball speed and ball position;
  - the margin carried forward and backward across ball gaps, with its age in frames;
  - collective team shape smoothed over ±1 s: centroid, spread, most advanced and deepest player,
    mean speed, forward velocity, players tracked, and home-minus-away differences.
- **Dead-ball features** (20):
  - ball outside the lines (instantaneous and smoothed over 1 s and 3 s);
  - ball speed and the fraction of time the ball is still;
  - distance to the nearest line; nearest-player distance;
  - time since and until the ball was last seen;
  - median player speed smoothed at 1, 3 and 6 s, and maximum player speed;
  - players tracked; team spread; speed difference between the teams.

Player speeds are computed from positions, with values above 12 m/s discarded as identity
switches, so they are consistent across both sources.

### 1.3 Classifiers

There are two scikit-learn `HistGradientBoostingClassifier`s (300 iterations, learning rate 0.05,
31 leaves, at least 200 samples per leaf, L2 = 1, fixed seed), trained on every 3rd frame:
- **Possession:** home vs away, on ball-alive frames.
- **Dead ball:** dead vs alive, on all graded frames.

### 1.4 Decoder and its tuning on external data

Emissions:
- `e_HA = ha_scale · logit(P_home)`
- `e_D = d_scale · logit(P_dead) + d_bias`

The costs are `p_switch` for H↔A and `p_dead` for alive↔dead.

All five parameters are tuned by coordinate descent over a grid. The objective is the
**organisers' own spellF1 code** (vendored unchanged), averaged over **leave-one-game-out**
predictions: the classifiers that score game *k* were trained on the other two games.

Tuned values: `p_switch = 13`, `p_dead = 25`, `ha_scale = 1.0`, `d_scale = 0.5`, `d_bias = 1.5`.

**Held-out results on the degraded external games** (each game scored by models that never saw
it):

| game | our spellF1 | our macroF1 | organisers' baseline spellF1 / macroF1, same data |
|---|---|---|---|
| Sample 1 | 0.551 | 0.841 | 0.229 / 0.415 |
| Sample 2 | 0.599 | 0.827 | 0.234 / 0.444 |
| Sample 3 | 0.549 | 0.852 | 0.200 / 0.449 |

The final classifiers are then retrained on all three games and saved to `models/model.joblib`.
`models/training_report.txt` records the full run.

### 1.5 Competition game: structural priors (fixed, not tuned)

- **Before each half's kickoff the ball is dead.** The kickoff is the first frame the ball moves
  faster than 3 m/s inside the Smart Tagging `Kick Off (Start)` interval, which handles the graded
  but pre-kickoff start of the second half.
- **+1.0 log-odds of dead over the 2 s before each tagged `SET PIECES` start.** That is the
  stoppage before a restart. There is no Smart Tagging for the external games, so this constant
  is set by reasoning and not tuned anywhere.

Ungraded frames are written as `X`. Graded frames are never abstained.

## 2. Assets used and dropped

| asset | used? | notes |
|---|---|---|
| Tactical ATD and metadata | **yes** | main input; track slot → team, first team = home |
| Smart Tagging | **yes** | kickoff frames and set-piece stoppage prior only |
| Homography | no | only needed to work in video pixels |
| Tactical / broadcast video | no | never opened |
| Broadcast ATD | no | not synchronised to the tactical timeline |
| Public 5-minute ground truth | format/alignment check only | never used to train, tune or select anything |
| Metrica sample games 1–3 (external) | **yes** | all training and tuning |

## 3. Training data

| dataset | source | who labelled it | amount |
|---|---|---|---|
| Metrica Sports Sample Games 1, 2, 3 | github.com/metrica-sports/sample-data (public, use with attribution) | event data produced by Metrica Sports; frame labels derived from it by our algorithm (§1.1) | 3 matches, about 430,000 frames, × 2 degradation seeds × 2 mirrorings |

None of it comes from the provided game.

## 4. Possession rules implemented

- **Change of possession:** only when the other team's evidence pays the switch cost. In
  practice, that means several frames of it clearly controlling the ball.
- **Loose and contested balls:** weak evidence either way, so possession stays with the last
  team.
- **Touches, deflections, aerial duels without control:** too short to pay for a switch.
- **Restarts:** the dead stretch ends when the alive evidence takes over. The restarting team is
  whichever the possession model favours at that moment.
- **Dead ball:** always `D`, never assigned to a team.

## 5. Limitations and next steps

- **Domain gap.** The external games are degraded with published averages, but the real
  automatic tracking has its own error structure (identity switches, burstier ball loss in crowded
  areas). We chose not to measure that structure on the provided game, to stay well inside the
  rules. With more external automatic-tracking data, we would fit the degradation model to it.
- **Only three external games.** The decoder tuning is noisy at that sample size. More games
  (other open tracking sets with events) would tighten it.
- **Dead balls without the ball leaving the pitch** (fouls, offsides) rest on player-speed cues
  and are the most frequent miss.
- **Next steps:** a ball detector on the video to fill the 41% gaps, which is allowed as input
  provided it is not trained on this video; and referee-whistle detection from broadcast audio.

## 6. Reproduction

```
pip install -r requirements.txt
python run.py --data <bundle> --team <TeamName>      # uses committed models/
# optional, external data only:
pip install -r requirements-train.txt
python train.py --sample-data <clone of metrica-sports/sample-data>
```

### Run summary for the submitted file

_(paste outputs/<TeamName>_run_log.txt here)_

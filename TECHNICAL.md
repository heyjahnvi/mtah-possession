# FC Data-lona: Technical document

More than a Hack 2027, Phase 1 (Ball Possession)

Team: FC Data-lona

Repository: https://github.com/heyjahnvi/mtah-possession

Submission file: `FC Data-lona.csv`

## 1. How it works

The leaderboard uses spellF1, which scores whole possessions rather than single frames. So we
split the job into two parts.

First, two classifiers look at every frame on its own. One gives the probability that the home
team has the ball (when the ball is in play). The other gives the probability that the ball is
dead.

Second, a Viterbi decoder with three states (H, A and D) turns those probabilities into a clean
sequence for each half. Every switch between states has a cost. A team only takes over if the
classifier keeps favouring it for long enough to cover that cost.

That cost is also how we handle the "controlled possession" rule. A touch, a deflection or an
aerial duel only gives a few frames of the other team near the ball, which isn't enough to pay
for a switch. Real control lasts longer and does pay for it.

Smoothing is the obvious alternative, but the organisers' baselines already show the problem
with it: a short window flickers and a long window switches late. The decoder avoids both.

### Staying inside the rules

All training and all tuning is done on external data, the three Metrica Sports sample games.
The competition match only goes through models that were finished before they ever saw it.
Nobody labelled anything by hand, we never opened the video, and the ground truth file was only
used once at the end to check the CSV format and that the frames lined up. The match data
stayed on our own laptop the whole time.

There are two small rules that use Smart Tagging (section 1.5). We set their values by hand
because the sample games don't have Smart Tagging, so there was nothing to tune them on.

### 1.1 Training data

Metrica has released three sample matches with tracking and event data for both teams, at
25 fps. We turned the events into frame labels with a short script that follows the competition
definition:

- A team has the ball from its first controlled action (pass, carry, recovery, shot or restart)
  until the other team gets one.
- The ball is dead from when it crosses a line, a foul is given or a goal goes in, until play
  restarts.
- Challenges, deflections and lost balls don't change possession by themselves. Only the
  opponent's recovery does.

That gives between 27% and 37% dead time and roughly 400 to 460 segments per match.

There was one catch. In the sample games the ball goes missing almost exactly when it's dead,
because that tracking has been cleaned up by hand. A model trained on that would just learn
"no ball means dead ball", and that's not true in the automatic tracking for the competition
match. So before training we:

1. filled in the ball gaps inside each half by interpolation,
2. removed the ball again in random bursts until 41% of frames were missing (the figure from
   DATA.md),
3. removed players in bursts so about 20.5 of 22 are tracked, also from DATA.md,
4. added a bit of slow-drifting noise to positions (0.3 m for the ball, 0.15 m for players).

We did this with two random seeds per match, and also made a copy of each match with home and
away swapped, so the model doesn't favour either side.

### 1.2 Features

The same code builds the features for the sample games and the competition match.

The ball track gets a quick clean-up first. We drop jumps faster than 40 m/s, fill gaps shorter
than half a second, and work out speed from position.

For possession (38 features), everything is turned to face the home team's attacking direction.
We work that out per half from where each team stands on average. The features are:

- the closest player from each team to the ball, and the gap between them (raw and smoothed)
- how many players from each team are within 3 m and 8 m of the ball
- ball speed and position
- the last known gap before the ball went missing, the next one after it comes back, and how
  long ago or how far ahead those were
- team shape over the last second: centre, spread, highest and deepest player, average speed,
  forward movement and how many players are tracked, plus home minus away for each

For dead balls (20 features):

- whether the ball is outside the lines, now and averaged over 1 s and 3 s
- ball speed and how often it's been still
- distance to the nearest line, and the closest player to the ball
- how long since the ball was last seen, and how long until it's seen again
- median player speed over 1, 3 and 6 seconds, and the fastest player
- how many players are tracked, how spread out they are, and the speed gap between the teams

Player speeds come from positions. Anything over 12 m/s is thrown away, because that's usually
the tracker swapping two players.

### 1.3 Classifiers

Both are scikit-learn `HistGradientBoostingClassifier` models with 300 iterations, learning rate
0.05, 31 leaves, at least 200 samples per leaf, L2 of 1 and a fixed seed. They train on every
third frame, since neighbouring frames are nearly identical.

- The possession model learns home vs away, on frames where the ball is in play.
- The dead-ball model learns dead vs in play, on all graded frames.

### 1.4 Decoder and how we tuned it

The decoder has five settings:

- `ha_scale` scales how much the possession probability counts
- `d_scale` scales how much the dead-ball probability counts
- `d_bias` shifts the dead-ball side up or down
- `p_switch` is the cost of switching between H and A
- `p_dead` is the cost of switching between in play and dead

To tune them we trained on two sample matches, predicted the third, and repeated that for all
three. Then we searched a grid of settings and kept whichever gave the best average spellF1 on
those held-out predictions, scored with the organisers' own code.

We ended up with `p_switch` = 13, `p_dead` = 25, `ha_scale` = 1.0, `d_scale` = 0.5 and
`d_bias` = 1.5.

Results on each sample match, scored by a model that never trained on it:

| Match | spellF1 | macroF1 | Organisers' baseline on the same data (spellF1 / macroF1) |
|---|---|---|---|
| Sample 1 | 0.551 | 0.841 | 0.229 / 0.415 |
| Sample 2 | 0.599 | 0.827 | 0.234 / 0.444 |
| Sample 3 | 0.549 | 0.852 | 0.200 / 0.449 |

After that we retrained both classifiers on all three matches and saved them in
`models/model.joblib`. The full training log is in `models/training_report.txt`.

### 1.5 Two rules for the competition match

- Before each half kicks off, the ball is dead. We find the kickoff as the first frame inside
  the Smart Tagging "Kick Off (Start)" tag where the ball moves faster than 3 m/s. This matters
  for the second half, which is graded from a couple of minutes before the kickoff.
- In the 2 seconds before each tagged set piece we nudge the decoder towards "dead" by 1.0
  (in log-odds), since play is usually stopped just before a restart.

Frames outside the graded halves are X. We never use X inside the graded halves.

## 2. What we used and what we left out

| Asset | Used | Why |
|---|---|---|
| Tactical tracking + metadata | Yes | Main input. The metadata tells us which track belongs to which team. |
| Smart Tagging | Yes | Only for the kickoff and the set piece rule. |
| Homography | No | Only needed if you work with the video pixels. |
| Tactical and broadcast video | No | Never opened. |
| Broadcast tracking | No | It isn't synced to the tactical timeline. |
| Public 5 minute ground truth | Only to check the format | Not used for training, tuning or choosing anything. |
| Metrica sample games 1 to 3 | Yes | All of the training and tuning. |

## 3. Training data

We only used the three Metrica Sports sample games, from
github.com/metrica-sports/sample-data. They're public and can be used with credit to Metrica.
The event data was made by Metrica, and our script turned it into frame labels as described in
1.1. In total that's about 430,000 frames, times two noise seeds, times two for the home/away
swap. None of it comes from the competition match.

## 4. Possession rules in our system

- Change of possession only happens when the other team keeps the decoder on its side long
  enough to cover the switch cost. In practice that means several frames of clearly having the
  ball.
- In loose or contested phases the evidence is weak either way, so the ball stays with whoever
  had it last.
- Touches, deflections and aerial duels are too short to cause a switch.
- At a restart, the dead spell ends when the in-play side takes over, and the team that restarts
  is whichever one the possession model favours at that moment.
- A dead ball is always D and never given to a team.

## 5. Limitations and what we'd do next

- The sample games are clean tracking that we made noisy using average numbers. The real
  automatic tracking has its own quirks, like players swapping IDs and the ball going missing
  more often in crowded areas. We chose not to measure those on the competition match, to stay
  on the safe side of the rules. With some external automatic tracking we could copy them
  better.
- Three matches isn't a lot to tune five settings on, so the decoder settings are a bit noisy.
  More open tracking data with events would help.
- Stoppages where the ball stays on the pitch, like fouls and offsides, are the hardest part.
  Right now we only catch them through players slowing down.
- Next we'd like to run a ball detector on the video to fill in the missing 41% (as an input,
  without training it on this video), and try picking up the referee's whistle from the
  broadcast audio.

## 6. How to reproduce

```
pip install -r requirements.txt
python run.py --data <bundle> --team "FC Data-lona"
```

To retrain from the sample games:

```
pip install -r requirements-train.txt
python train.py --sample-data <path to metrica-sports/sample-data>
```

### Log from the submitted run

```
decoder parameters (tuned on external games): {'p_switch': 13, 'p_dead': 25, 'ha_scale': 1.0, 'd_scale': 0.5, 'd_bias': 1.5}
teams: H = Barcelona, A = Atletic Club; 46 track slots; 638 smart-tagging instances (25s)
half 1: kickoff detected at video frame 11189 (4 graded frames before it set dead)
half 2: kickoff detected at video frame 113660 (2852 graded frames before it set dead)
summary (graded frames): H 0.329  A 0.204  D 0.467; 246 possessions; ball seen on 59.2% after bridging
wrote C:\mtah\mtah-possession\mtah-possession\mtah-possession\outputs\FC Data-lona.csv (31s)
```

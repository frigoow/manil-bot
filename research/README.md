# Research code

The code behind the app's bots and behind every number in the main README,
for anyone who wants to inspect it or reproduce it:

- **DMC blueprint**: Deep Monte Carlo self-play, with no search teacher. An
  action-as-input Q-network learns the expected final points of every legal
  card, alongside a **belief network** that predicts who holds which unseen
  card ([manille_dmc_plan.md](manille_dmc_plan.md)).
- **SPARTA**: test-time search on the frozen blueprint. It runs a one-ply
  lookahead over sampled deals, weighted by how likely each one makes the play
  seen so far, and plays out each candidate with the blueprint. An ISMCTS
  search takes over for the last tricks.
- **The rule agent** the app runs (Python reference, and the same C++ the
  phone uses), the **decision layer** that makes the declarations and knocks,
  and the **duplicate-deal harness** that measured all of them.

The C++ (engine, encoders, samplers, rule agent, ISMCTS) is shared with the
app in [`../cpp`](../cpp). The Python here reaches it through the
`manille_ismcts` module that the `MANILLE_PYTHON` build option produces. The
C++ engine decides what is legal and how tricks and deals are scored.
`manille_env/` is a Python game model the evaluation harness drives, checked
against the engine.

## Setup

Linux, Python 3.12, a C++17 compiler and CMake. A CUDA GPU is optional but
makes SPARTA about 20–50 times faster (about 1 s per move on an RTX 3060).

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install -r research/requirements.txt

cmake -B build -S cpp -DMANILLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
cmake --build build -j
cp build/manille_ismcts*.so research/

research/fetch_models.sh        # the trained networks, ~460 MB
cd research                     # every command below runs from here
```

The module links the libtorch inside the installed `torch` package.

## Reproduce the match against SPARTA

The table "Against SPARTA, version 1.0.4" in the main README comes from:

```bash
OMP_NUM_THREADS=1 python scripts/declared_ab.py --deals 500 --seed 116 --only suit \
    --tag rule_vs_sparta --procs 9 --opponent sparta \
    --arm standard '{"def_trump_margin": 1.0, "def_trump_lock_weak": true, "playout_nt_rules": 415}' \
    --arm whole_deal '{"def_trump_margin": 1.0, "def_trump_lock_weak": true, "playout_nt_rules": 415, "playout_net": "artifacts/fast_policy_256.pt"}'
```

- **Deals:** the decision layer makes the declarations and knocks. `--only suit`
  keeps the deals it declares in a suit; `--only mull` keeps the no-trump ones.
  Each deal is played twice, with the teams swapped.
- **Arms:** each arm's settings are added to the rule agent's defaults. These
  arms are the app's settings (`deployed_params()` in
  `cpp/mobile/manille_mobile.cpp`); `whole_deal` lets the learned play-out net
  play the whole simulated deal.
- **Output:** each arm prints points per deal for the rule agent, raw and with
  the doubled scores, with 95% intervals. The results per deal go to
  `runs/declared_ab/<tag>.npz`, so a second arm on the same tag and seed is
  paired deal by deal with the first.
- **Run time:** about 1½ hours per arm with 9 processes and an RTX 3060. Each
  process holds about 1.6 GB of networks.

Without `--opponent`, the opponent is the DMC blueprint on its own (the trump
and mull networks). That is the setting of the 3000-deal A/B tests the
README quotes.

SPARTA's parameters are `SPARTA` in `scripts/declared_ab.py`:
- UCB allocation over 128 sampled deals;
- likelihood temperature 0.1;
- a deviation from the blueprint's card only on paired significance;
- ISMCTS with 5000 iterations from trick 6.

That matches the project's desktop GUI.

## Train a blueprint

`learn/run_dmc.py` runs the training cycle unattended:
1. self-play on the current network (`learn/dmc_selfplay.py`), with an
   ε-ladder and an opponent league;
2. training (`learn/dmc_train.py`): Huber loss on the full-deal returns, plus
   the belief network's likelihood;
3. a checkpoint;
4. duplicate-deal benchmarks against random play and ISMCTS.

A run resumes from `runs/dmc/state.json` after a stop.

```bash
python -u -m learn.run_dmc --smoke                       # one tiny cycle, to check the setup
python -u -m learn.run_dmc --arch gru --run-dir runs/dmc_gru --data-dir data/dmc_gru
```

A cycle is 50,000 self-play deals and 1,500 gradient steps by default; see
`--help` for the rest. The published networks took several runs with warm
starts on one RTX 3060 (weeks of compute):

| File | What it is | Lineage | Measured |
|---|---|---|---|
| `artifacts/dmc_q_blueprint.pt` | MLP Q-network (1024×4, 11M parameters), the first blueprint | DMC from scratch | beats ISMCTS-1000 by 1.4 points per deal; the opponent in the main README's tables |
| `artifacts/dmc_q_gru.pt` | GRU over the public play history plus an MLP head (36M) | DMC warm start, frozen at cycle 207 | +0.31 [+0.14, +0.49] vs the MLP run's cycle 244 (5000 deals); SPARTA on it is +2.67 [+2.02, +3.32] vs ISMCTS-5000 (400 deals) |
| `artifacts/dmc_q_blueprint_best.pt` | the strongest trump blueprint without search, same architecture | warm start, then one cycle of on-policy training with rule-consistent augmentation | +0.11 [+0.03, +0.19] vs its predecessor (10,000 deals); the rule agent's Q blend |
| `artifacts/dmc_q_mull_cond.pt`, `dmc_belief_mull_cond.pt` | no-trump (mull) Q-network and belief network | 105 cycles of no-trump self-play, then 100 cycles on the deals the decision layer actually mulls | +0.73 [+0.19, +1.29] on those deals vs before the refit |
| `artifacts/dmc_belief_best.pt` | trump-deal belief network (512×2): for each unseen card, offsets on the constrained-uniform prior | trained on blueprint self-play, where the true hands are known | the rule agent's world weights (SPARTA weights its worlds by the Q-network's own action likelihoods) |
| `artifacts/fast_policy_256.pt` | small play-out policy (`learn/fast_policy.py`, `cpp/fast_policy.h`) | imitates the blueprint's greedy card from self-play shards (`python -m learn.fast_policy --h1 256 --h2 128`) | the app's "Bot thinking" setting |
| `runs/policy_state_rates.pt`, `policy_rule.pt` + `margin_*_m5.pt` | the decision layer: a margin network and the match-equity tables (`learn/decision_layer.py`, `learn/match_equity.py`) | `learn/dmc_margin_gen.py` + `dmc_margin_train.py` | `policy_state_rates.pt` makes the knocks, `policy_rule.pt` (refit to the rule agent's own deals) chooses trump or mull |

Training data is generated, not published: self-play writes it under `data/`.

## Where things are

| | |
|---|---|
| `learn/dmc_model.py`, `dmc_selfplay.py`, `dmc_replay.py`, `dmc_train.py`, `run_dmc.py`, `dmc_eval.py` | the DMC blueprint: networks, self-play, shards, training, cycle runner, benchmarks |
| `learn/dmc_sparta.py`, `inference.py`, `dmc_endgame_handoff.py` | SPARTA, its world weighting, the ISMCTS endgame |
| `learn/rule_agent.py`, `rule_playout.py` | the rule agent in Python, parity-tested against `cpp/rule_agent.cpp` and `cpp/rule_playout.cpp` |
| `learn/decision_layer.py`, `match_equity.py`, `dmc_margin_*.py` | declarations and knocks that play for the match |
| `learn/evaluate.py`, `declared_eval.py`, `tournament.py`, `scripts/declared_ab.py` | the duplicate-deal harness; `build_agent` in `evaluate.py` maps every agent spec to an agent |
| `learn/encoding.py` | a Python mirror of the C++ observation encoder, for the parity tests only |
| `manille_dmc_plan.md` | the DMC design: the "plan §" references in the code |

Some docstrings refer to other planning notes (`manille_match_plan.md`,
`endgame_leak_plan.md`, `sparta_manille_plan.md`); those are not published.
The design decisions they record are explained in the code.

## Tests

```bash
python -m pytest tests -q        # ~10 minutes; part of them needs the networks
```

They check:
- that the Python mirrors and the C++ match exactly (observation encoder,
  rule play-out, rule agent);
- that every agent plays only legal cards through whole deals;
- the replay format, training steps and the cycle runner.

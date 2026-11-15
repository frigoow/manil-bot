"""Autonomous DMC cycle runner (manille_dmc_plan §1.3, §6).

One cycle = generate G deals (learn.dmc_selfplay on the current EMA
checkpoint) -> train T steps (learn.dmc_train) -> checkpoint + league push
-> duplicate-deal benchmarks (learn.dmc_eval: random every cycle, ISMCTS-500
every cycle, full 100/500/1000/5000 ladder every --full-eval-every) -> log.

Everything is resumable: state.json is written only after a cycle fully
completes; a kill mid-cycle redoes that cycle from scratch (partial shard
dirs are removed first). Subprocesses stream to per-cycle logs; milestones
and hard stops go to the append-only run log (run_log.jsonl + run_log.md).

Hard stops (halt and summarize): non-finite training loss (trainer exit 3),
vs-random regression > 2 pts below the best for 3 consecutive cycles,
shard starvation, repeated eval failures.

    python -u -m learn.run_dmc                 # run forever
    python -u -m learn.run_dmc --smoke         # tiny cycle for validation
    python -u -m learn.run_dmc --max-cycles 3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RANDOM_FLOOR = 31.0      # sanity anchor: anything learned beats random early
MILESTONE_POINTS = (35.0, 37.0)
REGRESSION_PTS = 2.0
REGRESSION_CYCLES = 3
WARMUP_CYCLES = 5


def run_cmd(cmd, log_path):
    """Run a subprocess, streaming stdout/stderr to log_path. Returns the
    process exit code."""
    with open(log_path, "a") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=ROOT)
    return proc.returncode


def parse_marker(log_path, marker):
    """Last line of the form '<marker> {json}' in log_path."""
    with open(log_path) as f:
        for line in reversed(f.readlines()):
            if line.startswith(marker + " "):
                return json.loads(line[len(marker) + 1:])
    return None


def load_state(run_dir):
    path = os.path.join(run_dir, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_state(run_dir, state):
    tmp = os.path.join(run_dir, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(run_dir, "state.json"))


def config_changes(previous, current):
    """Return explicit old/new values for changed runner arguments."""
    missing = object()
    changes = {}
    for key in sorted(set(previous) | set(current)):
        old, new = previous.get(key, missing), current.get(key, missing)
        if old != new:
            changes[key] = {
                "from": None if old is missing else old,
                "to": None if new is missing else new,
            }
    return changes


def log_cycle(run_dir, entry):
    with open(os.path.join(run_dir, "run_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    lines = [f"## cycle {entry['cycle']}  ({entry['timestamp']})"]
    tr = entry.get("train") or {}
    if tr:
        sym = (f"sym={tr['train_sym']:.5f} "
               if tr.get("train_sym") is not None else "")
        lines.append(
            f"- train: q={tr.get('train_q', 0):.4f} "
            f"belief={tr.get('train_belief', 0):.4f} "
            f"{sym}"
            f"val q={tr.get('q_mean', 0):.4f} "
            f"belief={tr.get('belief_mean', 0):.4f} "
            f"(base {tr.get('belief_base_mean', 0):.4f}) "
            f"steps/s={tr.get('steps', 0) / max(tr.get('elapsed', 1), 1):.1f}")
    for ev in entry.get("evals", []):
        if ev["opponent"] == "random":
            lines.append(
                f"- vs random ({ev['deals']} deals): "
                f"points={ev['points']:.2f} [{ev['points_lo']:.2f}, "
                f"{ev['points_hi']:.2f}] diff={ev['diff']:+.2f}")
        else:
            lines.append(
                f"- vs {ev['opponent']} ({ev['deals']} deals): "
                f"diff={ev['diff']:+.2f} [{ev['diff_lo']:+.2f}, "
                f"{ev['diff_hi']:+.2f}]")
    for m in entry.get("milestones", []):
        lines.append(f"- **MILESTONE: {m}**")
    if entry.get("hard_stop"):
        lines.append(f"- **HARD STOP: {entry['hard_stop']}**")
    with open(os.path.join(run_dir, "run_log.md"), "a") as f:
        f.write("\n".join(lines) + "\n")


def count_deals(data_dir, cycle):
    from .dmc_replay import DMC_DEAL_DTYPE, HEADER_DTYPE
    total = 0
    for path in glob.glob(
            os.path.join(data_dir, f"cycle_{cycle:03d}", "shard_*.bin")):
        size = os.path.getsize(path)
        if size > HEADER_DTYPE.itemsize:
            total += (size - HEADER_DTYPE.itemsize) // DMC_DEAL_DTYPE.itemsize
    return total


def benchmark_opponents(anchor, random_deals, anchor_deals, full,
                        full_ismcts_deals):
    """Build a cycle's benchmark list without dropping the drift anchor.

    Full-ladder cycles used to REPLACE the per-cycle anchor.  That made the
    most important checkpoints (cycle 0 and every full-eval interval)
    impossible to rank directly against the incumbent.  Keep the anchor in
    addition to the historical ladder, unless it is already one of its
    ISMCTS opponents.
    """
    opponents = [("random", 2 * random_deals if full else random_deals)]
    if full:
        opponents.extend((f"ismcts:{n}", full_ismcts_deals)
                         for n in (100, 500, 1000, 5000))
    if not full or anchor not in {opp for opp, _ in opponents}:
        opponents.append((anchor, anchor_deals))
    return opponents


def is_full_eval_cycle(cycle, every):
    """Cycle 0 and every `every`-th cycle run the ISMCTS ladder; 0 = never
    (a run whose question is answered by the q: anchor alone, where the
    ladder at 400 deals is ~250 s of noise per full cycle)."""
    if every <= 0:
        return False
    return (cycle + 1) % every == 0 or cycle == 0


def generation_league(league, anchor, league_size, retain_anchor=False):
    """League passed to self-play, optionally pinning a frozen Q anchor.

    Evaluation anchoring alone detects drift after the fact. For long runs,
    a rolling league eventually forgets the strongest policy it is meant to
    improve upon. Reserve one slot for that checkpoint and roll the rest.
    """
    items = list(league)
    if not retain_anchor or not anchor.startswith("q:") or league_size <= 0:
        return items[-league_size:] if league_size > 0 else []
    frozen = anchor[2:]
    recent = [p for p in items if p != frozen]
    n_recent = league_size - 1
    return [frozen] + (recent[-n_recent:] if n_recent else [])


def best_response_error(learner_team, league_frac, anchor, league_size,
                        retain_anchor):
    """Why a --learner-team run is misconfigured, or None.

    The learner-team mask drops seats 1 & 3 from training on the premise
    that they are the frozen target in EVERY deal. That holds only when
    every deal is a league deal and the league is exactly the anchor --
    otherwise the mask silently throws away the learner's own self-play
    plies, or the run exploits a drifting pool instead of the target."""
    if learner_team is None:
        return None
    if learner_team != 0:
        return ("--learner-team must be 0: dmc_selfplay's league plays "
                "seats 1 & 3")
    if league_frac != 1.0:
        return "--learner-team requires --league-frac 1.0"
    if not (anchor.startswith("q:") and retain_anchor and league_size == 1):
        return ("--learner-team requires the league pinned to the frozen "
                "target: --anchor q:<ckpt> --league-size 1 "
                "--retain-anchor-in-league")
    return None


def trainer_learner_args(learner_team):
    """Trainer CLI fragment for --learner-team (forwarding is testable)."""
    return ([] if learner_team is None
            else ["--learner-team", str(learner_team)])


def trainer_lr_args(lr, lr_final):
    """Trainer CLI fragment for the learning rate (forwarding is testable).

    dmc_train's cosine runs over GLOBAL steps and ends at 200k, so every run
    resumed from a late checkpoint (the blueprint is at ~624k) trains at
    --lr-final throughout. A constant rate is --lr X --lr-final X."""
    out = []
    if lr is not None:
        out += ["--lr", repr(lr)]
    if lr_final is not None:
        out += ["--lr-final", repr(lr_final)]
    return out


def generation_exploration_args(eps_ladder):
    """Self-play CLI fragment; kept testable because run_dmc owns the CLI."""
    return (["--eps-ladder", eps_ladder] if eps_ladder is not None else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/dmc")
    ap.add_argument("--data-dir", default="data/dmc")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deals", type=int, default=50000, help="deals per cycle")
    ap.add_argument("--steps", type=int, default=1500, help="grad steps/cycle")
    ap.add_argument("--buffer-deals", type=int, default=150000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--league-frac", type=float, default=0.2)
    ap.add_argument("--eps-ladder", default=None,
                    help="comma-separated exploration ladder forwarded to "
                         "dmc_selfplay (use 0.01 for the low-epsilon arm)")
    # Trainer passthroughs. Adding a flag to learn.dmc_train is not enough:
    # the cyclic runner builds the trainer command itself, so anything it
    # does not forward is unreachable from the normal entry point however
    # well the trainer tests it.
    ap.add_argument("--augment", action="store_true",
                    help="suit-permutation augmentation (learn.dmc_train)")
    ap.add_argument("--q-forced-weight", type=float, default=1.0,
                    help="Q-loss weight for one-legal-move plies "
                         "(learn.dmc_train)")
    ap.add_argument("--sym-consistency", type=float, default=0.0,
                    help="paired suit-consistency weight forwarded to "
                         "learn.dmc_train")
    ap.add_argument("--sym-teacher-q", default=None,
                    help="frozen orbit teacher forwarded to dmc_train")
    ap.add_argument("--sym-teacher-weight", type=float, default=0.0,
                    help="stochastic orbit-distillation weight forwarded "
                         "to dmc_train")
    ap.add_argument("--learner-team", type=int, default=None,
                    help="best-response run: forwarded to dmc_train, which "
                         "then trains only on this team's plies. Requires "
                         "--league-frac 1.0 and the league pinned to the "
                         "q: anchor (--league-size 1 "
                         "--retain-anchor-in-league)")
    ap.add_argument("--lr", type=float, default=None,
                    help="forwarded to dmc_train (its default when unset)")
    ap.add_argument("--lr-final", type=float, default=None,
                    help="forwarded to dmc_train. Past its --total-steps "
                         "(200k global) the cosine sits here, which is "
                         "every warm-started run; pass --lr X --lr-final X "
                         "for a constant rate")
    ap.add_argument("--league-size", type=int, default=10)
    ap.add_argument("--retain-anchor-in-league", action="store_true",
                    help="reserve one self-play league slot for a q: anchor "
                         "instead of eventually rolling it out")
    ap.add_argument("--eval-random-deals", type=int, default=1000)
    ap.add_argument("--eval-ismcts-deals", type=int, default=200)
    ap.add_argument("--anchor", default="ismcts:500",
                    help="per-cycle strength anchor. 'q:<ckpt>' (a frozen "
                         "dmc_q blueprint) is supported as a fast drift "
                         "detector, but the standing anchor is ISMCTS "
                         "(user decision 2026-07-25: comparability beats "
                         "the modest speedup)")
    ap.add_argument("--eval-anchor-deals", type=int, default=1000)
    ap.add_argument("--full-eval-every", type=int, default=5,
                    help="ISMCTS-ladder cadence in cycles; 0 = never")
    ap.add_argument("--full-ismcts-deals", type=int, default=400)
    ap.add_argument("--eval-seed", type=int, default=1000,
                    help="fixed: all cycles play identical eval deals")
    ap.add_argument("--arch", choices=["mlp", "transformer", "gru"],
                    default="mlp")
    ap.add_argument("--choice", action="store_true",
                    help="Phase 6: net-chosen trump. Generation picks trump "
                         "with the newest ChoiceVNet (cycle 0: --choice-warm "
                         "mimic) + --choice-eps exploration; dmc_train also "
                         "fits choice_cycleNNN.pt each cycle")
    ap.add_argument("--choice-eps", type=float, default=0.1)
    ap.add_argument("--choice-warm", default="runs/trump_choice_mimic.pt",
                    help="cycle-0 chooser: the rule mimic keeps the start "
                         "distribution-identical to the rule-trump game")
    ap.add_argument("--choice-steps", type=int, default=200)
    ap.add_argument("--choice-gate-agree", type=float, default=0.5,
                    help="generation switches from the mimic to the trained "
                         "ChoiceVNet only once its val rule-agreement "
                         "reaches this (a noise-level chooser must not "
                         "steer the trump distribution)")
    ap.add_argument("--mull", action="store_true",
                    help="Phase B unconditional mull lineage: generation "
                         "plays every deal at no trump (--mull passthrough) "
                         "and per-cycle evals run at trump=-1. With "
                         "--decl-policy, applies to EVAL only.")
    ap.add_argument("--decl-policy", default=None,
                    help="M6: decision-layer policy file "
                         "(m3_fixed_point.py --policy-out); generation "
                         "declares with it instead of the trump rule or "
                         "--mull (passthrough to dmc_selfplay with "
                         "--decl-mode).")
    ap.add_argument("--decl-mode", choices=["all", "mull-only", "non-mull"],
                    default="all",
                    help="generation slice for --decl-policy: everything, "
                         "the conditional mull corpus (M6.1), or the "
                         "complement at the chooser's suit (M6.2)")
    ap.add_argument("--warm-q", default=None,
                    help="cycle-0 warm start: this checkpoint is the "
                         "generation policy AND the trainer's --resume-q "
                         "(opt/EMA/step restored). Requires --warm-b.")
    ap.add_argument("--warm-b", default=None)
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = forever")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny cycle sizes for pipeline validation")
    args = ap.parse_args()
    if args.mull and args.choice:
        ap.error("--mull and --choice are mutually exclusive: mull fixes "
                 "the declaration, the chooser picks it")
    if args.decl_policy and args.choice:
        ap.error("--decl-policy replaces the declaration mechanism; it "
                 "does not combine with --choice")
    if args.sym_teacher_weight and not args.sym_teacher_q:
        ap.error("--sym-teacher-weight requires --sym-teacher-q")
    if args.sym_teacher_q and not args.sym_teacher_weight:
        ap.error("--sym-teacher-q requires positive --sym-teacher-weight")
    br_error = best_response_error(
        args.learner_team, args.league_frac, args.anchor, args.league_size,
        args.retain_anchor_in_league)
    if br_error:
        ap.error(br_error)

    if args.smoke:
        args.deals, args.steps = 3000, 100
        args.eval_random_deals, args.eval_ismcts_deals = 100, 50
        args.full_ismcts_deals = 50

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    state = load_state(args.run_dir)
    if state is None:
        state = {
            "cycle": 0, "seed": args.seed, "league": [], "buffer_dirs": [],
            "best_random_points": 0.0, "regression_streak": 0,
            "milestones": [], "eval_failures": 0,
            # The run's own configuration, so a finished run can be read back
            # without guessing. state["league"] records WHICH checkpoints were
            # in the pool but never recorded --league-frac, --augment, the
            # arch or the step count, so questions like "did this run use a
            # league at all, and how much" had no answer after the fact. The
            # GRU run predates this and cannot be reconstructed.
            "config": vars(args).copy(),
        }
        save_state(args.run_dir, state)
    elif "config" not in state:
        state["config"] = vars(args).copy()
        state["config_backfilled_at_cycle"] = state["cycle"]
        save_state(args.run_dir, state)
    else:
        current_config = vars(args).copy()
        changes = config_changes(state["config"], current_config)
        if changes:
            # Resuming with a changed anchor or budget is legitimate, but the
            # old code silently left state.json claiming the original CLI.
            # Preserve an audit trail and make the effective configuration
            # authoritative before starting another cycle.
            state.setdefault("config_history", []).append({
                "cycle": state["cycle"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "changes": changes,
            })
            state["config"] = current_config
            save_state(args.run_dir, state)
            print(f"resume config changed at cycle {state['cycle']}: "
                  f"{', '.join(changes)}", flush=True)

    def init_checkpoint():
        """Random-weight initial Q checkpoint so generation has a policy."""
        path = os.path.join(args.run_dir, "q_init.pt")
        if not os.path.exists(path):
            import torch
            from .dmc_model import QGRU, QNet, QTransformer, save_checkpoint
            torch.manual_seed(args.seed)
            if args.arch == "transformer":
                net = QTransformer()
                config = {"arch": "transformer", "d_model": 128,
                          "layers": 4, "heads": 4}
            elif args.arch == "gru":
                net = QGRU(d_model=512, layers=1, hidden=2048, blocks=4)
                config = {"arch": "gru", "d_model": 512, "layers": 1,
                          "hidden": 2048, "blocks": 4}
            else:
                net = QNet()
                config = {"arch": "mlp", "hidden": 1024, "blocks": 4}
            save_checkpoint(path, net, net.state_dict(), None, None, 0,
                            config)
        return path

    while True:
        c = state["cycle"]
        if args.max_cycles and c >= args.max_cycles:
            print(f"reached --max-cycles {args.max_cycles}, stopping",
                  flush=True)
            return
        t_cycle = time.time()
        prev_q = (os.path.join(args.run_dir, f"q_cycle{c - 1:03d}.pt")
                  if c > 0 else (args.warm_q or init_checkpoint()))
        prev_b = (os.path.join(args.run_dir, f"belief_cycle{c - 1:03d}.pt")
                  if c > 0 else None)
        cur_q = os.path.join(args.run_dir, f"q_cycle{c:03d}.pt")
        cur_b = os.path.join(args.run_dir, f"belief_cycle{c:03d}.pt")
        prev_choice = args.choice_warm
        if c > 0 and state.get("choice_rule_agree",
                               0.0) >= args.choice_gate_agree:
            prev_choice = os.path.join(args.run_dir,
                                       f"choice_cycle{c - 1:03d}.pt")
        cur_choice = os.path.join(args.run_dir, f"choice_cycle{c:03d}.pt")
        entry = {"cycle": c, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "milestones": []}

        # --- generate ---
        shard_dir = os.path.join(args.data_dir, f"cycle_{c:03d}")
        if os.path.exists(shard_dir):
            import shutil
            shutil.rmtree(shard_dir)  # partial dir from a killed attempt
        gen_log = os.path.join(args.run_dir, f"gen_cycle{c:03d}.log")
        cmd = [sys.executable, "-u", "-m", "learn.dmc_selfplay",
               "--model", prev_q, "--out", shard_dir,
               "--deals", str(args.deals), "--workers", str(args.workers),
               "--slots", str(args.slots), "--seed", str(args.seed + c),
               "--league-frac", str(args.league_frac),
               "--arch", args.arch]
        cmd += generation_exploration_args(args.eps_ladder)
        if args.choice:
            cmd += ["--choice-model", prev_choice,
                    "--choice-eps", str(args.choice_eps)]
        if args.decl_policy:
            cmd += ["--decl-policy", args.decl_policy,
                    "--decl-mode", args.decl_mode]
        elif args.mull:
            cmd.append("--mull")
        gen_league = generation_league(
            state["league"], args.anchor, args.league_size,
            args.retain_anchor_in_league)
        if gen_league:
            cmd += ["--league"] + gen_league
        # The league is otherwise invisible after the fact: state["league"]
        # holds the rolling list, not what generation actually received.
        print(f"[cycle {c}] generation league (frac {args.league_frac}): "
              f"{gen_league}", flush=True)
        print(f"[cycle {c}] generating {args.deals} deals -> {shard_dir}",
              flush=True)
        rc = run_cmd(cmd, gen_log)
        n_deals = count_deals(args.data_dir, c)
        expected_deals = args.deals
        if args.decl_policy and args.decl_mode != "all":
            # Filtered streams keep only a slice; the starvation threshold
            # must be against the slice, not the stream.
            from .dmc_selfplay import decl_stream
            decls = decl_stream(args.decl_policy, args.seed + c, args.deals,
                                "cuda")
            expected_deals = int(
                (decls == 4).sum() if args.decl_mode == "mull-only"
                else (decls < 4).sum())
        if rc != 0 or n_deals < 0.9 * expected_deals:
            entry["hard_stop"] = (f"shard starvation: rc={rc}, "
                                  f"{n_deals}/{expected_deals} deals")
            log_cycle(args.run_dir, entry)
            print(entry["hard_stop"], flush=True)
            return
        entry["deals"] = n_deals

        # --- train ---
        buffer_dirs = [d for d in state["buffer_dirs"] if os.path.isdir(d)]
        buffer_dirs.append(shard_dir)
        state["buffer_dirs"] = buffer_dirs
        train_log = os.path.join(args.run_dir, f"train_cycle{c:03d}.log")
        cmd = [sys.executable, "-u", "-m", "learn.dmc_train",
               "--data"] + buffer_dirs + [
               "--out-q", cur_q, "--out-b", cur_b,
               "--steps", str(args.steps),
               "--buffer-deals", str(args.buffer_deals),
               "--seed", str(args.seed),
               "--q-forced-weight", str(args.q_forced_weight),
               "--sym-consistency", str(args.sym_consistency),
               "--sym-teacher-weight", str(args.sym_teacher_weight),
               "--arch", args.arch]
        if args.sym_teacher_q:
            cmd += ["--sym-teacher-q", args.sym_teacher_q]
        if args.augment:
            cmd += ["--augment"]
        cmd += trainer_learner_args(args.learner_team)
        cmd += trainer_lr_args(args.lr, args.lr_final)
        if c > 0:
            cmd += ["--resume-q", prev_q, "--resume-b", prev_b]
        elif args.warm_q and args.warm_b:
            cmd += ["--resume-q", args.warm_q, "--resume-b", args.warm_b]
        if args.choice:
            cmd += ["--choice-out", cur_choice,
                    "--choice-steps", str(args.choice_steps)]
            if c > 0:
                # Training always resumes from the previous cycle's trained
                # chooser (full checkpoint). prev_choice is the GENERATION
                # chooser — it stays on the mimic until the agree gate
                # flips and is not a resumable checkpoint.
                cmd += ["--resume-choice",
                        os.path.join(args.run_dir,
                                     f"choice_cycle{c - 1:03d}.pt")]
        print(f"[cycle {c}] training {args.steps} steps on "
              f"{len(buffer_dirs)} dirs", flush=True)
        rc = run_cmd(cmd, train_log)
        if rc == 3:
            entry["hard_stop"] = "non-finite training loss"
            log_cycle(args.run_dir, entry)
            print(entry["hard_stop"], flush=True)
            return
        if rc != 0:
            entry["hard_stop"] = f"trainer failed rc={rc}"
            log_cycle(args.run_dir, entry)
            print(entry["hard_stop"], flush=True)
            return
        entry["train"] = parse_marker(train_log, "VAL")
        if args.choice:
            entry["choice"] = parse_marker(train_log, "CHOICE")
            ch = entry["choice"]
            if ch:
                state["choice_rule_agree"] = ch["choice_rule_agree"]
                print(f"[cycle {c}] choice val={ch['choice_val']:.4f} "
                      f"rule-agree={ch['choice_rule_agree']:.3f} "
                      f"off-rule-data={ch['choice_data_offrule']:.3f} "
                      f"(gen chooser: {os.path.basename(prev_choice)})",
                      flush=True)
        tr = entry["train"]
        if tr:
            # Belief-learning check, visible per cycle in run.log: the
            # belief net must beat its own zero-offset baseline, and the
            # gap must widen across cycles (dmc_train VAL line).
            print(f"[cycle {c}] val q={tr['q_mean']:.4f} "
                  f"belief={tr['belief_mean']:.4f} "
                  f"(baseline {tr['belief_base_mean']:.4f})", flush=True)

        # --- league ---
        state["league"] = (state["league"] + [cur_q])[-args.league_size:]

        # --- benchmark ---
        full_eval = is_full_eval_cycle(c, args.full_eval_every)
        opponents = benchmark_opponents(
            args.anchor, args.eval_random_deals, args.eval_anchor_deals,
            full_eval, args.full_ismcts_deals)
        evals = []
        for opp, n in opponents:
            opp_tag = (opp.replace(":", "").replace("/", "_")
                       .replace(".", "_"))
            ev_log = os.path.join(
                args.run_dir, f"eval_cycle{c:03d}_{opp_tag}.log")
            cmd = [sys.executable, "-u", "-m", "learn.dmc_eval",
                   "--model", cur_q, "--opponent", opp,
                   "--deals", str(n), "--seed", str(args.eval_seed)]
            if args.mull:
                cmd.append("--mull")
            print(f"[cycle {c}] eval vs {opp} ({n} deals)", flush=True)
            rc = run_cmd(cmd, ev_log)
            res = parse_marker(ev_log, "RESULT") if rc == 0 else None
            if res is None:
                state["eval_failures"] += 1
                if state["eval_failures"] >= 3:
                    entry["hard_stop"] = f"eval vs {opp} failed 3x"
                    entry["evals"] = evals
                    log_cycle(args.run_dir, entry)
                    print(entry["hard_stop"], flush=True)
                    return
                continue
            state["eval_failures"] = 0
            evals.append(res)

            if opp == "random":
                pts = res["points"]
                for m in MILESTONE_POINTS:
                    tag = f"{m:.0f}+ pts/deal vs random"
                    if pts >= m and tag not in state["milestones"]:
                        state["milestones"].append(tag)
                        entry["milestones"].append(tag)
                if c >= WARMUP_CYCLES:
                    if pts > state["best_random_points"]:
                        state["best_random_points"] = pts
                        state["regression_streak"] = 0
                    elif pts < state["best_random_points"] - REGRESSION_PTS:
                        state["regression_streak"] += 1
                    else:
                        state["regression_streak"] = 0
            elif opp.startswith("ismcts:"):
                tier = opp.split(":")[1]
                tag = f"beats ISMCTS-{tier} (CI-positive)"
                if res["diff_lo"] > 0 and tag not in state["milestones"]:
                    state["milestones"].append(tag)
                    entry["milestones"].append(tag)
            elif opp.startswith("q:"):
                tag = f"beats frozen anchor {opp[2:]} (CI-positive)"
                if res["diff_lo"] > 0 and tag not in state["milestones"]:
                    state["milestones"].append(tag)
                    entry["milestones"].append(tag)
        entry["evals"] = evals

        if state["regression_streak"] >= REGRESSION_CYCLES:
            entry["hard_stop"] = (
                f"vs-random regression > {REGRESSION_PTS} pts for "
                f"{REGRESSION_CYCLES} cycles")
        entry["elapsed"] = time.time() - t_cycle
        log_cycle(args.run_dir, entry)
        state["cycle"] = c + 1
        save_state(args.run_dir, state)
        print(f"[cycle {c}] done in {entry['elapsed']:.0f}s; "
              f"milestones so far: {state['milestones']}", flush=True)
        if entry.get("hard_stop"):
            print(entry["hard_stop"], flush=True)
            return


if __name__ == "__main__":
    main()

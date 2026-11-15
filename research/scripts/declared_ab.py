"""Paired A/B of rule-agent variants in the deployment setting: duplicate
deals with the decision layer's declarations and knocks
(learn/declared_eval.run_declared_match), every arm against the same opponent
on the same deals, so arm minus arm is paired per deal.

  PYTHONPATH=. venv/bin/python scripts/declared_ab.py --deals 3000 --seed 110 \\
      --only suit --tag trump_plies \\
      --arm base '{}' \\
      --arm k4 '{"playout_net": "artifacts/fast_policy_256.pt", "playout_net_plies": 4}'

Every arm's params are added to the deploy params (cpu nets,
trump_rule_known=False). --only suit / mull keeps just the deals the policy
declares that way. Per-deal results land in runs/declared_ab/<tag>.npz, so a
crash loses at most the arm in progress.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = {"root_q_device": "cpu", "trump_rule_known": False}
OPPONENT = {"type": "dmc_q", "model": "artifacts/dmc_q_blueprint.pt", "cuda": False,
            "mull": {"type": "dmc_q", "model": "artifacts/dmc_q_mull_cond.pt", "cuda": False}}


# The GUI's SPARTA champion (gui/models_registry.py SPARTA_GUI on CHAMPION_PATH):
# UCB over 128 belief-free Q-likelihood worlds, conf deviation rule, ISMCTS-5000
# from trick 6, declaration tilt from the served decision policy.
SPARTA_CFG = {"particles": 128, "delta": 1.0, "tau_likelihood": 0.1, "exact_leaves": False,
              "ismcts_trick": 6, "ismcts_iters": 5000, "alloc": "ucb", "ucb_chunk": 8,
              "deviate_rule": "conf", "conf_c": 1.0, "decl_tilt": "runs/policy_state_rates.pt"}
SPARTA = {"type": "dmc_sparta", "model": "artifacts/dmc_q_gru.pt", **SPARTA_CFG,
          "mull": {"type": "dmc_sparta", "model": "artifacts/dmc_q_mull_cond.pt", **SPARTA_CFG}}


def ci(x):
    m, se = float(np.mean(x)), float(np.std(x) / np.sqrt(len(x)))
    return f"{m:+.2f} [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=110)
    ap.add_argument("--only", choices=["suit", "mull"], default=None)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--policy", default="runs/policy_state_rates.pt")
    ap.add_argument("--opponent", default=None,
                    help="opponent spec as JSON (default: the DMC blueprint, mull net on mull deals); "
                         "\"sparta\" = the GUI's SPARTA champion")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "PARAMS_JSON"), required=True)
    args = ap.parse_args()

    from learn.declared_eval import declare_deals, load_equity_policy, run_declared_match
    from learn.evaluate import make_deals
    t0 = time.time()
    pool = make_deals(args.deals * (8 if args.only == "mull" else 2) if args.only else args.deals, args.seed)
    calls = declare_deals(pool, load_equity_policy(args.policy))
    keep = [i for i, c in enumerate(calls)
            if args.only is None or (c[0] == 4) == (args.only == "mull")][:args.deals]
    deals = [pool[i] for i in keep]
    calls = [calls[i] for i in keep]
    print(f"{len(deals)} deals ({args.only or 'all'}), declared in {time.time() - t0:.0f}s; "
          f"mull {np.mean([c[0] == 4 for c in calls]):.3f}, "
          f"knock {np.mean([c[1] is not None for c in calls]):.3f}", flush=True)

    opponent = (SPARTA if args.opponent == "sparta" else json.loads(args.opponent)
                if args.opponent else OPPONENT)
    os.makedirs("runs/declared_ab", exist_ok=True)
    out = f"runs/declared_ab/{args.tag}.npz"
    res = dict(np.load(out)) if os.path.exists(out) else {}
    for name, params in args.arm:
        if f"{name}_raw" in res:
            print(f"{name}: done earlier", flush=True)
            continue
        spec = {"type": "rule", "params": {**BASE, **json.loads(params)}}
        t = time.time()
        raw, mult, _ = run_declared_match(spec, opponent, len(deals), seed=args.seed,
                                          processes=args.procs, deals=deals, calls=calls)
        res[f"{name}_raw"], res[f"{name}_mult"] = raw, mult
        np.savez(out, **res)
        print(f"{name} {params}: vs opponent raw {ci(raw)}  mult {ci(mult)}  ({time.time() - t:.0f}s)", flush=True)
        first = args.arm[0][0]
        if name != first:
            print(f"      vs {first}: raw {ci(raw - res[first + '_raw'])}  "
                  f"mult {ci(mult - res[first + '_mult'])}", flush=True)


if __name__ == "__main__":
    main()

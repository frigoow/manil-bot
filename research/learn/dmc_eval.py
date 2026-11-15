"""Single-match DMC benchmark CLI for the cycle runner.

Plays one duplicate-deal match of the DMC blueprint (team 0) against an
opponent spec and prints one JSON line on stdout for the orchestrator:

    python -m learn.dmc_eval --model runs/dmc/q_cycle003.pt \
        --opponent random --deals 1000 --seed 42
    python -m learn.dmc_eval --model runs/dmc/q_cycle003.pt \
        --opponent ismcts:500 --deals 200 --seed 42
"""
import argparse
import json

from .evaluate import bootstrap_ci, run_match


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--opponent", required=True,
                    help="'random', 'ismcts:<iters>' or 'q:<checkpoint>' "
                         "(a frozen dmc_q blueprint — ~200x faster than "
                         "ISMCTS at comparable strength; the standing "
                         "anchor is q:artifacts/dmc_q_gru.pt)")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mull", action="store_true",
                    help="play every deal at no trump (Phase B mull evals)")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    spec0 = {"type": "dmc_q", "model": args.model, "cuda": not args.no_cuda}
    if args.opponent == "random":
        spec1 = {"type": "random"}
    elif args.opponent.startswith("ismcts:"):
        spec1 = {"type": "ismcts", "iters": int(args.opponent.split(":")[1])}
    elif args.opponent.startswith("q:"):
        spec1 = {"type": "dmc_q", "model": args.opponent[2:],
                 "cuda": not args.no_cuda}
    else:
        raise SystemExit(f"unknown opponent {args.opponent!r}")

    diffs, t0_points, elapsed = run_match(
        spec0, spec1, args.deals, seed=args.seed, mull=args.mull)
    mean, lo, hi = bootstrap_ci(diffs)
    pmean, plo, phi = bootstrap_ci(t0_points)
    print("RESULT " + json.dumps({
        "model": args.model,
        "opponent": args.opponent,
        "deals": len(diffs),
        "diff": mean, "diff_lo": lo, "diff_hi": hi,
        "points": pmean, "points_lo": plo, "points_hi": phi,
        "elapsed": elapsed,
    }), flush=True)


if __name__ == "__main__":
    main()

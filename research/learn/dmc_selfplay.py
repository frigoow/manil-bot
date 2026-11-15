"""DMC self-play data generation (manille_dmc_plan §1.2).

No search anywhere: all four seats play ε-greedy with the current Q_θ
(action-as-input; all legal candidates batch-evaluated, argmax or uniform
exploration). Cross-game batching mirrors learn/selfplay.py: each worker
interleaves `--slots` deals and every pending decision's candidates go
through the GPU in one batch.

Exploration: per-deal ε from a fixed Ape-X-style ladder (deal_idx % 8), so
every worker covers the whole ladder. Opponent league: per deal, with
probability --league-frac the opposing team (seats 1&3) is played argmax by
a frozen past checkpoint; the partner is always the current policy.

The league checkpoint is drawn ONCE PER WORKER, not per deal -- one model
stays resident per process, so a cycle mixes over the league in proportion
to how many workers drew each member, not per-deal-uniformly. This docstring
used to claim per-deal uniform sampling, which the code never did; worse,
both the league draw and the exploration rng were seeded on cfg["seed"]
alone, and every worker gets the same cfg, so every worker picked the SAME
checkpoint and replayed the SAME exploration coins. Cycles generated before
2026-08-22 have a one-member league and correlated exploration whatever
--league listed. Both rngs now mix in the worker's first deal index.

Declarations mode (M6): `--decl-policy policy.pt --decl-mode ...` replaces
the trump rule with the decision layer's pick per deal. The policy file is
written by `scripts/m3_fixed_point.py --policy-out` and carries the margin
net path, the solved V and the response rates, so generation can declare on
ANY (seed, idx) stream -- including run_dmc's per-cycle seeds, which a
static per-index dump cannot cover. `mull-only` keeps only the deals the
chooser mulls (the conditional mull corpus, M6.1), `non-mull` keeps the
complement at the chooser's suit (M6.2), `all` plays every deal at the
chooser's declaration. Declarations are evaluated at the representative
match state (0,0), matching the fixed-point dump.

Shards are VERSION=2 records (learn/dmc_replay.py): true hands, per-ply
chosen card + ε + legal summary, final team points.

    python -m learn.dmc_selfplay --model runs/dmc/q_cycle000.pt \
        --out data/dmc/cycle_001 --deals 50000 --workers 6 \
        --league runs/dmc/q_cycle000.pt --league-frac 0.2
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from .dmc_replay import DMC_DEAL_DTYPE, write_shard_header
from .selfplay import legal_summary, make_deal

EPS_LADDER = (0.30, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01, 0.005)


def parse_eps_ladder(value):
    """Parse a comma-separated exploration ladder with strict validation."""
    try:
        ladder = tuple(float(item.strip()) for item in value.split(",")
                       if item.strip())
    except ValueError as exc:
        raise ValueError("epsilon ladder must contain numbers") from exc
    if not ladder:
        raise ValueError("epsilon ladder must not be empty")
    if any(not 0.0 <= eps <= 1.0 for eps in ladder):
        raise ValueError("epsilon ladder values must be in [0, 1]")
    return ladder


class Slot:
    """One interleaved game: python engine state + record."""

    def __init__(self, hands, declarer, trump, eps, league_opponents):
        from manille_env.card import Card, Suit
        from manille_env.game_logic import ManilleGame

        self.game = ManilleGame(dealer=declarer)
        self.game.hands = [
            [Card.from_index(c) for c in range(32) if hands[p] & (1 << c)]
            for p in range(4)
        ]
        # trump == -1 is mull (no trump); Suit(-1) would throw.
        self.game.select_trump(Suit(trump) if trump >= 0 else None)
        self.game.start_playing()
        self.declarer = declarer
        self.eps = eps
        self.league_opponents = league_opponents
        self.played_cards = []
        self.played_players = []
        self.record = np.zeros((), dtype=DMC_DEAL_DTYPE)
        self.record["hands"] = hands
        self.record["trump"] = trump
        self.record["declarer"] = declarer
        self.ply = 0


def decl_stream(decl_policy_path, seed, deals, device):
    """The decision layer's declaration for every index of the deterministic
    generation stream. Shared by main() (progress denominator) and
    run_dmc (the shard-starvation threshold): a filtered mode keeps only a
    slice of the stream, and both sides must agree on the slice."""
    import torch

    from .decision_layer import EquityPolicy, MarginOracle
    from .dmc_margin_train import load_margin_net

    spec = torch.load(decl_policy_path, map_location="cpu",
                      weights_only=False)
    cond = bool(torch.load(spec["net"], map_location="cpu",
                           weights_only=False)
                .get("config", {}).get("cond", False))
    policy = EquityPolicy(
        MarginOracle(load_margin_net(spec["net"], device), device, cond=cond),
        spec["V"], spec["p_meegaan"], spec["p_terug"],
        spec["enable_meegaan"], spec["enable_terug"],
        spec.get("price_terug", spec["enable_terug"]))
    masks = np.empty(deals, dtype=np.uint64)
    for idx in range(deals):
        hands, declarer, _ = make_deal(seed, idx)
        masks[idx] = hands[declarer]
    return batch_declare(policy, masks)


# Per-call cost of EquityPolicy.declare is ~5.3 KB of memory that is never
# returned (measured 2026-07-30: linear over 40k calls, survives gc.collect).
# At --deals 230000 a per-deal loop leaks ~1.2 GB, and run_dmc calls this once
# per cycle from a process that lives for the whole run, so it accumulates
# until the machine dies. Batching collapses 230k calls into ~30 and makes the
# leak irrelevant; it is also several times faster, since decl_evs was already
# vectorized and the loop was throwing that away.
DECL_BATCH = 8192


def batch_declare(policy, masks):
    """Declaration per hand mask, batched. Identical to calling
    policy.declare on each mask -- decl_evs is the same code path."""
    masks = np.asarray(masks, dtype=np.uint64).reshape(-1)
    out = np.empty(len(masks), dtype=np.int8)
    for lo in range(0, len(masks), DECL_BATCH):
        chunk = masks[lo:lo + DECL_BATCH]
        out[lo:lo + len(chunk)] = policy.decl_evs(chunk, 0, 0).argmax(1)
    return out


def worker_rng_seeds(seed, first_deal_idx):
    """(exploration seed, league seed) for the worker owning this split.

    Both mix in the worker's first deal index. Every worker is handed the
    same cfg, so seeding on the run seed alone gave byte-identical streams:
    the same exploration coins in every process, and the same single league
    checkpoint drawn everywhere. Deterministic given the split, so runs stay
    reproducible."""
    return (seed, int(first_deal_idx)), (seed, 999983, int(first_deal_idx))


def _play_chunk(worker_args, deal_indices, shard_path):
    import torch

    import manille_ismcts as mi
    from manille_env.card import Card
    from manille_env.game_logic import GamePhase
    from tournament import game_to_cpp_arrays
    from .dmc_model import load_q_model
    from .dmc_tokens import encode_live

    cfg = worker_args
    torch.set_num_threads(1)
    device = "cuda" if cfg["cuda"] and torch.cuda.is_available() else "cpu"
    model = load_q_model(cfg["model"], device)
    league_model = None
    if cfg["league"] and cfg["league_frac"] > 0:
        # Mixed with this worker's first deal index, not just the run seed.
        # Every worker receives the SAME cfg, so seeding on cfg["seed"] alone
        # made all of them draw the same index and load the SAME league
        # checkpoint for the whole cycle -- the league had one member per
        # cycle regardless of how many were passed. deal_indices[0] is unique
        # per worker and fixed by the split, so this decorrelates the workers
        # without giving up run-level reproducibility.
        league_rng = np.random.default_rng(
            worker_rng_seeds(cfg["seed"], deal_indices[0])[1])
        pick = cfg["league"][league_rng.integers(len(cfg["league"]))]
        league_model = load_q_model(pick, device)
    choice_model = None
    if cfg.get("choice_model"):
        from .dmc_trump_choice import choose_trump, load_choice_model
        choice_model = load_choice_model(cfg["choice_model"], device)
    decl_policy = None
    if cfg.get("decl_policy"):
        from .decision_layer import EquityPolicy, MarginOracle
        from .dmc_margin_train import load_margin_net
        spec = torch.load(cfg["decl_policy"], map_location="cpu",
                          weights_only=False)
        cond = bool(torch.load(spec["net"], map_location="cpu",
                               weights_only=False)
                    .get("config", {}).get("cond", False))
        decl_policy = EquityPolicy(
            MarginOracle(load_margin_net(spec["net"], device), device,
                         cond=cond),
            spec["V"], spec["p_meegaan"], spec["p_terug"],
            spec["enable_meegaan"], spec["enable_terug"],
            spec.get("price_terug", spec["enable_terug"]))
    # Same reason, and this one matters more: this rng drives the ε-greedy
    # exploration draws and the per-deal league coin. Seeded on cfg["seed"]
    # alone it was byte-identical in every worker, so the k-th exploration
    # decision was the same coin in all of them -- correlated exploration
    # dressed up as W independent streams.
    rng = np.random.default_rng(
        worker_rng_seeds(cfg["seed"], deal_indices[0])[0])
    arch = cfg.get("arch", "mlp")

    @torch.no_grad()
    def batch_q(m, rows):
        """rows: list of per-slot dicts with obs/tokens and 'act'
        [n, ACT_DIM] -> list of q[n] per slot."""
        counts = [r["act"].shape[0] for r in rows]
        qs = np.empty(sum(counts), dtype=np.float32)
        for lo in range(0, len(rows), 64):
            chunk = rows[lo : lo + 64]
            cc = [r["act"].shape[0] for r in chunk]
            act = torch.from_numpy(
                np.concatenate([r["act"] for r in chunk], axis=0)).to(device)
            if arch == "gru":
                # Trunk once per decision row; expand only the embedding.
                raw = torch.from_numpy(
                    np.stack([r["raw"] for r in chunk])).to(device)
                n_raw = torch.from_numpy(np.array(
                    [r["n_raw"] for r in chunk], dtype=np.int64)).to(device)
                glob = torch.from_numpy(
                    np.stack([r["glob"] for r in chunk])).to(device)
                reps = torch.as_tensor(cc, device=device)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    last = m.trunk(raw, n_raw)
                    q = m.head(torch.repeat_interleave(last, reps, dim=0),
                               torch.repeat_interleave(glob, reps, dim=0),
                               act)
            elif arch == "transformer":
                rep = lambda key: torch.from_numpy(np.repeat(
                    np.stack([r[key] for r in chunk]), cc, axis=0)).to(device)
                raw, glob, hand = rep("raw"), rep("glob"), rep("hand")
                n_raw = torch.from_numpy(np.repeat(
                    np.array([r["n_raw"] for r in chunk]), cc)).to(device)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    q = m(raw, n_raw, glob, hand, act)
            else:
                obs = torch.from_numpy(np.repeat(
                    np.stack([r["obs"] for r in chunk]), cc, axis=0)
                ).to(device)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    q = m(obs, act)
            qs[sum(counts[:lo]) : sum(counts[:lo]) + sum(cc)] = (
                q.float().cpu().numpy())
        out, pos = [], 0
        for n in counts:
            out.append(qs[pos : pos + n])
            pos += n
        return out

    records = []
    n_done = 0
    M = cfg["slots"]

    def flush():
        nonlocal records
        if not records:
            return
        arr = np.asarray(records, dtype=DMC_DEAL_DTYPE)
        write_shard_header(shard_path, cfg["seed"])
        with open(shard_path, "ab") as f:
            f.write(arr.tobytes())
        records = []

    # Declarations for the whole worker's stream in one batched pass, and the
    # rejected indices dropped BEFORE slots are filled. Deciding inside the
    # slot loop and `continue`ing past rejects silently shrank every wave to
    # the acceptance rate -- at mull-only's ~16% a 32-slot wave carried ~5 live
    # games, which is why this ran at 44 deals/s against the unconditional mull
    # run's 237. Filtering first restores full waves.
    decl_of = {}
    if decl_policy is not None:
        masks = np.empty(len(deal_indices), dtype=np.uint64)
        for j, idx in enumerate(deal_indices):
            hands, declarer, _ = make_deal(cfg["seed"], idx)
            masks[j] = hands[declarer]
        ds = batch_declare(decl_policy, masks)
        mode = cfg["decl_mode"]
        kept = []
        for idx, d in zip(deal_indices, ds):
            d = int(d)
            if mode == "mull-only" and d != 4:
                continue
            if mode == "non-mull" and d == 4:
                continue
            decl_of[idx] = d
            kept.append(idx)
        deal_indices = kept

    for chunk_start in range(0, len(deal_indices), M):
        slots = []
        for idx in deal_indices[chunk_start : chunk_start + M]:
            hands, declarer, trump = make_deal(cfg["seed"], idx)
            if decl_policy is not None:
                # The decision layer's pick replaces the trump rule
                # (declaration 4 = mull -> trump -1).
                trump = -1 if decl_of[idx] == 4 else decl_of[idx]
            elif cfg.get("mull"):
                # Unconditional mull lineage: every deal is played at no
                # trump, over the FULL deal distribution. No selection, so
                # no hand-quality threshold has to be invented -- and the
                # sampler's trump-rule constraint correctly switches itself
                # off at trump = -1, because there is nothing to condition
                # on. Conditioning on "hands worth a mull" is Phase D, and
                # arrives as a learned chooser tilt, not a rule.
                trump = -1
            elif choice_model is not None:
                trump = choose_trump(choice_model, hands[declarer], device,
                                     cfg["choice_eps"], rng)
            league = (
                league_model is not None and rng.random() < cfg["league_frac"]
            )
            ladder = cfg.get("eps_ladder", EPS_LADDER)
            slots.append(
                Slot(hands, declarer, trump, ladder[idx % len(ladder)],
                     league))

        active = list(slots)
        while active:
            # Encode every pending decision, grouped by the acting model.
            rows_cur, rows_lg, metas = [], [], []
            for slot in active:
                pid = slot.game.current_player
                arrays = game_to_cpp_arrays(
                    slot.game, slot.played_cards, slot.played_players)
                enc = mi.encode_q_observation(
                    *arrays, slot.declarer, pid)
                n = int(enc["n_legal"])
                assert n > 0, "playing phase but no legal moves"
                use_league = slot.league_opponents and pid % 2 == 1
                own_mask = 0
                if arch == "transformer":  # QGRU never reads hand tokens
                    for c in slot.game.hands[pid]:
                        own_mask |= 1 << c.to_index()
                n_plays = len(slot.played_cards) + len(
                    slot.game.current_trick)
                row = encode_live(enc["obs"], n_plays, own_mask)
                row["obs"] = enc["obs"]
                row["act"] = enc["act"][:n]
                if use_league:
                    rows_lg.append(row)
                else:
                    rows_cur.append(row)
                metas.append((slot, enc["legal"][:n], use_league))

            qs_cur = batch_q(model, rows_cur) if rows_cur else []
            qs_lg = batch_q(league_model, rows_lg) if rows_lg else []
            it_cur, it_lg = iter(qs_cur), iter(qs_lg)

            still = []
            for slot, legal, use_league in metas:
                q = next(it_lg) if use_league else next(it_cur)
                eps = 0.0 if use_league else slot.eps
                if eps > 0.0 and rng.random() < eps:
                    j = int(rng.integers(len(legal)))
                else:
                    j = int(np.argmax(q))
                chosen = int(legal[j])

                pid = slot.game.current_player
                rec = slot.record["plies"][slot.ply]
                rec["player"] = pid
                rec["card"] = chosen
                rec["summary"] = legal_summary(slot.game, pid)
                rec["nlegal"] = len(legal)
                rec["eps"] = eps

                pre = list(slot.game.current_trick)
                card = Card.from_index(chosen)
                assert slot.game.play_card(pid, card), \
                    "Q policy played an illegal card"
                if len(pre) == 3:
                    for tp_, tc_ in pre + [(pid, card)]:
                        slot.played_cards.append(tc_.to_index())
                        slot.played_players.append(tp_)
                slot.ply += 1

                if slot.game.phase != GamePhase.PLAYING:
                    slot.record["pts0"] = slot.game.hand_points[0]
                    slot.record["pts1"] = slot.game.hand_points[1]
                    records.append(slot.record)
                    n_done += 1
                else:
                    still.append(slot)
            active = still
        flush()
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deals", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--league", nargs="*", default=[],
                    help="frozen checkpoints for the opponent league")
    ap.add_argument("--league-frac", type=float, default=0.2)
    ap.add_argument("--eps-ladder", default=None,
                    help="comma-separated per-deal exploration rates; "
                         "e.g. 0.01 for the fresh low-epsilon control. "
                         "Omitted keeps the historical ladder")
    ap.add_argument("--arch", choices=["mlp", "transformer", "gru"],
                    default="mlp")
    ap.add_argument("--choice-model", default=None,
                    help="trump-choice net: when set, the dealer's trump is "
                         "chosen by the net (argmax, or uniform with "
                         "--choice-eps) instead of apply_trump_rule")
    ap.add_argument("--choice-eps", type=float, default=0.1)
    ap.add_argument("--mull", action="store_true",
                    help="mull lineage: play EVERY deal at no trump "
                         "(trump = -1), over the full deal distribution. "
                         "Mutually exclusive with --choice-model.")
    ap.add_argument("--decl-policy", default=None,
                    help="M6: policy file from m3_fixed_point.py "
                         "--policy-out; the decision layer declares every "
                         "deal (replacing the trump rule). Mutually "
                         "exclusive with --mull and --choice-model.")
    ap.add_argument("--decl-mode", choices=["all", "mull-only", "non-mull"],
                    default="all",
                    help="all: play every deal at the chooser's pick. "
                         "mull-only: keep only deals the chooser mulls "
                         "(M6.1 corpus). non-mull: keep the complement at "
                         "the chooser's suit (M6.2 corpus).")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if args.mull and args.choice_model:
        ap.error("--mull and --choice-model are mutually exclusive: mull "
                 "fixes the declaration, the chooser picks it")
    if args.decl_policy and (args.mull or args.choice_model):
        ap.error("--decl-policy replaces the declaration mechanism; it "
                 "combines with neither --mull nor --choice-model")
    if args.decl_policy and not os.path.exists(args.decl_policy):
        ap.error(f"--decl-policy not found: {args.decl_policy}")
    try:
        eps_ladder = (parse_eps_ladder(args.eps_ladder)
                      if args.eps_ladder is not None else EPS_LADDER)
    except ValueError as exc:
        ap.error(str(exc))

    os.makedirs(args.out, exist_ok=True)
    cfg = dict(
        model=args.model, slots=args.slots, seed=args.seed,
        cuda=not args.no_cuda, league=list(args.league),
        league_frac=args.league_frac, arch=args.arch,
        choice_model=args.choice_model, choice_eps=args.choice_eps,
        mull=args.mull,
        decl_policy=args.decl_policy, decl_mode=args.decl_mode,
        eps_ladder=eps_ladder,
    )

    idx_splits = [
        list(range(w, args.deals, args.workers)) for w in range(args.workers)
    ]
    if not args.decl_policy or args.decl_mode == "all":
        total = sum(len(x) for x in idx_splits)
    else:
        # Skipped deals never become slots, so the progress denominator is
        # the count this mode actually generates. One cheap parent-side
        # pass over the deterministic stream gives it exactly.
        decls = decl_stream(args.decl_policy, args.seed, args.deals,
                            "cuda" if not args.no_cuda else "cpu")
        total = int((decls == 4).sum() if args.decl_mode == "mull-only"
                    else (decls < 4).sum())
        print(f"decl-policy pass: mull {(decls == 4).mean():.1%}, "
              f"generating {total} deals in {args.decl_mode} mode",
              flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = [
            pool.apply_async(
                _play_chunk,
                (cfg, idxs, os.path.join(args.out, f"shard_{w}.bin")),
            )
            for w, idxs in enumerate(idx_splits)
        ]
        while True:
            time.sleep(10)
            written = 0
            for w in range(args.workers):
                p = os.path.join(args.out, f"shard_{w}.bin")
                if os.path.exists(p) and os.path.getsize(p) > 24:
                    written += (os.path.getsize(p) - 24) // DMC_DEAL_DTYPE.itemsize
            dt = time.time() - t0
            print(f"  [{written}/{total}] {written / max(dt, 1e-9):.1f} deals/s",
                  flush=True)
            if all(r.ready() for r in results):
                break
        n = sum(r.get() for r in results)
    dt = time.time() - t0
    print(f"Done: {n} deals in {dt:.0f}s ({n / max(dt, 1e-9):.1f} deals/s)")


if __name__ == "__main__":
    main()

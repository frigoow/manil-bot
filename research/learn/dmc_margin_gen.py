"""M2 deal-outcome margin data (manille_match_plan.md §4a).

For each sampled declarer hand and each of the 5 declarations (4 suits +
mull), deal the remaining 24 cards uniformly to the other three seats
(dealer = declarer) and roll the deal out with the frozen blueprints -- the
mull model for the mull branch, the trump champion for the suit branches.
The row is (declarer hand mask, declaration, signed margin from the declaring
team, declarer seat); `dmc_margin_expand` then recovers the other three seats'
views from the same rollouts, and `dmc_margin_train` fits the distribution.

Games are interleaved and batched across slots, the same way
`learn/dmc_selfplay.py` does it. That is not an optimisation detail, it is the
difference between this being a 35-minute job and a 6-hour one: the sequential
predecessor asked the 36M-param GRU for one decision at a time (<=8 candidate
rows per forward, trunk recomputed over the whole history every ply) and ran
at 22 deals/s, against 237 deals/s for the batched self-play generator on the
same model and GPU. Every deal is exactly 32 plies, so a wave of slots
finishes in lockstep and no slot idles.

Determinism is load-bearing and must not be casually changed. The rng is
seeded per (hand, declaration) and drawn once per rollout, and rows are
written in canonical (hand, declaration, rollout) order regardless of the
order slots actually run in -- `dmc_margin_expand` replays exactly that stream
to recover the hands this file does not store. Play is fp32 argmax by default
so results match `QNetAgent` decision for decision; --fp16 is faster but can
flip a near-tie.

  PYTHONPATH=. venv/bin/python -m learn.dmc_margin_gen \
      --hands 50000 --rollouts 4 --workers 4 --slots 64 \
      --out data/margin/round_002
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from .evaluate import make_deals

TRUMP_Q = "artifacts/dmc_q_gru.pt"
MULL_Q = "runs/dmc_mull/q_cycle105.pt"
MULL = 4

ROW_DTYPE = np.dtype([("hand", "<u8"), ("declaration", "u1"),
                      ("margin", "i1"), ("declarer", "u1")])

_STATE = {}


def deal_rollouts(deals, hi, declaration, rollouts, seed):
    """The (hands-by-seat, declarer, declarer mask) for every rollout of one
    (hand, declaration) cell.

    This rng stream is the contract with `dmc_margin_expand`, which replays it
    to recover the three hands the shard does not store. Changing the seeding
    or the draw order silently invalidates every expanded shard."""
    hands, declarer = deals[hi]
    dec_mask = 0
    dec_set = set()
    for c in hands[declarer]:
        dec_mask |= 1 << c.to_index()
        dec_set.add(c.to_index())
    rest = [c for c in range(32) if c not in dec_set]
    rng = np.random.default_rng(
        (seed * 1000003 + hi * 41 + declaration * 7) % (2**63 - 1))

    from manille_env.card import Card
    out = []
    for _ in range(rollouts):
        perm = rng.permutation(rest)
        full, k = [], 0
        for seat in range(4):
            if seat == declarer:
                full.append(list(hands[declarer]))
            else:
                full.append([Card.from_index(int(c))
                             for c in perm[k * 8:(k + 1) * 8]])
                k += 1
        out.append((full, declarer, dec_mask))
    return out


def _margin(game, declarer):
    p0, p1 = game.hand_points
    if p0 == p1:
        return 0
    win = 0 if p0 > p1 else 1
    return (max(p0, p1) - 30) * (1 if win == declarer % 2 else -1)


def _batch_q(model, rows, device, fp16):
    """rows: per-decision dicts with 'act' [n, ACT_DIM] plus the encoder
    payload -> list of q[n]. Mirrors dmc_selfplay's batching: for the GRU the
    candidate-independent trunk runs ONCE per decision and only the pooled
    embedding is expanded over candidates."""
    import torch

    from .dmc_model import QGRU

    counts = [r["act"].shape[0] for r in rows]
    qs = np.empty(sum(counts), dtype=np.float32)
    written = 0
    for lo in range(0, len(rows), 64):
        chunk = rows[lo:lo + 64]
        cc = [r["act"].shape[0] for r in chunk]
        act = torch.from_numpy(
            np.concatenate([r["act"] for r in chunk], axis=0)).to(device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=fp16 and device == "cuda"):
            if isinstance(model, QGRU):
                raw = torch.from_numpy(
                    np.stack([r["raw"] for r in chunk])).to(device)
                n_raw = torch.from_numpy(np.array(
                    [r["n_raw"] for r in chunk], dtype=np.int64)).to(device)
                glob = torch.from_numpy(
                    np.stack([r["glob"] for r in chunk])).to(device)
                reps = torch.as_tensor(cc, device=device)
                last = model.trunk(raw, n_raw)
                q = model.head(torch.repeat_interleave(last, reps, dim=0),
                               torch.repeat_interleave(glob, reps, dim=0), act)
            else:
                obs = torch.from_numpy(np.repeat(
                    np.stack([r["obs"] for r in chunk]), cc, axis=0)).to(device)
                q = model(obs, act)
        n = sum(cc)
        qs[written:written + n] = q.float().cpu().numpy()
        written += n
    out, pos = [], 0
    for n in counts:
        out.append(qs[pos:pos + n])
        pos += n
    return out


def play_wave(model, wave, declaration, device, fp16):
    """Run a batch of independent deals to completion, one ply at a time
    across all of them. Returns the signed margin per entry, in wave order."""
    import torch

    import manille_ismcts as mi
    from manille_env.card import Card, Suit
    from manille_env.game_logic import GamePhase, ManilleGame
    from tournament import game_to_cpp_arrays
    from .dmc_model import QGRU
    from .dmc_tokens import encode_live

    is_gru = isinstance(model, QGRU)
    slots = []
    for hands, declarer, _ in wave:
        game = ManilleGame(dealer=declarer)
        game.hands = [list(h) for h in hands]
        game.select_trump(Suit(declaration) if declaration < MULL else None)
        game.start_playing()
        slots.append({"game": game, "declarer": declarer, "pc": [], "pp": []})

    active = list(slots)
    while active:
        rows, metas = [], []
        for s in active:
            game = s["game"]
            pid = game.current_player
            enc = mi.encode_q_observation(
                *game_to_cpp_arrays(game, s["pc"], s["pp"]),
                s["declarer"], pid)
            n = int(enc["n_legal"])
            assert n > 0, "playing phase but no legal moves"
            if is_gru:
                n_plays = len(s["pc"]) + len(game.current_trick)
                row = encode_live(enc["obs"], n_plays, 0)
            else:
                row = {}
            row["obs"] = enc["obs"]
            row["act"] = enc["act"][:n]
            rows.append(row)
            metas.append((s, enc["legal"][:n], pid))

        with torch.no_grad():
            qs = _batch_q(model, rows, device, fp16)

        still = []
        for (s, legal, pid), q in zip(metas, qs):
            game = s["game"]
            card = Card.from_index(int(legal[int(np.argmax(q))]))
            pre = list(game.current_trick)
            assert game.play_card(pid, card), "blueprint played an illegal card"
            if len(pre) == 3:
                for tp_, tc_ in pre + [(pid, card)]:
                    s["pc"].append(tc_.to_index())
                    s["pp"].append(tp_)
            if game.phase == GamePhase.PLAYING:
                still.append(s)
        active = still

    return [_margin(s["game"], s["declarer"]) for s in slots]


def _worker_init(deals, trump_q, mull_q, cuda, seed, rollouts, slots, fp16):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    from .dmc_model import load_q_model

    torch.set_num_threads(1)
    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
    _STATE.update(deals=deals, seed=seed, rollouts=rollouts, slots=slots,
                  device=device, fp16=fp16,
                  models={"trump": load_q_model(trump_q, device),
                          "mull": load_q_model(mull_q, device)})


def _work(hand_indices):
    """One chunk of hands -> rows in canonical order.

    Slots are grouped by declaration so a wave needs one model, and results
    are scattered back by job id, so the batching never perturbs row order."""
    st = _STATE
    rollouts, per_hand = st["rollouts"], 5 * st["rollouts"]
    out = np.empty(len(hand_indices) * per_hand, dtype=ROW_DTYPE)

    for declaration in range(5):
        model = st["models"]["mull" if declaration == MULL else "trump"]
        jobs = []
        for local, hi in enumerate(hand_indices):
            cells = deal_rollouts(st["deals"], hi, declaration, rollouts,
                                  st["seed"])
            for j, cell in enumerate(cells):
                jobs.append((local * per_hand + declaration * rollouts + j,
                             cell))
        for lo in range(0, len(jobs), st["slots"]):
            wave = jobs[lo:lo + st["slots"]]
            margins = play_wave(model, [c for _, c in wave], declaration,
                                st["device"], st["fp16"])
            for (job_id, (_, declarer, dec_mask)), m in zip(wave, margins):
                out[job_id] = (dec_mask, declaration, m, declarer)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", type=int, default=50000)
    ap.add_argument("--rollouts", type=int, default=4,
                    help="rollouts per (hand, declaration). Prefer MORE "
                         "hands over more rollouts: at equal row count, 4500 "
                         "hands x 5 beat 1125 x 20 on held-out NLL (6.008 vs "
                         "6.717) — the net pools across hands, so per-cell "
                         "noise averages out but hand coverage does not")
    ap.add_argument("--chunk", type=int, default=200,
                    help="hands per chunk — the incremental-write unit")
    ap.add_argument("--slots", type=int, default=64,
                    help="deals interleaved per wave; every deal is 32 plies "
                         "so a wave finishes in lockstep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/margin/round_002")
    ap.add_argument("--trump-q", default=TRUMP_Q)
    ap.add_argument("--mull-q", default=MULL_Q)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many leading chunks (resume)")
    ap.add_argument("--fp16", action="store_true",
                    help="autocast the rollouts. Faster, but a near-tie "
                         "argmax can flip, so results no longer match "
                         "QNetAgent decision for decision")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    deals = make_deals(args.hands, args.seed)
    chunks = [list(range(c, min(c + args.chunk, args.hands)))
              for c in range(0, args.hands, args.chunk)][args.skip:]
    shard = os.path.join(args.out, "shard_all.bin")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(deals, args.trump_q, args.mull_q,
                            not args.no_cuda, args.seed, args.rollouts,
                            args.slots, args.fp16)) as pool:
        for n_done, rows in enumerate(pool.imap(_work, chunks), start=1):
            with open(shard, "ab") as f:
                rows.tofile(f)
            dt = time.time() - t0
            hands_done = n_done * args.chunk
            print(f"  [chunk {n_done + args.skip}/{len(chunks) + args.skip}] "
                  f"{hands_done * 5 * args.rollouts / dt:.0f} deals/s, "
                  f"eta {(len(chunks) - n_done) * dt / n_done / 60:.0f} min",
                  flush=True)
    print(f"Done: {shard} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

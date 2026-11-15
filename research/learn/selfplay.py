"""ExIt self-play data generation (plan Phase 2, actor side).

Every seat is played by the NN-guided actor: ISMCTS with B_phi-weighted
determinizations, pi_theta priors/rollouts and v_theta leaf evals (config:
--leaf value --puct is the AlphaZero-style default).

Cross-game batching (review §1.1): each worker interleaves `--slots` deals;
each deal runs a resumable BatchedISMCTS at wave=1 — reference-quality
sequential search — and NN evaluations are batched across all slots at once.
Output shards are byte-identical in layout to manille_generate's, so
learn/replay.py reads both.

    python -m learn.selfplay --model runs/m1a.pt --out data/p2_g1 \
        --deals 10000 --sims 150 --leaf value --puct --particles 64
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from .replay import DEAL_DTYPE, HEADER_DTYPE, MAGIC, VERSION


def legal_summary(game, player_id):
    """Factual legal-set summary (diagnostics only, mirrors generate.cpp)."""
    if not game.current_trick:
        return 0
    led_suit = game.current_trick[0][1].suit
    hand = game.hands[player_id]
    summary = 0
    if any(c.suit == led_suit for c in hand):
        summary |= 1
    trump = game.trump_suit
    if trump is not None and any(c.suit == trump for c in hand):
        summary |= 2
    if trump is not None:
        trumps_played = [c for _, c in game.current_trick if c.suit == trump]
        if trumps_played:
            top = max(c.rank for c in trumps_played)
            if any(c.suit == trump and c.rank > top for c in hand):
                summary |= 4
    return summary


def make_deal(seed, idx):
    """Deterministic per-deal content: hands (as masks) + rotating declarer,
    trump by the C++ rule."""
    import manille_ismcts as mi

    rng = np.random.default_rng((seed, idx))
    perm = rng.permutation(32)
    hands = np.zeros(4, dtype=np.int64)
    for p in range(4):
        for c in perm[p * 8 : (p + 1) * 8]:
            hands[p] |= np.int64(1) << int(c)
    declarer = idx % 4
    trump = int(mi.apply_trump_rule(hands[declarer]))
    return hands, declarer, trump


class Slot:
    """One interleaved game: python engine state + resumable solver + record."""

    def __init__(self, hands, declarer, trump, solver, cfg):
        from manille_env.card import Card, Suit
        from manille_env.game_logic import ManilleGame

        self.game = ManilleGame(dealer=declarer)
        self.game.hands = [
            [Card.from_index(c) for c in range(32) if hands[p] & (1 << c)]
            for p in range(4)
        ]
        self.game.select_trump(Suit(trump) if trump >= 0 else None)
        self.game.start_playing()
        self.hands_mask = hands
        self.declarer = declarer
        self.trump = trump
        self.solver = solver
        self.cfg = cfg
        self.played_cards = []
        self.played_players = []
        self.record = np.zeros((), dtype=DEAL_DTYPE)
        self.record["hands"] = hands
        self.record["trump"] = trump
        self.record["declarer"] = declarer
        self.ply = 0
        self.started = False


def _play_chunk(worker_args, deal_indices, shard_path):
    import torch

    import manille_ismcts as mi
    from manille_env.game_logic import GamePhase
    from tournament import game_to_cpp_arrays
    from .agents import (belief_weighted_particles, encode_current,
                         make_policy_value_fn)
    from .model import load_model

    cfg = worker_args
    torch.set_num_threads(1)
    device = "cuda" if cfg["cuda"] and torch.cuda.is_available() else "cpu"
    model = load_model(cfg["model"], device)
    model.eval()
    # Separate belief checkpoint (defaults to the policy model): lets belief
    # and policy improvements promote independently (review R8).
    belief_model = model
    if cfg.get("belief_model"):
        belief_model = load_model(cfg["belief_model"], device)
        belief_model.eval()
    pv_fn = make_policy_value_fn(model, device, temperature=cfg["temperature"])

    rng = np.random.default_rng(cfg["seed"])

    def new_solver():
        return mi.BatchedISMCTS(
            sims=cfg["sims"], wave=1, mode=cfg["leaf"], puct=cfg["puct"],
            exploration=cfg["exploration"], c_puct=cfg["c_puct"],
            dirichlet_epsilon=cfg["dirichlet_epsilon"],
            partner_belief=cfg["partner_belief"],
        )

    records = []
    n_done = 0
    M = cfg["slots"]

    def flush():
        nonlocal records
        if not records:
            return
        arr = np.asarray(records, dtype=DEAL_DTYPE)
        with open(shard_path, "ab") as f:
            if os.path.getsize(shard_path) == 0:
                header = np.zeros((), dtype=HEADER_DTYPE)
                header["magic"] = MAGIC
                header["version"] = VERSION
                header["iters"] = cfg["sims"]
                header["seed"] = cfg["seed"]
                f.write(header.tobytes())
            f.write(arr.tobytes())
        records = []

    for chunk_start in range(0, len(deal_indices), M):
        slots = []
        for idx in deal_indices[chunk_start : chunk_start + M]:
            hands, declarer, trump = make_deal(cfg["seed"], idx)
            slots.append(Slot(hands, declarer, trump, new_solver(), cfg))

        active = list(slots)
        while active:
            # Start searches for slots at a fresh decision point.
            for slot in active:
                if slot.started:
                    continue
                slot.started = True
                pid = slot.game.current_player
                arrays = game_to_cpp_arrays(
                    slot.game, slot.played_cards, slot.played_players)
                particles, weights = None, None
                if cfg["particles"] > 0:
                    obs = encode_current(slot.game, pid, slot.played_cards,
                                         slot.played_players, slot.declarer)
                    particles, weights = belief_weighted_particles(
                        belief_model, device, obs, arrays, pid,
                        cfg["particles"], slot.declarer)
                slot.solver.begin(*arrays, slot.declarer,
                                  particles=particles, particle_weights=weights)

            # Collect pending evals across all slots and batch them.
            pending = []
            for slot in active:
                if not slot.solver.done():
                    pending.append((slot, slot.solver.pending_obs()))
            if pending:
                batch = np.concatenate([o for _, o in pending], axis=0)
                probs, values = pv_fn(batch)
                pos = 0
                for slot, obs in pending:
                    k = obs.shape[0]
                    slot.solver.provide(
                        np.ascontiguousarray(probs[pos : pos + k]).reshape(-1),
                        np.ascontiguousarray(values[pos : pos + k]),
                    )
                    pos += k

            # Finish moves whose search is done.
            still = []
            for slot in active:
                if not slot.solver.done():
                    still.append(slot)
                    continue
                result = slot.solver.result()
                visit_counts = result["visit_counts"]
                pid = slot.game.current_player
                legal = slot.game.get_legal_moves(pid)
                if not visit_counts:  # defensive: no children expanded
                    visit_counts = {legal[0].to_index(): 1}

                rec = slot.record["plies"][slot.ply]
                rec["player"] = pid
                rec["summary"] = legal_summary(slot.game, pid)
                rec["nlegal"] = len(legal)
                for i, (card_idx, count) in enumerate(
                        sorted(visit_counts.items())[:8]):
                    rec["visit_cards"][i] = card_idx
                    rec["visit_counts"][i] = count

                # Move selection: sample from visits in tricks 0-1
                # (exploration anneal per plan §6), argmax afterwards.
                trick = slot.ply // 4
                cards = np.array(list(visit_counts.keys()))
                counts = np.array(list(visit_counts.values()), dtype=np.float64)
                if trick < 2 and counts.sum() > 0 and len(cards) > 1:
                    chosen = int(rng.choice(cards, p=counts / counts.sum()))
                else:
                    chosen = int(cards[np.argmax(counts)])
                rec["card"] = chosen

                from manille_env.card import Card
                card = Card.from_index(chosen)
                pre = list(slot.game.current_trick)
                assert slot.game.play_card(pid, card), "actor played illegal card"
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
                    slot.started = False
                    still.append(slot)
            active = still
        flush()
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--belief-model", type=str, default=None,
                    help="separate checkpoint for determinization weighting "
                         "(defaults to --model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deals", type=int, default=10000)
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--leaf", choices=["rollout", "value"], default="value")
    ap.add_argument("--puct", action="store_true")
    ap.add_argument("--particles", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--exploration", type=float, default=0.7)
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--partner-belief", action="store_true",
                    help="Phase S: sample partner moves from the policy "
                         "(convention-forming generation)")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = dict(
        model=args.model, belief_model=args.belief_model,
        sims=args.sims, leaf=args.leaf, puct=args.puct,
        particles=args.particles, slots=args.slots,
        temperature=args.temperature, exploration=args.exploration,
        c_puct=args.c_puct, dirichlet_epsilon=args.dirichlet_epsilon,
        seed=args.seed, cuda=not args.no_cuda,
        partner_belief=args.partner_belief,
    )

    # Split deal indices across workers; each writes its own shard.
    idx_splits = [
        list(range(w, args.deals, args.workers)) for w in range(args.workers)
    ]
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
        total = sum(len(x) for x in idx_splits)
        while True:
            time.sleep(10)
            written = 0
            for w in range(args.workers):
                p = os.path.join(args.out, f"shard_{w}.bin")
                if os.path.exists(p) and os.path.getsize(p) > 24:
                    written += (os.path.getsize(p) - 24) // DEAL_DTYPE.itemsize
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

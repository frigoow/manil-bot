"""Expand declarer-only margin rows into all four seat perspectives.

`dmc_margin_gen` stores one row per rollout holding only the DECLARER's hand
and the margin signed from the declaring team. That serves D1 (the dealer
choosing trump or mull) and nothing else: a defender deciding whether to
meegaan sees their OWN hand, not the declarer's, so it needs
P(margin | my hand, declaration, I am defending) -- a different conditioning
on the same rollouts.

The other three hands were never written, but they are recoverable exactly.
The generator seeds a fresh rng per (hand index, declaration) and draws one
`permutation(rest)` per rollout in file order, so replaying that stream
reproduces the deal byte-for-byte -- no rollouts, no GPU, seconds instead of
hours. Each input row becomes four output rows, one per seat, carrying that
seat's hand, its role, and the margin signed from ITS team.

`rid` is the rollout id shared by a row's four seats. It is what makes the D3
conditioning possible later: given a D2 policy, join a declaring seat's row to
its defenders' rows in the same rollout and ask whether either would have
called meegaan.

Seat order within a rollout is preserved -- declarer first, then the other
three seats ascending -- so `rid` plus row position recovers the actual seat,
not just the role. The join above needs that.

  PYTHONPATH=. venv/bin/python -m learn.dmc_margin_expand \
      --data data/margin/round_001 --out data/margin/round_001_seats \
      --hands 5000 --rollouts 20 --seed 1
"""
import argparse
import os
import time

import numpy as np

from .dmc_margin_gen import ROW_DTYPE
from .evaluate import make_deals

# role: who the seat is relative to the declaration.
ROLE_DECLARER = 0
ROLE_PARTNER = 1   # declaring team, not the declarer
ROLE_DEFENDER = 2

# `rel` = (seat - declarer) % 4. It subsumes role (0 declarer, 2 partner,
# 1/3 defenders) and additionally says WHICH defender, i.e. whether this seat
# plays before or after the declarer -- positional information that is worth
# points and costs a byte.
SEAT_DTYPE = np.dtype([("hand", "<u8"), ("declaration", "u1"), ("role", "u1"),
                       ("rel", "u1"), ("margin", "i1"), ("rid", "<u4")])


def expand(rows, deals, n_hands, rollouts, seed):
    """One input row -> four seat rows. Returns (seat_rows, n_checked)."""
    expected = n_hands * 5 * rollouts
    if len(rows) != expected:
        raise ValueError(
            f"row count {len(rows)} != hands*5*rollouts = {expected}; the "
            "reconstruction indexes rows positionally and cannot proceed on "
            "a partial or double-written shard")

    out = np.empty(len(rows) * 4, dtype=SEAT_DTYPE)
    n = 0
    for hi in range(n_hands):
        hands, declarer = deals[hi]
        dec_mask = np.uint64(0)
        dec_set = set()
        for c in hands[declarer]:
            dec_mask |= np.uint64(1) << np.uint64(c.to_index())
            dec_set.add(c.to_index())
        rest = [c for c in range(32) if c not in dec_set]
        others = [s for s in range(4) if s != declarer]

        for declaration in range(5):
            rng = np.random.default_rng(
                (seed * 1000003 + hi * 41 + declaration * 7) % (2**63 - 1))
            base = (hi * 5 + declaration) * rollouts
            for j in range(rollouts):
                perm = rng.permutation(rest)
                row = rows[base + j]
                if row["hand"] != dec_mask or row["declaration"] != declaration:
                    raise ValueError(
                        f"stream desync at hand {hi} decl {declaration} "
                        f"rollout {j}: the stored row does not match the "
                        "replayed deal -- seed or row order is wrong")
                margin = int(row["margin"])
                rid = base + j

                out[n] = (dec_mask, declaration, ROLE_DECLARER, 0, margin, rid)
                n += 1
                for k, seat in enumerate(others):
                    mask = np.uint64(0)
                    for c in perm[k * 8:(k + 1) * 8]:
                        mask |= np.uint64(1) << np.uint64(int(c))
                    same_team = (seat % 2) == (declarer % 2)
                    out[n] = (mask, declaration,
                              ROLE_PARTNER if same_team else ROLE_DEFENDER,
                              (seat - declarer) % 4,
                              margin if same_team else -margin, rid)
                    n += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", type=int, required=True)
    ap.add_argument("--rollouts", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = np.fromfile(os.path.join(args.data, "shard_all.bin"),
                       dtype=ROW_DTYPE)
    print(f"read {len(rows)} declarer rows", flush=True)
    t0 = time.time()
    seats = expand(rows, make_deals(args.hands, args.seed),
                   args.hands, args.rollouts, args.seed)
    os.makedirs(args.out, exist_ok=True)
    seats.tofile(os.path.join(args.out, "shard_all.bin"))

    counts = np.bincount(seats["role"], minlength=3)
    print(f"wrote {len(seats)} seat rows in {time.time() - t0:.0f}s "
          f"(declarer {counts[0]}, partner {counts[1]}, "
          f"defender {counts[2]})", flush=True)


if __name__ == "__main__":
    main()

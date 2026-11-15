"""Replay shard reader and dataset for the Manille learning stack.

Shards are written by `manille_generate` (see cpp/generate.cpp for the exact
binary layout). Records are compact raw state (deal + per-ply actions and
visit counts); observation tensors are produced on demand by the C++ encoder
(`manille_ismcts.encode_trajectory`) so there is exactly one encoding
implementation.
"""
import glob
import os

import numpy as np

from .encoding import LEGAL_MASK_OFF

MAGIC = 0x494E414D  # 'MANI'
VERSION = 1
HEADER_DTYPE = np.dtype(
    [("magic", "<u4"), ("version", "<u4"), ("iters", "<u4"), ("pad", "<u4"),
     ("seed", "<u8")]
)
PLY_DTYPE = np.dtype(
    [("player", "u1"), ("card", "u1"), ("summary", "u1"), ("nlegal", "u1"),
     ("pad", "<u4"), ("visit_cards", "u1", (8,)), ("visit_counts", "<u4", (8,))]
)
DEAL_DTYPE = np.dtype(
    [("hands", "<i8", (4,)), ("trump", "i1"), ("declarer", "u1"),  # signed: -1 = mull
     ("pts0", "u1"), ("pts1", "u1"), ("pad", "<u4"), ("plies", PLY_DTYPE, (32,))]
)

assert PLY_DTYPE.itemsize == 48
assert DEAL_DTYPE.itemsize == 1576


def load_deals(directory):
    """Read all shards in `directory` into one structured array of deals."""
    shards = sorted(glob.glob(os.path.join(directory, "shard_*.bin")))
    if not shards:
        raise FileNotFoundError(f"no shard files in {directory}")
    parts = []
    for path in shards:
        raw = np.fromfile(path, dtype=np.uint8)
        header = np.frombuffer(raw[: HEADER_DTYPE.itemsize], dtype=HEADER_DTYPE)
        if int(header["magic"][0]) != MAGIC:
            raise ValueError(f"{path}: bad magic")
        if int(header["version"][0]) != VERSION:
            raise ValueError(f"{path}: unsupported version {int(header['version'][0])}")
        body = raw[HEADER_DTYPE.itemsize:]
        if body.size % DEAL_DTYPE.itemsize != 0:
            raise ValueError(f"{path}: truncated shard")
        parts.append(np.frombuffer(body, dtype=DEAL_DTYPE))
    return np.concatenate(parts)


def _card_holder(hands):
    """holder[c] = absolute seat holding card c in the true deal."""
    holder = np.full(32, -1, dtype=np.int64)
    for p in range(4):
        mask = int(hands[p])
        for c in range(32):
            if mask & (1 << c):
                holder[c] = p
    return holder


def deal_to_training_arrays(deal, obs, players):
    """Derive per-ply training targets for one deal.

    Args:
        deal: one element of the structured array from load_deals.
        obs: float32 [32, OBS_DIM] from manille_ismcts.encode_trajectory.
        players: int32 [32] from the same call (observer per ply).

    Returns dict of numpy arrays:
        obs        float32 [32, OBS_DIM]
        policy     float32 [32, 32]  visit-count distribution over cards
        value      float32 [32]      (team points - 30) / 30 from the
                                     observer's team perspective
        belief     float32 [32, 3, 32] one-hot true holder of each unseen card
                   (slots: hidden seats next/partner/prev relative to observer)
        trick      int64 [32]        trick index 0..7
        legal      float32 [32, 32]  legal mask (sliced out of obs)
    """
    import manille_ismcts as mi  # local import: keeps shard parsing module-free

    actions = deal["plies"]["card"].astype(np.int64)
    n_legal = deal["plies"]["nlegal"].astype(np.int64)
    holder = _card_holder(deal["hands"])

    policy = np.zeros((32, 32), dtype=np.float32)
    value = np.zeros(32, dtype=np.float32)
    belief = np.zeros((32, 3, 32), dtype=np.float32)
    trick = np.arange(32, dtype=np.int64) // 4

    played_mask = 0
    for t in range(32):
        obs_seat = int(players[t])

        counts = deal["plies"]["visit_counts"][t].astype(np.float32)
        cards = deal["plies"]["visit_cards"][t].astype(np.int64)
        total = counts.sum()
        if total > 0:
            filled = counts > 0  # zero-padded slots have card 0, count 0
            policy[t, cards[filled]] = counts[filled] / total

        team = obs_seat % 2
        pts = int(deal["pts0"]) if team == 0 else int(deal["pts1"])
        value[t] = (pts - 30.0) / 30.0

        own_mask = int(deal["hands"][obs_seat]) & ~played_mask
        seen = own_mask | played_mask
        for c in range(32):
            if seen & (1 << c):
                continue
            rel = (int(holder[c]) - obs_seat) % 4
            assert rel != 0, "unseen card held by observer: replay corrupted"
            belief[t, rel - 1, c] = 1.0

        played_mask |= 1 << int(actions[t])

    legal = obs[:, LEGAL_MASK_OFF : LEGAL_MASK_OFF + 32].copy()
    # Sanity: the stored legal-set size must match the encoded mask.
    assert (legal.sum(axis=1).astype(np.int64) == n_legal).all()

    return {
        "obs": obs,
        "policy": policy,
        "value": value,
        "belief": belief,
        "trick": trick,
        "legal": legal,
    }


def encode_deal(deal):
    """Run the C++ trajectory encoder for one deal; returns (obs, players)."""
    import manille_ismcts as mi

    out = mi.encode_trajectory(
        np.ascontiguousarray(deal["hands"]),
        int(deal["trump"]),
        int(deal["declarer"]),
        np.ascontiguousarray(deal["plies"]["card"].astype(np.int32)),
    )
    assert int(out["pts0"]) == int(deal["pts0"])
    assert int(out["pts1"]) == int(deal["pts1"])
    return out["obs"], out["players"]


class DealDataset:
    """Torch-free dataset (avoids multiprocessing pickling issues with the
    extension module): index -> dict of numpy arrays for one deal (32 plies).
    The training loop shuffles deal order and flattens plies into batches."""

    def __init__(self, deals):
        self.deals = deals

    def __len__(self):
        return len(self.deals)

    def __getitem__(self, idx):
        deal = self.deals[idx]
        obs, players = encode_deal(deal)
        return deal_to_training_arrays(deal, obs, players)


def collate(deal_arrays):
    """Concatenate per-deal array dicts along the ply axis."""
    return {
        k: np.concatenate([d[k] for d in deal_arrays], axis=0)
        for k in deal_arrays[0]
    }


def iter_ply_batches(dataset, order, batch_plies=1024, workers=8):
    """Yield collated batches of ~batch_plies plies, in order.

    Encoding runs in a thread pool — encode_trajectory releases the GIL, so
    this scales with cores and keeps the GPU fed. In-flight work is bounded
    to 2*workers batches: Executor.map() submits every group up front and
    retains results until consumed in order, which at 200k-deal scale
    buffers ~65 GB of encoded batches and OOMs the machine.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    deals_per_batch = max(1, batch_plies // 32)
    groups = [
        order[i : i + deals_per_batch]
        for i in range(0, len(order), deals_per_batch)
    ]

    def load_group(idxs):
        return collate([dataset[i] for i in idxs])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        gi = iter(groups)
        for _ in range(2 * workers):
            try:
                pending.append(pool.submit(load_group, next(gi)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(load_group, next(gi)))
            except StopIteration:
                pass

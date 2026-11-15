"""DMC replay shard reader and dataset.

Shards are written by `learn.dmc_selfplay` (VERSION=2). Records are compact
raw state (deal + per-ply chosen actions and the ε used); observation and
candidate-action tensors are produced on demand by the C++ encoder
(`manille_ismcts.encode_q_trajectory`) so there is exactly one encoding
implementation.

Per-ply training targets:
  - Q: the final team points of the deciding player's team, normalized
    (points - 30) / 30 — full-episode Monte Carlo return, no bootstrapping.
  - belief: one-hot true holder of each unseen card (same derivation as
    replay.py, from the recorded true hands).
"""
import glob
import itertools
import os

import numpy as np

from manille_env.card import permute_card_index, permute_hand_mask

from .dmc_tokens import (GLOBAL_DIM, MAX_HAND, MAX_PLAYS, RAW_TOKEN_DIM,
                         global_from_obs, hand_token_ids, tokens_from_obs)
from .encoding import ACT_DIM, MAX_LEGAL, OBS_DIM

MAGIC = 0x494E414D  # 'MANI'
VERSION = 2
HEADER_DTYPE = np.dtype(
    [("magic", "<u4"), ("version", "<u4"), ("iters", "<u4"), ("pad", "<u4"),
     ("seed", "<u8")]
)
DMC_PLY_DTYPE = np.dtype(
    [("player", "u1"), ("card", "u1"), ("summary", "u1"), ("nlegal", "u1"),
     ("eps", "<f4")]
)
# trump is SIGNED: -1 is mull (no trump), 0-3 are the suits. i1 is
# byte-identical to the u1 this used to be, and 0-3 read the same under
# both, so existing shards stay valid and no VERSION bump is needed --
# but written as u1 a mull deal would silently read back as trump 255.
DMC_DEAL_DTYPE = np.dtype(
    [("hands", "<i8", (4,)), ("trump", "i1"), ("declarer", "u1"),
     ("pts0", "u1"), ("pts1", "u1"), ("pad", "<u4"),
     ("plies", DMC_PLY_DTYPE, (32,))]
)

assert DMC_PLY_DTYPE.itemsize == 8
assert DMC_DEAL_DTYPE.itemsize == 296

# VERSION=3: SPARTA-distillation shards (written by learn.dmc_distill). Same
# raw deal record plus per-ply teacher targets: for each labelled legal arm
# (sq_n of them; 0 = no label at this ply, e.g. forced moves), the candidate
# card, the teacher rollout value (raw team points, 0..60) and its sample
# count. SPARTA/UCB counts may differ; true-world labels have count one.
VERSION_DISTILL = 3
DISTILL_PLY_DTYPE = np.dtype(
    [("player", "u1"), ("card", "u1"), ("summary", "u1"), ("nlegal", "u1"),
     ("eps", "<f4"), ("sq_n", "u1"), ("sq_pad", "u1"),
     ("sq_cards", "u1", (8,)), ("sq_vals", "<f4", (8,)),
     ("sq_cnt", "<u2", (8,))]
)
DISTILL_DEAL_DTYPE = np.dtype(
    [("hands", "<i8", (4,)), ("trump", "i1"), ("declarer", "u1"),  # signed: -1 = mull
     ("pts0", "u1"), ("pts1", "u1"), ("pad", "<u4"),
     ("plies", DISTILL_PLY_DTYPE, (32,))]
)

assert DISTILL_PLY_DTYPE.itemsize == 66
assert DISTILL_DEAL_DTYPE.itemsize == 2152


def write_shard_header(path, seed, version=VERSION):
    header = np.zeros((), dtype=HEADER_DTYPE)
    header["magic"] = MAGIC
    header["version"] = version
    header["iters"] = 0
    header["seed"] = seed
    with open(path, "ab") as f:
        if os.path.getsize(path) == 0:
            f.write(header.tobytes())


def _load_shards(directories, version, dtype):
    if isinstance(directories, str):
        directories = [directories]
    parts = []
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, "shard_*.bin"))):
            raw = np.fromfile(path, dtype=np.uint8)
            header = np.frombuffer(
                raw[: HEADER_DTYPE.itemsize], dtype=HEADER_DTYPE)
            if int(header["magic"][0]) != MAGIC:
                raise ValueError(f"{path}: bad magic")
            if int(header["version"][0]) != version:
                raise ValueError(
                    f"{path}: unsupported version {int(header['version'][0])}")
            body = raw[HEADER_DTYPE.itemsize:]
            if body.size % dtype.itemsize != 0:
                raise ValueError(f"{path}: truncated shard")
            parts.append(np.frombuffer(body, dtype=dtype))
    if not parts:
        raise FileNotFoundError(f"no shards in {directories}")
    return np.concatenate(parts)


def load_dmc_deals(directories):
    """Read all DMC shards under one or more cycle directories into one
    structured array. `directories` may be a single path or a list."""
    return _load_shards(directories, VERSION, DMC_DEAL_DTYPE)


def load_distill_deals(directories):
    """Read VERSION=3 SPARTA-distillation shards (learn.dmc_distill)."""
    return _load_shards(directories, VERSION_DISTILL, DISTILL_DEAL_DTYPE)


def _card_holder(hands):
    """holder[c] = absolute seat holding card c in the true deal."""
    holder = np.full(32, -1, dtype=np.int64)
    for p in range(4):
        mask = int(hands[p])
        for c in range(32):
            if mask & (1 << c):
                holder[c] = p
    return holder


def encode_dmc_deal(deal):
    """Run the C++ Q-trajectory encoder for one deal."""
    import manille_ismcts as mi

    out = mi.encode_q_trajectory(
        np.ascontiguousarray(deal["hands"]),
        int(deal["trump"]),
        int(deal["declarer"]),
        np.ascontiguousarray(deal["plies"]["card"].astype(np.int32)),
    )
    assert int(out["pts0"]) == int(deal["pts0"])
    assert int(out["pts1"]) == int(deal["pts1"])
    return out


def deal_to_dmc_arrays(deal, traj):
    """Derive per-ply DMC training targets for one deal.

    Args:
        deal: one element of the structured array from load_dmc_deals.
        traj: dict from encode_dmc_deal.

    Returns dict of numpy arrays:
        obs      float32 [32, OBS_DIM]
        act      float32 [32, ACT_DIM]   candidate features of the played card
        ret      float32 [32]            final team points of the deciding
                                         player's team, (pts - 30) / 30
        belief   float32 [32, 3, 32]     one-hot true holder per unseen card
        trick    int64 [32]
    """
    actions = deal["plies"]["card"].astype(np.int64)
    holder = _card_holder(deal["hands"])

    act = np.zeros((32, ACT_DIM), dtype=np.float32)
    ret = np.zeros(32, dtype=np.float32)
    belief = np.zeros((32, 3, 32), dtype=np.float32)
    trick = np.arange(32, dtype=np.int64) // 4
    raw = np.zeros((32, MAX_PLAYS, RAW_TOKEN_DIM), dtype=np.float32)
    n_raw = np.zeros(32, dtype=np.int64)
    glob_tok = np.zeros((32, GLOBAL_DIM), dtype=np.float32)
    hand = np.full((32, MAX_HAND), 32, dtype=np.int64)

    played_mask = 0
    for t in range(32):
        seat = int(traj["players"][t])
        # The played card's row in the candidate matrix.
        n = int(traj["n_legal"][t])
        row = -1
        for j in range(n):
            if int(traj["legal"][t][j]) == int(actions[t]):
                row = j
                break
        assert row >= 0, f"ply {t}: played card not among legal candidates"
        act[t] = traj["act"][t][row]

        team = seat % 2
        pts = int(deal["pts0"]) if team == 0 else int(deal["pts1"])
        ret[t] = (pts - 30.0) / 30.0

        own_mask = int(deal["hands"][seat]) & ~played_mask
        seen = own_mask | played_mask
        for c in range(32):
            if seen & (1 << c):
                continue
            rel = (int(holder[c]) - seat) % 4
            assert rel != 0, "unseen card held by observer: replay corrupted"
            belief[t, rel - 1, c] = 1.0

        # Transformer tokens (Architecture B): the public C++ history ply
        # records, the global token, and the observer's current hand.
        raw[t], n_raw[t] = tokens_from_obs(traj["obs"][t], t)
        glob_tok[t] = global_from_obs(traj["obs"][t])
        hand[t] = hand_token_ids(own_mask)

        played_mask |= 1 << int(actions[t])

    return {
        "obs": traj["obs"],
        "act": act,
        "ret": ret,
        "belief": belief,
        "trick": trick,
        "raw": raw,
        "n_raw": n_raw,
        "glob": glob_tok,
        "hand": hand,
        # Legal-move count per ply. This is PUBLIC (it is |legal| for the
        # acting seat, not the true-hand-derived legal_summary nibble that
        # contaminated D4), and the trainer needs it to tell a real decision
        # from a forced card: 38.5% of plies have exactly one legal move and
        # teach nothing about ranking, trick 7 being 100% forced.
        "nlegal": traj["n_legal"].astype(np.int64),
        # The ε this ply was played under. Stored in every shard since
        # VERSION=2 and then dropped here, which quietly blocked any
        # continuation-aware weighting: a DMC target is the return under an
        # ε-soft continuation, but deployment is greedy, and the ladder
        # spans 0.30 down to 0.005. Exposing it costs nothing and makes
        # low-ε reweighting possible on shards that already exist.
        "eps": deal["plies"]["eps"].astype(np.float32),
        # Absolute acting seat. A best-response run plays seats 1 & 3 with
        # a frozen checkpoint in every deal; the learner must fit only its
        # own team's decisions (dmc_train --learner-team), and the team is
        # seat % 2.
        "seat": traj["players"].astype(np.int64),
    }


def permute_deal(deal, perm):
    """Relabel the suits of one stored deal: hands, trump and the played
    sequence together.

    This is exact data augmentation, not regularization. Relabelling is an
    automorphism of the game, so the permuted deal is a real deal that was
    really played -- and its MC return (final team points) is literally the
    same number, so the label needs no adjustment. The encoder is untouched:
    the permuted deal goes through the same `encode_q_trajectory` as any
    other, which keeps this clear of the C++/Python mirror parity discipline.
    """
    out = np.zeros((), dtype=DMC_DEAL_DTYPE)
    out[()] = deal
    out["hands"] = [permute_hand_mask(int(h), perm) for h in deal["hands"]]
    if int(deal["trump"]) >= 0:
        out["trump"] = perm[int(deal["trump"])]
    out["plies"]["card"] = [permute_card_index(int(c), perm)
                            for c in deal["plies"]["card"]]
    return out


def augmentation_perms(deal):
    """Suit permutations that leave this deal's TRUMP SELECTION intact.

    Trump is not a free label: these shards were generated with
    `apply_trump_rule`, and the rule breaks ties on the lowest suit index,
    so a permutation can move a tied suit below trump and change what the
    rule would have picked. Training on a deal whose trump the rule would
    never choose invents states the net never meets at deployment, so the
    engine's own rule is asked (CLAUDE.md: the engine is authoritative)
    rather than assumed. Mull deals have no trump and admit all 24.
    """
    import manille_ismcts as mi

    trump = int(deal["trump"])
    declarer_hand = int(deal["hands"][int(deal["declarer"])])
    keep = []
    for perm in itertools.permutations(range(4)):
        if trump < 0:
            keep.append(perm)
            continue
        if perm[trump] != trump:
            continue
        if mi.apply_trump_rule(permute_hand_mask(declarer_hand, perm)) == trump:
            keep.append(perm)
    return keep or [(0, 1, 2, 3)]


class DMCDataset:
    """Torch-free dataset (pickling-safe): index -> dict of numpy arrays for
    one deal (32 plies). The training loop shuffles deal order and flattens
    plies into batches.

    `augment` draws a suit relabelling per deal per epoch. The net is not
    suit-invariant even though the game is -- it changes its argmax on 21.5%
    of non-forced plies under a pure relabelling, and averaging over the
    relabellings at test time is worth +0.55 [+0.40, +0.69] pts/deal on the
    MLP blueprint (scripts/suit_symmetry_gate.py). This is the training-time
    route to the same invariance, at 1x inference cost instead of 6x.

    `aug_seed` is mixed with the deal index rather than drawn from shared
    state: `iter_ply_batches` encodes in a thread pool, so a shared RNG would
    make the epoch depend on thread scheduling. Bump it per epoch.
    """

    def __init__(self, deals, augment=False, aug_seed=0,
                 paired_original=False, max_eps=None):
        if paired_original and not augment:
            raise ValueError("paired_original requires augmentation")
        if max_eps is not None:
            if not np.isfinite(max_eps) or max_eps < 0:
                raise ValueError("max_eps must be a finite non-negative value")
            # ε is drawn once per deal in self-play and recorded on every
            # ply. Filtering at deal granularity keeps full trajectories and
            # their returns intact; using ply 0 also makes the contract
            # explicit rather than silently assuming all 32 records agree.
            # MAX over plies, not ply 0. A league deal records eps 0 on
            # the frozen-opponent seats, so reading the first ply lets a
            # 0.30 deal through whenever its opening card came from one
            # of those seats: measured 8.4% of the retained set, and it
            # also skewed the league share 19.9% -> 26.5%, which would
            # have made the arms differ by more than the epsilon.
            deals = deals[deals["plies"]["eps"].max(axis=1) <= max_eps]
        self.deals = deals
        self.augment = augment
        self.aug_seed = aug_seed
        self.paired_original = paired_original
        self.max_eps = max_eps

    def __len__(self):
        return len(self.deals)

    def __getitem__(self, idx):
        deal = self.deals[idx]
        original = None
        if self.augment:
            if self.paired_original:
                original = deal_to_dmc_arrays(deal, encode_dmc_deal(deal))
            perms = augmentation_perms(deal)
            rng = np.random.default_rng((self.aug_seed, idx))
            deal = permute_deal(deal, perms[rng.integers(len(perms))])
        arrays = deal_to_dmc_arrays(deal, encode_dmc_deal(deal))
        if original is not None:
            # Only the Q inputs are needed for the paired invariance loss.
            # Labels and ply ordering are identical under the automorphism.
            for key in ("obs", "act", "raw", "n_raw", "glob", "hand"):
                arrays["orig_" + key] = original[key]
        return arrays


def collate(deal_arrays):
    """Concatenate per-deal array dicts along the ply axis."""
    return {
        k: np.concatenate([d[k] for d in deal_arrays], axis=0)
        for k in deal_arrays[0]
    }


def iter_ply_batches(dataset, order, batch_plies=2048, workers=8):
    """Yield collated batches of ~batch_plies plies, in order.

    Encoding runs in a thread pool — encode_q_trajectory releases the GIL.
    In-flight work is bounded to 2*workers batches (the 65 GB OOM fix from
    replay.py carries over).
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


def deal_to_distill_arrays(deal, traj):
    """Derive per-ply distillation targets for one VERSION=3 deal.

    Args:
        deal: one element of the structured array from load_distill_deals.
        traj: dict from encode_dmc_deal.

    Returns dict of numpy arrays:
        obs        float32 [32, OBS_DIM]
        cand_act   float32 [32, MAX_LEGAL, ACT_DIM]  all candidate rows
        sq_val     float32 [32, MAX_LEGAL]  teacher values, (pts - 30) / 30
        sq_mask    bool    [32, MAX_LEGAL]  searched arms (the loss mask)
        sq_cnt     float32 [32, MAX_LEGAL]  teacher rollout counts per arm
        played_row int64   [32]             candidate row of the played card
        ret        float32 [32]             MC return (mix option, see --c-ret)
        trick      int64   [32]
    """
    actions = deal["plies"]["card"].astype(np.int64)
    plies = deal["plies"]

    cand_act = np.zeros((32, MAX_LEGAL, ACT_DIM), dtype=np.float32)
    sq_val = np.zeros((32, MAX_LEGAL), dtype=np.float32)
    sq_mask = np.zeros((32, MAX_LEGAL), dtype=bool)
    sq_cnt = np.zeros((32, MAX_LEGAL), dtype=np.float32)
    played_row = np.zeros(32, dtype=np.int64)
    ret = np.zeros(32, dtype=np.float32)
    trick = np.arange(32, dtype=np.int64) // 4
    raw = np.zeros((32, MAX_PLAYS, RAW_TOKEN_DIM), dtype=np.float32)
    n_raw = np.zeros(32, dtype=np.int64)
    glob_tok = np.zeros((32, GLOBAL_DIM), dtype=np.float32)

    for t in range(32):
        seat = int(traj["players"][t])
        n = int(traj["n_legal"][t])
        legal = list(traj["legal"][t][:n])
        cand_act[t, :n] = traj["act"][t][:n]
        raw[t], n_raw[t] = tokens_from_obs(traj["obs"][t], t)
        glob_tok[t] = global_from_obs(traj["obs"][t])

        row = -1
        for j in range(n):
            if int(legal[j]) == int(actions[t]):
                row = j
                break
        assert row >= 0, f"ply {t}: played card not among legal candidates"
        played_row[t] = row

        team = seat % 2
        pts = int(deal["pts0"]) if team == 0 else int(deal["pts1"])
        ret[t] = (pts - 30.0) / 30.0

        for j in range(int(plies["sq_n"][t])):
            card = int(plies["sq_cards"][t][j])
            assert card in legal, \
                f"ply {t}: teacher card {card} not legal in replay"
            r = legal.index(card)
            sq_val[t, r] = (float(plies["sq_vals"][t][j]) - 30.0) / 30.0
            sq_mask[t, r] = True
            sq_cnt[t, r] = float(plies["sq_cnt"][t][j])

    return {
        "obs": traj["obs"],
        "cand_act": cand_act,
        "sq_val": sq_val,
        "sq_mask": sq_mask,
        "sq_cnt": sq_cnt,
        "played_row": played_row,
        "ret": ret,
        "trick": trick,
        "raw": raw,
        "n_raw": n_raw,
        "glob": glob_tok,
    }


class DistillDataset:
    """Torch-free dataset over VERSION=3 deals (pickling-safe), same batching
    contract as DMCDataset."""

    def __init__(self, deals):
        self.deals = deals

    def __len__(self):
        return len(self.deals)

    def __getitem__(self, idx):
        deal = self.deals[idx]
        return deal_to_distill_arrays(deal, encode_dmc_deal(deal))

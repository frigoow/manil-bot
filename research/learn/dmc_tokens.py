"""Token construction for the DMC transformer (Architecture B, plan §4).

One canonical builder used by training (from shard re-encoding) and by live
play (self-play workers and the eval agent), so both sides always see
identical features. Every token derives from the C++ observation (parity-
tested) — no new mirroring, and nothing leaks hidden information.

IMPORTANT (why the raw summary nibble is not used): the shard's
could-trump / could-overtrump / nlegal fields are computed from true hands
and are only partially public (a player who follows suit may or may not
hold trumps; nlegal reveals suit counts when following is forced). The C++
encoder's history block is the strictly-public record — its 10 flags
already encode every legal-context fact an observer can derive (including
"couldn't beat" and "couldn't overtrump" where the rules force them).

Play tokens (RAW_TOKEN_DIM=58): the C++ history ply record — relative
actor (4), card (32), led suit (4), position in trick (4), winning-so-far
seat (4), 10 public context flags.

Global token (GLOBAL_DIM=130): own hand (32), trump (4), trick number (8),
team points (2), declarer (4), tactical block (80) — sliced from the same
observation.

Hand tokens: up to 8 card ids of the observer's current hand (-1 padded);
the model embeds them. Candidate token: the ACT_DIM action features from
the C++ encoder.
"""
import numpy as np

from .encoding import (ACT_DIM, BELIEF_MASK_OFF, BELIEF_PRIOR_OFF,
                       DECLARER_OFF, HISTORY_OFF, MASTER_OFF, OWN_HAND_OFF,
                       PLY_DIM, TACTICAL_DIM, TEAM_PTS_OFF, TRICK_NUM_OFF,
                       TRUMP_OFF, VOID_OFF)

RAW_TOKEN_DIM = PLY_DIM  # 58 — the C++ history ply record
GLOBAL_DIM = 50 + TACTICAL_DIM  # 130
LEGACY_GLOBAL_DIM = 50  # pre-1.3 global token
MAX_PLAYS = 31
MAX_HAND = 8
PAD_CARD = 32  # embedding index for padded hand slots


def tokens_from_obs(obs, n_plays):
    """Raw play tokens from the observation's history block. Returns
    (raw [MAX_PLAYS, RAW_TOKEN_DIM], n)."""
    raw = obs[HISTORY_OFF : HISTORY_OFF + MAX_PLAYS * PLY_DIM].reshape(
        MAX_PLAYS, PLY_DIM)
    return np.ascontiguousarray(raw), min(n_plays, MAX_PLAYS)


def global_from_obs(obs):
    """The global token, sliced from the C++ observation."""
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0:32] = obs[OWN_HAND_OFF : OWN_HAND_OFF + 32]
    g[32:36] = obs[TRUMP_OFF : TRUMP_OFF + 4]
    g[36:44] = obs[TRICK_NUM_OFF : TRICK_NUM_OFF + 8]
    g[44:46] = obs[TEAM_PTS_OFF : TEAM_PTS_OFF + 2]
    g[46:50] = obs[DECLARER_OFF : DECLARER_OFF + 4]
    # Tactical block (plan 1.3). The GRU sees only glob + history tokens, so
    # anything left out here is invisible to it however rich the observation.
    g[50:GLOBAL_DIM] = obs[MASTER_OFF : MASTER_OFF + TACTICAL_DIM]
    return g


def hand_token_ids(hand_mask):
    """Card ids of a hand bitmask, PAD_CARD-padded to MAX_HAND."""
    ids = [c for c in range(32) if hand_mask & (1 << c)]
    ids = ids[:MAX_HAND]
    ids += [PAD_CARD] * (MAX_HAND - len(ids))
    return np.array(ids, dtype=np.int64)


BELIEF_CTX_DIM = GLOBAL_DIM + 16 + 96 + 96  # 258


def belief_context_from_obs(obs):
    """Belief-net context for the recurrent belief arch: the global token
    plus the voids block and the belief prior/mask blocks — the same
    conditioning the flat BeliefNet sees inside OBS (first probe's GRU
    lacked these and was input-handicapped)."""
    g = global_from_obs(obs)
    return np.concatenate([
        g,
        obs[VOID_OFF : VOID_OFF + 16],
        obs[BELIEF_PRIOR_OFF : BELIEF_PRIOR_OFF + 96],
        obs[BELIEF_MASK_OFF : BELIEF_MASK_OFF + 96],
    ]).astype(np.float32)


def encode_live(obs, n_plays, observer_hand_mask):
    """Live-play tokens for one decision, from the C++ observation."""
    raw, n = tokens_from_obs(obs, n_plays)
    return {
        "raw": raw,
        "n_raw": n,
        "glob": global_from_obs(obs),
        "hand": hand_token_ids(observer_hand_mask),
    }


def tokens_batch(obs_rows, n_plays_rows, own_masks=None):
    """Batched token tensors for token models (QTransformer/QGRU),
    equivalent to stacking encode_live per row but built by fixed-offset
    slicing over the stacked obs matrix — no per-row Python. own_masks are
    int bitmasks of each row observer's current hand; pass None to skip the
    hand tokens (QGRU never reads them)."""
    obs = np.ascontiguousarray(obs_rows)
    B = obs.shape[0]
    raw = np.ascontiguousarray(
        obs[:, HISTORY_OFF : HISTORY_OFF + MAX_PLAYS * PLY_DIM]
    ).reshape(B, MAX_PLAYS, PLY_DIM)
    n_raw = np.minimum(np.asarray(n_plays_rows, dtype=np.int64), MAX_PLAYS)
    glob = np.empty((B, GLOBAL_DIM), dtype=np.float32)
    glob[:, 0:32] = obs[:, OWN_HAND_OFF : OWN_HAND_OFF + 32]
    glob[:, 32:36] = obs[:, TRUMP_OFF : TRUMP_OFF + 4]
    glob[:, 36:44] = obs[:, TRICK_NUM_OFF : TRICK_NUM_OFF + 8]
    glob[:, 44:46] = obs[:, TEAM_PTS_OFF : TEAM_PTS_OFF + 2]
    glob[:, 46:50] = obs[:, DECLARER_OFF : DECLARER_OFF + 4]
    # Tactical block. `glob` is np.empty, so every column MUST be written:
    # omitting this slice fed uninitialized memory to the net on the batched
    # path (training and search both use it) while the per-row builder was
    # correct, which is the kind of gap only a batch-vs-row parity test finds.
    glob[:, 50:GLOBAL_DIM] = obs[:, MASTER_OFF : MASTER_OFF + TACTICAL_DIM]
    hand = None
    if own_masks is not None:
        bits = (np.asarray(own_masks, dtype=np.int64)[:, None]
                >> np.arange(32)[None, :]) & 1
        r, c = np.nonzero(bits)
        slot = np.arange(len(r)) - np.searchsorted(r, np.arange(B))[r]
        keep = slot < MAX_HAND
        hand = np.full((B, MAX_HAND), PAD_CARD, dtype=np.int64)
        hand[r[keep], slot[keep]] = c[keep]
    return raw, n_raw, glob, hand

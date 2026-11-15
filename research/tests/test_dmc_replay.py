"""DMC shard round-trip, target derivation, and generation integration."""
import os

import numpy as np

from learn.dmc_tokens import GLOBAL_DIM
from learn.encoding import ACT_DIM
import pytest
import torch

from learn.dmc_model import QNet, save_checkpoint
from learn.dmc_replay import (DMC_DEAL_DTYPE, DMCDataset, deal_to_dmc_arrays,
                              encode_dmc_deal, load_dmc_deals)
from learn.dmc_selfplay import _play_chunk
from manille_env.game_logic import GamePhase, ManilleGame


def _generate(tmp_path, n_deals=12, seed=7):
    """Run the real generation worker (CPU) on a random-weight Q net."""
    torch.manual_seed(0)
    model_path = os.path.join(tmp_path, "q_init.pt")
    net = QNet()
    save_checkpoint(model_path, net, net.state_dict(), None, None, 0,
                    {"hidden": 1024, "blocks": 4})
    cfg = {"model": model_path, "slots": 4, "seed": seed, "cuda": False,
           "league": [], "league_frac": 0.2}
    shard = os.path.join(tmp_path, "cycle_000", "shard_0.bin")
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    n = _play_chunk(cfg, list(range(n_deals)), shard)
    assert n == n_deals
    return os.path.join(tmp_path, "cycle_000")


def test_mull_shards_roundtrip_with_signed_trump(tmp_path):
    """A no-trump lineage must survive the shard format.

    The trump field used to be "u1" (unsigned), so a mull deal's -1 was
    written as 255 and read back as 255 -- every mull shard silently
    corrupt, with nothing failing loudly. It is "i1" now; i1 and u1 are
    byte-identical and 0-3 read the same under both, so existing shards
    stay valid and no VERSION bump was needed.
    """
    torch.manual_seed(0)
    model_path = os.path.join(tmp_path, "q_init.pt")
    net = QNet()
    save_checkpoint(model_path, net, net.state_dict(), None, None, 0,
                    {"hidden": 1024, "blocks": 4})
    cfg = {"model": model_path, "slots": 4, "seed": 11, "cuda": False,
           "league": [], "league_frac": 0.2, "mull": True}
    shard = os.path.join(tmp_path, "cycle_000", "shard_0.bin")
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    assert _play_chunk(cfg, list(range(8)), shard) == 8

    deals = load_dmc_deals(os.path.join(tmp_path, "cycle_000"))
    assert len(deals) == 8
    for deal in deals:
        assert int(deal["trump"]) == -1, "mull must round-trip as -1, not 255"
        # The deal really was played at no trump, and the recorded points
        # are reproducible through the encoder.
        assert int(deal["pts0"]) + int(deal["pts1"]) == 60
        encode_dmc_deal(deal)


def test_shard_roundtrip_and_targets(tmp_path):
    data_dir = _generate(tmp_path)
    deals = load_dmc_deals(data_dir)
    assert len(deals) == 12

    deal = deals[0]
    # Replay the recorded actions through the Python engine: every play must
    # be legal and the final points must match the record.
    from manille_env.card import Card, Suit
    game = ManilleGame(dealer=int(deal["declarer"]))
    game.hands = [
        [Card.from_index(c) for c in range(32) if int(deal["hands"][p]) & (1 << c)]
        for p in range(4)
    ]
    game.select_trump(Suit(int(deal["trump"])))
    game.start_playing()
    for t in range(32):
        ply = deal["plies"][t]
        assert int(ply["player"]) == game.current_player
        assert game.play_card(game.current_player,
                              Card.from_index(int(ply["card"])))
    assert game.phase != GamePhase.PLAYING
    assert int(deal["pts0"]) == game.hand_points[0]
    assert int(deal["pts1"]) == game.hand_points[1]

    # Targets: return is the deciding team's final points; belief one-hots
    # cover every unseen card exactly once and never point at the observer.
    arrays = deal_to_dmc_arrays(deal, encode_dmc_deal(deal))
    for t in range(32):
        seat = int(deal["plies"][t]["player"])
        pts = int(deal["pts0"]) if seat % 2 == 0 else int(deal["pts1"])
        assert arrays["ret"][t] == pytest.approx((pts - 30.0) / 30.0)
        b = arrays["belief"][t]
        assert (b.sum(axis=0) <= 1.0).all()
        n_unseen = 32 - 8 - t  # own hand minus... own hand unseen? no:
        # unseen = cards not in the observer's remaining hand and not played
        played_mask = 0
        for k in range(t):
            played_mask |= 1 << int(deal["plies"][k]["card"])
        own = int(deal["hands"][seat]) & ~played_mask
        n_unseen = 32 - bin(own).count("1") - bin(played_mask).count("1")
        assert b.sum() == pytest.approx(n_unseen)
    assert arrays["act"].shape == (32, ACT_DIM)
    # The act row must be the candidate vector of the played card: one-hot
    # matches the recorded card.
    for t in range(32):
        assert int(np.argmax(arrays["act"][t][:32])) == int(
            deal["plies"][t]["card"])


def test_dataset_and_batches(tmp_path):
    data_dir = _generate(tmp_path, n_deals=8)
    dataset = DMCDataset(load_dmc_deals(data_dir))
    assert len(dataset) == 8
    item = dataset[0]
    assert set(item) == {"obs", "act", "ret", "belief", "trick",
                       "raw", "n_raw", "glob", "hand", "nlegal",
                       "eps", "seat"}
    assert item["raw"].shape == (32, 31, 58)
    assert item["glob"].shape == (32, GLOBAL_DIM)
    assert item["hand"].shape == (32, 8)
    assert (item["n_raw"] == np.arange(32)).all()

    from learn.dmc_replay import iter_ply_batches
    batches = list(iter_ply_batches(dataset, np.arange(8), batch_plies=64,
                                    workers=2))
    assert sum(b["obs"].shape[0] for b in batches) == 8 * 32


def test_dataset_max_eps_filters_complete_deals(tmp_path):
    """The low-ε probe must filter whole deals, not selected plies."""
    deals = load_dmc_deals(_generate(tmp_path, n_deals=8))
    # Make the contract visible even if generation's ladder changes: the
    # first four are low ε and the rest are intentionally above the cutoff.
    deals["plies"]["eps"][:4] = 0.02
    deals["plies"]["eps"][4:] = 0.10

    low = DMCDataset(deals, max_eps=0.04)
    assert len(low) == 4
    assert (low.deals["plies"]["eps"][:, 0] <= 0.04).all()
    # Each selected record remains a normal 32-ply trajectory.
    assert low[0]["obs"].shape[0] == 32

    with pytest.raises(ValueError, match="non-negative"):
        DMCDataset(deals, max_eps=-0.01)


def test_transformer_forward_and_legality(tmp_path):
    """QTransformer: forward shapes, tanh range, and the replay's raw tokens
    must agree with live-play tokens (train/inference consistency)."""
    import torch
    from learn.dmc_model import QTransformer

    data_dir = _generate(tmp_path, n_deals=4)
    dataset = DMCDataset(load_dmc_deals(data_dir))
    item = dataset[0]
    net = QTransformer()
    out = net(
        torch.from_numpy(item["raw"]),
        torch.from_numpy(item["n_raw"]),
        torch.from_numpy(item["glob"]),
        torch.from_numpy(item["hand"]),
        torch.from_numpy(item["act"]),
    )
    assert out.shape == (32,)
    assert (out.abs() <= 1.0).all()

    # Live tokens from a stepped encode_q_observation must equal the
    # trajectory tokens at the same ply (same public information).
    import manille_ismcts as mi
    from learn.dmc_tokens import GLOBAL_DIM, encode_live
    deal = dataset.deals[0]
    actions = deal["plies"]["card"].astype(np.int32)
    traj_step = mi.encode_q_trajectory(
        np.ascontiguousarray(deal["hands"]), int(deal["trump"]),
        int(deal["declarer"]), np.ascontiguousarray(actions))
    t = 5
    obs5 = traj_step["obs"][t]
    played_mask = 0
    for k in range(t):
        played_mask |= 1 << int(actions[k])
    seat = int(traj_step["players"][t])
    own_mask = int(deal["hands"][seat]) & ~played_mask
    live = encode_live(obs5, t, own_mask)
    np.testing.assert_array_equal(live["raw"], item["raw"][t])
    np.testing.assert_array_equal(live["glob"], item["glob"][t])
    np.testing.assert_array_equal(live["hand"], item["hand"][t])
    assert live["n_raw"] == item["n_raw"][t]


def _shard_deals(tmp_path, n=12):
    from learn.dmc_replay import load_dmc_deals
    return load_dmc_deals(_generate(tmp_path, n_deals=n))


def test_permute_deal_is_an_automorphism(tmp_path):
    """A relabelled deal is the same deal: same score, same legal play."""
    import manille_ismcts as mi
    from learn.dmc_replay import (augmentation_perms, encode_dmc_deal,
                                  permute_deal)
    from manille_env.card import permute_hand_mask

    deals = _shard_deals(tmp_path)
    for deal in deals[:6]:
        base = encode_dmc_deal(deal)
        for perm in augmentation_perms(deal):
            moved = permute_deal(deal, perm)
            enc = encode_dmc_deal(moved)
            # The engine re-derives the score from the permuted play; if the
            # relabelling were not an automorphism this is where it breaks.
            assert int(enc["pts0"]) == int(base["pts0"])
            assert int(enc["pts1"]) == int(base["pts1"])
            assert int(moved["declarer"]) == int(deal["declarer"])
            # And the permuted deal must still be one the trump rule would
            # have produced, or it is a state the net never meets.
            dh = int(moved["hands"][int(moved["declarer"])])
            if int(moved["trump"]) >= 0:
                assert mi.apply_trump_rule(dh) == int(moved["trump"])


def test_augmentation_perms_reject_rule_inconsistent_relabellings(tmp_path):
    from learn.dmc_replay import augmentation_perms

    deals = _shard_deals(tmp_path)
    sizes = set()
    for deal in deals:
        perms = augmentation_perms(deal)
        assert (0, 1, 2, 3) in perms      # identity is always valid
        sizes.add(len(perms))
    # Diamonds/spades give 6, hearts/clubs 2, and the per-deal check can
    # admit more than the globally-safe 2 when no suit actually ties.
    assert sizes and max(sizes) <= 24


def test_augmented_dataset_changes_rows_but_not_returns(tmp_path):
    from learn.dmc_replay import DMCDataset

    deals = _shard_deals(tmp_path)[:8]
    plain = DMCDataset(deals)
    aug = DMCDataset(deals, augment=True, aug_seed=3)
    moved = 0
    for i in range(len(deals)):
        a, b = plain[i], aug[i]
        # The MC return is the deal's own final score: exactly invariant.
        assert np.allclose(a["ret"], b["ret"])
        if not np.array_equal(a["obs"], b["obs"]):
            moved += 1
    assert moved > 0, "augmentation never changed a single deal"


def test_paired_augmentation_returns_aligned_original_q_inputs(tmp_path):
    from learn.dmc_replay import DMCDataset

    deals = _shard_deals(tmp_path, n=1)
    plain = DMCDataset(deals)[0]
    paired = DMCDataset(deals, augment=True, aug_seed=3,
                        paired_original=True)[0]

    for key in ("obs", "act", "raw", "n_raw", "glob", "hand"):
        assert np.array_equal(paired["orig_" + key], plain[key])
        assert len(paired["orig_" + key]) == len(paired[key])
    assert np.array_equal(paired["ret"], plain["ret"])


def test_worker_rng_seeds_vary_with_the_deal_split():
    """Every worker receives the same cfg. Seeding on cfg["seed"] alone made
    all of them replay identical exploration coins and load the SAME league
    checkpoint, collapsing the league to one member per cycle."""
    from learn.dmc_selfplay import worker_rng_seeds

    a = worker_rng_seeds(5, 0)
    b = worker_rng_seeds(5, 12500)
    assert a != b, "workers share a seed again"
    assert a[0] != a[1], "exploration and league draws must not share a stream"
    assert worker_rng_seeds(5, 0) == a, "seeding is not reproducible"
    assert worker_rng_seeds(6, 0) != a, "run seed no longer matters"


def test_epsilon_ladder_parser_accepts_fixed_and_rejects_invalid_values():
    from learn.dmc_selfplay import parse_eps_ladder

    assert parse_eps_ladder("0.01") == (0.01,)
    assert parse_eps_ladder("0.3, 0.1,0") == (0.3, 0.1, 0.0)
    with pytest.raises(ValueError, match="empty"):
        parse_eps_ladder("")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_eps_ladder("0.1,1.1")


def test_max_eps_filter_uses_every_ply(tmp_path):
    """The exploration filter must read the WHOLE deal, not its first ply.

    A league deal records eps 0 on the frozen-opponent seats, so filtering on
    ply 0 keeps a high-exploration deal whenever its opening card happened to
    come from one of those seats -- 8.4% of the retained set on real shards,
    and it skewed the league share too, which would make the A/B arms differ
    by more than the epsilon under test.
    """
    from learn.dmc_replay import DMCDataset

    deals = np.zeros(2, dtype=DMC_DEAL_DTYPE)
    for d in deals:
        d["hands"] = [0, 0, 0, 0]
        d["trump"] = 0
    deals[0]["plies"]["eps"] = 0.01          # genuinely low-exploration
    deals[1]["plies"]["eps"] = 0.30          # a 0.30 deal ...
    deals[1]["plies"]["eps"][0] = 0.0        # ... whose first ply was a league seat

    kept = DMCDataset(deals, max_eps=0.04).deals
    assert len(kept) == 1
    assert kept["plies"]["eps"].max() <= 0.04

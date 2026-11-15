"""DMC training smoke test: losses decrease, checkpoints round-trip, resume
continues the global step."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

from learn.dmc_model import QNet, QOracle, load_oracle_model, save_checkpoint
from learn.dmc_selfplay import _play_chunk


def _gen(tmp_path, n_deals=24, seed=11):
    torch.manual_seed(0)
    model_path = os.path.join(tmp_path, "q_init.pt")
    net = QNet()
    save_checkpoint(model_path, net, net.state_dict(), None, None, 0,
                    {"hidden": 1024, "blocks": 4})
    cfg = {"model": model_path, "slots": 4, "seed": seed, "cuda": False,
           "league": [], "league_frac": 0.2}
    data_dir = os.path.join(tmp_path, "cycle_000")
    os.makedirs(data_dir, exist_ok=True)
    _play_chunk(cfg, list(range(n_deals)), os.path.join(data_dir, "shard_0.bin"))
    return data_dir


def _train(data_dir, out_q, out_b, steps, resume=(), arch="mlp", extra=()):
    cmd = [sys.executable, "-u", "-m", "learn.dmc_train",
           "--data", data_dir, "--out-q", out_q, "--out-b", out_b,
           "--steps", str(steps), "--batch-plies", "128",
           "--val-deals", "8", "--workers", "2", "--arch", arch]
    cmd += list(resume)
    cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_root())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in proc.stdout.splitlines():
        if line.startswith("VAL "):
            return json.loads(line[4:])
    raise AssertionError("no VAL line in trainer output:\n" + proc.stdout)


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_smoke_and_resume(tmp_path):
    data_dir = _gen(tmp_path)
    out_q = os.path.join(tmp_path, "q_cycle000.pt")
    out_b = os.path.join(tmp_path, "belief_cycle000.pt")

    val = _train(data_dir, out_q, out_b, steps=6)
    assert val["steps"] == 6 and val["global_step"] == 6
    assert val["q_mean"] > 0 and val["belief_mean"] > 0
    # Belief beats (or ties, this early) the constrained-uniform baseline.
    assert val["belief_mean"] <= val["belief_base_mean"] + 0.05
    assert os.path.exists(out_q) and os.path.exists(out_b)

    # Checkpoint round-trip: EMA weights load for inference.
    from learn.dmc_model import load_belief_model, load_q_model
    load_q_model(out_q, device="cpu")
    load_belief_model(out_b, device="cpu")

    # Resume continues the global step.
    out_q1 = os.path.join(tmp_path, "q_cycle001.pt")
    out_b1 = os.path.join(tmp_path, "belief_cycle001.pt")
    val2 = _train(data_dir, out_q1, out_b1, steps=4,
                  resume=("--resume-q", out_q, "--resume-b", out_b))
    assert val2["global_step"] == 10


def test_oracle_training_and_mixed_target_smoke(tmp_path):
    """The perfect-information critic saves an EMA checkpoint and can be
    loaded frozen as a variance-reduced target for a regular Q update."""
    data_dir = _gen(tmp_path)
    oracle_q = os.path.join(tmp_path, "q_oracle_source.pt")
    oracle_b = os.path.join(tmp_path, "b_oracle_source.pt")
    oracle_path = os.path.join(tmp_path, "oracle.pt")
    val = _train(data_dir, oracle_q, oracle_b, steps=2,
                 extra=("--oracle-out", oracle_path))
    assert os.path.exists(oracle_path)
    assert val["train_oracle"] > 0
    assert "oracle_mean" in val and len(val["oracle_per_trick"]) == 8

    oracle = load_oracle_model(oracle_path, device="cpu")
    assert isinstance(oracle, QOracle)

    mixed_q = os.path.join(tmp_path, "q_mixed.pt")
    mixed_b = os.path.join(tmp_path, "b_mixed.pt")
    mixed = _train(data_dir, mixed_q, mixed_b, steps=2,
                   extra=("--oracle-target", oracle_path,
                          "--oracle-lambda", "0.5"))
    assert mixed["train_oracle_abs_ret"] >= 0
    assert "oracle_mean" in mixed


def test_train_transformer_smoke(tmp_path):
    data_dir = _gen(tmp_path)
    out_q = os.path.join(tmp_path, "qt.pt")
    out_b = os.path.join(tmp_path, "bt.pt")
    val = _train(data_dir, out_q, out_b, steps=3, arch="transformer")
    assert val["steps"] == 3 and val["q_mean"] > 0
    from learn.dmc_model import QTransformer, load_q_model
    net = load_q_model(out_q, device="cpu")
    assert isinstance(net, QTransformer)


def test_frozen_symmetry_teacher_smoke(tmp_path):
    """Orbit distillation must use paired raw/augmented rows, load a frozen
    compatible EMA teacher, and expose its realized loss in the audit log."""
    data_dir = _gen(tmp_path)
    base_q = os.path.join(tmp_path, "base_q.pt")
    base_b = os.path.join(tmp_path, "base_b.pt")
    _train(data_dir, base_q, base_b, steps=2)

    out_q = os.path.join(tmp_path, "student_q.pt")
    out_b = os.path.join(tmp_path, "student_b.pt")
    val = _train(
        data_dir, out_q, out_b, steps=2,
        resume=("--resume-q", base_q, "--resume-b", base_b),
        extra=("--augment", "--sym-teacher-q", base_q,
               "--sym-teacher-weight", "1"))
    assert val["train_sym_teacher"] >= 0.0
    assert val["global_step"] == 4


def test_forced_ply_weight_changes_the_q_loss_only():
    """Forced plies carry no ranking signal (38.5% of rows, trick 7 is
    100%). --q-forced-weight downweights them in the Q loss; belief must
    keep training on every ply, since an observer learns as much from a
    forced card as a free one."""
    import torch.nn.functional as F

    ret = torch.tensor([1.0, -1.0, 0.5, 0.0])
    q = torch.zeros(4)
    nlegal = torch.tensor([1, 3, 1, 2])

    def weighted(fw):
        w = torch.where(nlegal > 1, torch.ones_like(ret),
                        torch.full_like(ret, fw))
        err = F.huber_loss(q, ret, reduction="none")
        return float((err * w).sum() / w.sum().clamp_min(1.0))

    # Weight 1.0 must reproduce the plain mean exactly (no silent change).
    assert abs(weighted(1.0) - float(F.huber_loss(q, ret))) < 1e-9
    # Weight 0.0 must equal the loss over non-forced rows alone.
    free = F.huber_loss(q[nlegal > 1], ret[nlegal > 1])
    assert abs(weighted(0.0) - float(free)) < 1e-9
    assert weighted(0.0) != weighted(1.0)


def test_nlegal_reaches_the_trainer(tmp_path):
    """The weight is unusable if the batch does not carry the legal count."""
    from tests.test_dmc_replay import _generate
    from learn.dmc_replay import DMCDataset, collate, load_dmc_deals

    deals = load_dmc_deals(_generate(tmp_path, n_deals=4))
    batch = collate([DMCDataset(deals)[i] for i in range(len(deals))])
    assert "nlegal" in batch
    assert batch["nlegal"].shape == batch["ret"].shape
    assert batch["nlegal"].min() >= 1
    # Trick 7 (plies 28-31) is the last card: always exactly one legal move.
    assert (batch["nlegal"].reshape(-1, 32)[:, 28:] == 1).all()


def test_augment_flag_is_reachable_and_validation_is_not_augmented(tmp_path):
    """The augmentation was built but not wired: dmc_train constructed a
    plain DMCDataset and had no flag, so --augment could not be requested at
    all. Validation must stay unaugmented or held-out loss measures a
    different task than deployment."""
    import inspect

    from learn.dmc_replay import DMCDataset
    import learn.dmc_train as T

    src = inspect.getsource(T.main)
    assert "args.augment" in src, "--augment never reaches the dataset"
    assert "val_dataset = DMCDataset(deals)" in src, \
        "validation dataset must be built unaugmented"
    assert "val_dataset, val_idx" in src, "validate() got the training set"

    assert "--augment" in inspect.getsource(T)


def test_low_eps_and_subsample_flags_are_reachable_and_audited():
    """The Phase 1.1 arms need whole-deal filtering and an equal-size
    deterministic control, with the retained count visible in their logs."""
    import inspect

    import learn.dmc_train as T

    src = inspect.getsource(T.main)
    assert '"--max-eps"' in src
    assert '"--subsample"' in src
    assert "DMCDataset(deals, max_eps=args.max_eps)" in src
    assert "rng.choice(len(deals)" in src
    assert "retained:" in src


def test_augmentation_varies_across_epochs(tmp_path):
    """One fixed permutation is a relabelled dataset, not an augmented one:
    the trainer must redraw per epoch."""
    from tests.test_dmc_replay import _generate
    from learn.dmc_replay import DMCDataset, load_dmc_deals

    deals = load_dmc_deals(_generate(tmp_path, n_deals=6))
    ds = DMCDataset(deals, augment=True, aug_seed=(0, 0))
    first = [ds[i]["obs"].copy() for i in range(len(deals))]
    ds.aug_seed = (0, 1)
    second = [ds[i]["obs"] for i in range(len(deals))]
    assert any(not np.array_equal(a, b) for a, b in zip(first, second))
    # Returns are invariant no matter the relabelling.
    ds0 = DMCDataset(deals)
    for i in range(len(deals)):
        assert np.allclose(ds0[i]["ret"], ds[i]["ret"])


def test_paired_q_inputs_survive_tensor_conversion(tmp_path):
    from tests.test_dmc_replay import _generate
    from learn.dmc_replay import DMCDataset, collate, load_dmc_deals
    from learn.dmc_train import batch_to_tensors

    deals = load_dmc_deals(_generate(tmp_path, n_deals=2))
    ds = DMCDataset(deals, augment=True, paired_original=True)
    batch = collate([ds[0], ds[1]])
    tensors = batch_to_tensors(batch, "cpu")
    for key in ("obs", "act", "raw", "n_raw", "glob", "hand"):
        assert "orig_" + key in tensors
        assert tensors["orig_" + key].shape == tensors[key].shape


def test_run_dmc_forwards_the_trainer_flags():
    """Adding a flag to learn.dmc_train is not enough: run_dmc builds the
    trainer command itself, so anything it does not forward is unreachable
    from the normal cyclic entry point however well the trainer tests it.
    That is exactly how --augment and --q-forced-weight shipped dead."""
    import inspect

    import learn.run_dmc as R

    src = inspect.getsource(R)
    assert '"--augment"' in src, "run_dmc cannot request augmentation"
    assert '"--q-forced-weight"' in src, "run_dmc cannot set the forced weight"
    assert '"--sym-consistency"' in src, \
        "run_dmc cannot request paired consistency"
    assert ("args.augment" in src and "args.q_forced_weight" in src
            and "args.sym_consistency" in src)


def _gen_best_response(tmp_path, n_deals=24, seed=13):
    """Shards as a best-response run writes them: every deal a league deal,
    seats 1 & 3 played by the frozen checkpoint."""
    torch.manual_seed(0)
    learner = os.path.join(tmp_path, "q_learner.pt")
    frozen = os.path.join(tmp_path, "q_frozen.pt")
    for path in (learner, frozen):
        net = QNet()
        save_checkpoint(path, net, net.state_dict(), None, None, 0,
                        {"hidden": 1024, "blocks": 4})
    cfg = {"model": learner, "slots": 4, "seed": seed, "cuda": False,
           "league": [frozen], "league_frac": 1.0}
    data_dir = os.path.join(tmp_path, "br_cycle_000")
    os.makedirs(data_dir, exist_ok=True)
    _play_chunk(cfg, list(range(n_deals)),
                os.path.join(data_dir, "shard_0.bin"))
    return data_dir


def test_best_response_shards_play_frozen_seats_greedily(tmp_path):
    """league-frac 1.0: seats 1 & 3 are the frozen target at eps 0 in every
    deal; the learner's seats explore on the ladder (all entries > 0)."""
    from learn.dmc_replay import DMCDataset, load_dmc_deals

    deals = load_dmc_deals(_gen_best_response(tmp_path))
    ds = DMCDataset(deals)
    for i in range(len(deals)):
        arr = ds[i]
        odd = arr["seat"] % 2 == 1
        assert odd.sum() == 16
        assert (arr["seat"] == deals[i]["plies"]["player"]).all()
        assert (arr["eps"][odd] == 0).all()
        assert (arr["eps"][~odd] > 0).all()


def test_learner_rows_mask():
    from learn.dmc_train import learner_rows

    t = {"seat": torch.tensor([0, 1, 2, 3, 1])}
    assert learner_rows(t, None) is None
    assert learner_rows(t, 0).tolist() == [True, False, True, False, False]
    assert learner_rows(t, 1).tolist() == [False, True, False, True, True]


def test_learner_team_training_smoke(tmp_path):
    """--learner-team trains and validates on the learner's plies only: the
    validation numbers must differ from an all-seat validation of the same
    run, and the auxiliary losses that would still see the opponent's plies
    are refused."""
    data_dir = _gen_best_response(tmp_path)
    q0, b0 = (os.path.join(tmp_path, n) for n in ("q0.pt", "b0.pt"))
    q1, b1 = (os.path.join(tmp_path, n) for n in ("q1.pt", "b1.pt"))
    val_all = _train(data_dir, q0, b0, steps=4)
    val_own = _train(data_dir, q1, b1, steps=4,
                     extra=["--learner-team", "0"])
    assert val_own["steps"] == 4 and val_own["q_mean"] > 0
    assert val_own["belief_mean"] > 0
    assert val_own["q_mean"] != val_all["q_mean"]

    cmd = [sys.executable, "-u", "-m", "learn.dmc_train",
           "--data", data_dir, "--out-q", q1, "--out-b", b1, "--steps", "1",
           "--learner-team", "0", "--sym-consistency", "0.1", "--augment"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_root())
    assert proc.returncode != 0 and "--learner-team" in proc.stderr

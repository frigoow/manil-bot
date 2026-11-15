"""DMC training (manille_dmc_plan §1.3, §1.4).

Q loss: Huber between Q_θ(s, a_played) and the full-episode Monte Carlo
return (normalized final team points of the deciding player's team). No
bootstrapping, no target network. Belief loss: NLL of the true holder under
B_phi (mandatory §5.2 prior-offset parameterization, no label smoothing).
Both nets train on the same batches in one process.

The buffer is the sliding window of the newest --buffer-deals deals across
the cycle shard dirs passed in --data (oldest first). One invocation runs
--steps gradient steps and writes atomic EMA checkpoints; the orchestrator
(learn/run_dmc.py) calls it once per cycle. Resume restores both nets,
optimizers, scaler, EMA shadows and the global step.

    python -m learn.dmc_train --data data/dmc/cycle_000 data/dmc/cycle_001 \
        --out-q runs/dmc/q_cycle001.pt --out-b runs/dmc/belief_cycle001.pt \
        --steps 1500 --resume-q runs/dmc/q_cycle000.pt --resume-b runs/dmc/belief_cycle000.pt
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .dmc_model import (BeliefNet, QOracle, expand_checkpoint_inputs,
                        load_oracle_model, make_q_model,
                        save_checkpoint)
from .dmc_replay import DMCDataset, iter_ply_batches, load_dmc_deals
from .model import belief_probs, split_belief_inputs


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


# NOTE for anyone tempted by a per-column learning rate on the 1.3 inputs:
# scaling their GRADIENT does nothing. AdamW divides by the square root of the
# gradient's own second moment, so a constant factor cancels and the update is
# identical -- measured, the weights came out the same to five decimals. A real
# per-column rate needs those columns as their own parameter (hence their own
# param group), which means changing the input layer's structure. Raising the
# learning rate for the whole net, in both arms, is the cheap alternative.
def mask_new_features(t):
    """Zero the 1.3 tactical features in place — the control arm of the
    feature A/B. Masking rather than a second encoder keeps both arms on
    byte-identical data, so the only difference is what the net can see.
    Covers all three entry points: the observation (MLP and belief), the
    global token (GRU) and the candidate vector."""
    from .dmc_tokens import LEGACY_GLOBAL_DIM
    from .encoding import LEGACY_ACT_DIM, LEGACY_OBS_DIM

    t["obs"][:, LEGACY_OBS_DIM:] = 0.0
    t["act"][:, LEGACY_ACT_DIM:] = 0.0
    t["glob"][:, LEGACY_GLOBAL_DIM:] = 0.0
    for key in ("orig_obs", "orig_act", "orig_glob"):
        if key in t:
            legacy = {"orig_obs": LEGACY_OBS_DIM, "orig_act": LEGACY_ACT_DIM,
                      "orig_glob": LEGACY_GLOBAL_DIM}[key]
            t[key][:, legacy:] = 0.0
    return t


def batch_to_tensors(batch, device):
    tensors = {
        "obs": torch.from_numpy(batch["obs"]).to(device),
        "act": torch.from_numpy(batch["act"]).to(device),
        "ret": torch.from_numpy(batch["ret"]).to(device),
        "belief": torch.from_numpy(batch["belief"]).to(device),
        "trick": torch.from_numpy(batch["trick"]).to(device),
        "raw": torch.from_numpy(batch["raw"]).to(device),
        "n_raw": torch.from_numpy(batch["n_raw"]).to(device),
        "glob": torch.from_numpy(batch["glob"]).to(device),
        "hand": torch.from_numpy(batch["hand"]).to(device),
        "nlegal": torch.from_numpy(batch["nlegal"]).to(device),
        "eps": torch.from_numpy(batch["eps"]).to(device),
    }
    if "seat" in batch:  # absent from distill/policy-head datasets
        tensors["seat"] = torch.from_numpy(batch["seat"]).to(device)
    for key in ("obs", "act", "raw", "n_raw", "glob", "hand"):
        paired_key = "orig_" + key
        if paired_key in batch:
            tensors[paired_key] = torch.from_numpy(
                batch[paired_key]).to(device)
    return tensors


def choice_rows(deals):
    """Per-deal ChoiceVNet training rows from the raw shard array:
    (declarer initial-hand multi-hot [n,32], chosen-suit one-hot [n,4],
    declarer-team normalized return [n], declarer hand masks [n])."""
    dec = deals["declarer"].astype(np.int64)
    masks = np.take_along_axis(deals["hands"], dec[:, None], axis=1)[:, 0]
    hands = ((masks[:, None] >> np.arange(32)[None, :]) & 1
             ).astype(np.float32)
    suits = np.eye(4, dtype=np.float32)[deals["trump"].astype(np.int64)]
    pts = np.where(dec % 2 == 0, deals["pts0"],
                   deals["pts1"]).astype(np.float32)
    return hands, suits, (pts - 30.0) / 30.0, masks


def train_choice(args, deals, train_idx, val_idx, device, rng):
    """Fit ChoiceVNet on per-deal (hand, chosen suit, return) rows — the
    trump-choice DMC head (Phase 6). Same MC-return target as the play Q;
    counterfactual coverage comes from the generator's --choice-eps
    exploration. Prints a 'CHOICE {json}' marker and saves --choice-out."""
    import manille_ismcts as mi
    from .dmc_trump_choice import ChoiceVNet

    hands, suits, rets, masks = choice_rows(deals)
    cfg = {"kind": "value", "hidden": 256, "blocks": 2}
    net = ChoiceVNet(hidden=cfg["hidden"], blocks=cfg["blocks"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.choice_lr,
                            weight_decay=1e-4)
    ema = EMA(net)
    step0 = 0
    if args.resume_choice:
        ck = torch.load(args.resume_choice, map_location=device,
                        weights_only=True)
        net.load_state_dict(ck["raw_model"])
        ema.shadow = ck["model"]
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        step0 = ck["step"]

    t0 = time.time()
    net.train()
    running, seen = 0.0, 0
    for _ in range(args.choice_steps):
        idx = train_idx[rng.integers(len(train_idx),
                                     size=args.choice_batch)]
        h = torch.from_numpy(hands[idx]).to(device)
        s = torch.from_numpy(suits[idx]).to(device)
        r = torch.from_numpy(rets[idx]).to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.huber_loss(net(h, s), r)
        if not torch.isfinite(loss):
            print("ERROR non-finite choice loss — hard stop", flush=True)
            raise SystemExit(3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema.update(net)
        running += float(loss.detach()) * len(idx)
        seen += len(idx)

    # EMA-weight validation: huber on the chosen suit, and how the argmax
    # relates to the deterministic rule (agreement + margin).
    val_net = ChoiceVNet(hidden=cfg["hidden"], blocks=cfg["blocks"]).to(device)
    val_net.load_state_dict(ema.shadow)
    val_net.eval()
    with torch.no_grad():
        h = torch.from_numpy(hands[val_idx]).to(device)
        s = torch.from_numpy(suits[val_idx]).to(device)
        r = torch.from_numpy(rets[val_idx]).to(device)
        val_huber = float(F.huber_loss(val_net(h, s), r))
        n = len(val_idx)
        eye = torch.eye(4, device=device)
        v4 = val_net(
            h.unsqueeze(1).expand(n, 4, 32).reshape(-1, 32),
            eye.unsqueeze(0).expand(n, 4, 4).reshape(-1, 4),
        ).reshape(n, 4)
        pick = v4.argmax(dim=1).cpu().numpy()
        top2 = v4.topk(2, dim=1).values
        margin = float((top2[:, 0] - top2[:, 1]).mean())
    rule = np.array([int(mi.apply_trump_rule(np.int64(int(m))))
                     for m in masks[val_idx]])
    agree = float((pick == rule).mean())
    data_offrule = float(np.mean(
        deals["trump"][val_idx].astype(np.int64) != rule))

    out = {
        "choice_steps": args.choice_steps,
        "choice_global_step": step0 + args.choice_steps,
        "choice_train": running / max(seen, 1),
        "choice_val": val_huber,
        "choice_rule_agree": agree,
        "choice_margin": margin,
        "choice_data_offrule": data_offrule,
        "choice_elapsed": time.time() - t0,
    }
    print("CHOICE " + json.dumps(out), flush=True)
    save_checkpoint(args.choice_out, net, ema.shadow, opt, None,
                    step0 + args.choice_steps, cfg)
    print(f"saved {args.choice_out}", flush=True)


def q_forward(q_net, arch, t):
    if arch in ("transformer", "gru"):
        return q_net(t["raw"], t["n_raw"], t["glob"], t["hand"], t["act"])
    return q_net(t["obs"], t["act"])


def belief_nll(offsets, obs, belief):
    prior, mask = split_belief_inputs(obs)
    probs = belief_probs(offsets.float(), prior, mask)
    logq = torch.log(probs.clamp_min(1e-9))
    n_unseen = belief.sum().clamp_min(1.0)
    return -(belief * logq).sum() / n_unseen


def learner_rows(t, learner_team):
    """Bool [n] mask of the plies the learner trains on, or None for all.

    In a best-response run seats 1 & 3 are a frozen opponent in every deal.
    Fitting Q to its decisions would spend capacity modelling the target
    rather than exploiting it, and its belief rows describe the learner's
    own team from outside -- so both losses (and validation) drop them."""
    if learner_team is None:
        return None
    return (t["seat"] % 2) == learner_team


@torch.no_grad()
def validate(q_ema, b_ema, q_cfg, b_cfg, dataset, val_idx, device,
             learner_team=None):
    """EMA-weight validation: Q huber and belief NLL per trick (belief next
    to the zero-offset constrained-uniform baseline — the A-4 health check).
    With learner_team set, only that team's plies count."""
    q_net = make_q_model(q_cfg, device)
    q_net.load_state_dict(q_ema.shadow)
    q_net.eval()
    b_net = BeliefNet(hidden=b_cfg["hidden"], blocks=b_cfg["blocks"]).to(device)
    b_net.load_state_dict(b_ema.shadow)
    b_net.eval()

    q_loss = np.zeros(8)
    b_model = np.zeros(8)
    b_base = np.zeros(8)
    b_n = np.zeros(8)
    q_n = np.zeros(8)
    for batch in iter_ply_batches(dataset, val_idx, 2048, workers=4):
        t = batch_to_tensors(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            q = q_forward(q_net, q_cfg.get("arch", "mlp"), t)
            offsets = b_net(t["obs"])
        err = F.huber_loss(q.float(), t["ret"], reduction="none")
        own = learner_rows(t, learner_team)
        if own is not None:
            # Zeroed belief targets drop a row from both the NLL sum and
            # the unseen-card count; the Q selection is masked below.
            t["belief"] = t["belief"] * own[:, None, None]
        prior, mask = split_belief_inputs(t["obs"])
        n_unseen = t["belief"].sum(dim=(1, 2))
        for arr, offs in ((b_model, offsets.float()),
                          (b_base, torch.zeros_like(offsets))):
            probs = belief_probs(offs, prior, mask)
            logq = torch.log(probs.clamp_min(1e-9))
            nll = -(t["belief"] * logq).sum(dim=(1, 2))
            for trick in range(8):
                sel = t["trick"] == trick
                arr[trick] += float(nll[sel].sum())
        for trick in range(8):
            sel = t["trick"] == trick
            if own is not None:
                sel = sel & own
            q_loss[trick] += float(err[sel].sum())
            q_n[trick] += int(sel.sum())
            b_n[trick] += float(n_unseen[sel].sum())

    return {
        "q_per_trick": (q_loss / np.maximum(q_n, 1)).tolist(),
        "q_mean": float(q_loss.sum() / max(q_n.sum(), 1)),
        "belief_per_trick": (b_model / np.maximum(b_n, 1)).tolist(),
        "belief_base_per_trick": (b_base / np.maximum(b_n, 1)).tolist(),
        "belief_mean": float(b_model.sum() / max(b_n.sum(), 1)),
        "belief_base_mean": float(b_base.sum() / max(b_n.sum(), 1)),
    }


@torch.no_grad()
def validate_oracle(oracle, dataset, val_idx, device):
    """Held-out oracle Huber against actual returns, split by trick.

    An oracle target is a lower-variance surrogate, not new ground truth. Its
    validation metric therefore remains the same realized-return Huber used
    for the deployable Q model.
    """
    oracle.eval()
    loss = np.zeros(8)
    count = np.zeros(8)
    for batch in iter_ply_batches(dataset, val_idx, 2048, workers=4):
        t = batch_to_tensors(batch, device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device == "cuda"):
            q_star = oracle(t["obs"], t["belief"], t["act"])
        err = F.huber_loss(q_star.float(), t["ret"], reduction="none")
        for trick in range(8):
            sel = t["trick"] == trick
            loss[trick] += float(err[sel].sum())
            count[trick] += int(sel.sum())
    return {
        "oracle_per_trick": (loss / np.maximum(count, 1)).tolist(),
        "oracle_mean": float(loss.sum() / max(count.sum(), 1)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True,
                    help="cycle shard dirs, oldest first")
    ap.add_argument("--out-q", required=True)
    ap.add_argument("--out-b", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--buffer-deals", type=int, default=150000)
    ap.add_argument("--max-eps", type=float, default=None,
                    help="retain only deals recorded at ε <= this value; "
                         "ε is sampled once per deal, so the filter keeps "
                         "whole trajectories")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="deterministically retain this fraction of the "
                         "loaded deals before the train/validation split; "
                         "use with --seed for matched control arms")
    ap.add_argument("--batch-plies", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--total-steps", type=int, default=200000,
                    help="cosine horizon across the whole run")
    ap.add_argument("--c-b", type=float, default=0.5)
    ap.add_argument("--mask-new-features", action="store_true",
                    help="zero the 1.3 tactical features (observation, "
                         "global token and candidate). This is the CONTROL "
                         "arm for the feature A/B: identical data, identical "
                         "weights, the net simply cannot see them.")
    ap.add_argument("--oracle-out", default=None,
                    help="train and save a perfect-information QOracle on "
                         "the replay's true-holder labels")
    ap.add_argument("--oracle-target", default=None,
                    help="frozen QOracle EMA checkpoint used to reduce the "
                         "deployable Q target variance")
    ap.add_argument("--oracle-lambda", type=float, default=0.0,
                    help="Q target mixture: λ·QOracle + (1−λ)·return; "
                         "requires --oracle-target")
    ap.add_argument("--augment", action="store_true",
                    help="suit-permutation augmentation on the training "
                         "deals. Relabelling suits is an exact automorphism "
                         "and the MC return is literally the same number, so "
                         "this is label-preserving data multiplication (up to "
                         "6x per deal, 24x at mull), not regularization — and "
                         "it is the 1x-inference route to the invariance that "
                         "test-time orbit averaging buys at 6x.")
    ap.add_argument("--sym-consistency", type=float, default=0.0,
                    help="paired raw/relabelled Q consistency weight. "
                         "Requires --augment; adds one Q forward during "
                         "training but no inference work.")
    ap.add_argument("--sym-teacher-q", default=None,
                    help="frozen EMA Q checkpoint whose prediction on the "
                         "random rule-consistent view teaches the student's "
                         "raw view (stochastic orbit distillation)")
    ap.add_argument("--sym-teacher-weight", type=float, default=0.0,
                    help="weight of --sym-teacher-q orbit distillation; "
                         "requires --augment and --sym-teacher-q")
    ap.add_argument("--q-forced-weight", type=float, default=1.0,
                    help="Q-loss weight for plies with one legal card. "
                         "38.5%% of rows are forced (trick 7 is 100%%) and "
                         "teach nothing about ranking, but they do anchor "
                         "absolute Q calibration — 0.0 drops them entirely, "
                         "1.0 is the historical behaviour. Belief always "
                         "trains on every ply.")
    ap.add_argument("--learner-team", type=int, choices=[0, 1],
                    default=None,
                    help="best-response runs: train Q and belief only on "
                         "this team's plies (seat %% 2). dmc_selfplay's "
                         "league plays seats 1 & 3, so the learner is 0")
    ap.add_argument("--val-deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume-q", type=str, default=None)
    ap.add_argument("--resume-b", type=str, default=None)
    ap.add_argument("--arch", choices=["mlp", "transformer", "gru"],
                    default="mlp")
    ap.add_argument("--choice-out", default=None,
                    help="also train the trump-choice head (ChoiceVNet) on "
                         "per-deal rows and save it here (Phase 6)")
    ap.add_argument("--resume-choice", default=None)
    ap.add_argument("--choice-steps", type=int, default=200)
    ap.add_argument("--choice-batch", type=int, default=4096)
    ap.add_argument("--choice-lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.sym_consistency < 0:
        ap.error("--sym-consistency must be non-negative")
    if args.sym_teacher_weight < 0:
        ap.error("--sym-teacher-weight must be non-negative")
    if args.sym_consistency and not args.augment:
        ap.error("--sym-consistency requires --augment")
    if args.sym_teacher_weight and not args.augment:
        ap.error("--sym-teacher-weight requires --augment")
    if args.sym_teacher_weight and not args.sym_teacher_q:
        ap.error("--sym-teacher-weight requires --sym-teacher-q")
    if args.sym_teacher_q and not args.sym_teacher_weight:
        ap.error("--sym-teacher-q requires positive --sym-teacher-weight")
    if not 0 < args.subsample <= 1:
        ap.error("--subsample must be in (0, 1]")
    if not 0 <= args.oracle_lambda <= 1:
        ap.error("--oracle-lambda must be in [0, 1]")
    if args.oracle_target is None and args.oracle_lambda:
        ap.error("--oracle-lambda requires --oracle-target")
    if args.oracle_target is not None and not args.oracle_lambda:
        ap.error("--oracle-target requires positive --oracle-lambda")

    if args.learner_team is not None and (
            args.sym_consistency or args.sym_teacher_weight
            or args.oracle_out or args.oracle_lambda):
        # Those auxiliary losses would still read the opponent's plies.
        ap.error("--learner-team masks only the Q and belief losses; it "
                 "does not combine with symmetry/oracle losses")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    deals = load_dmc_deals(args.data)
    if len(deals) > args.buffer_deals:
        deals = deals[-args.buffer_deals:]
    loaded = len(deals)
    rng = np.random.default_rng(args.seed)
    # Keep the low-ε arm and the random control arm comparable at the deal
    # level. Do this before the split so validation remains a held-out sample
    # from precisely the distribution the arm trains on.
    filtered_dataset = DMCDataset(deals, max_eps=args.max_eps)
    deals = filtered_dataset.deals
    after_eps = len(deals)
    if args.subsample < 1:
        keep = int(len(deals) * args.subsample)
        if keep == 0:
            ap.error("--subsample retained no deals")
        idx = np.sort(rng.choice(len(deals), size=keep, replace=False))
        deals = deals[idx]
    if not len(deals):
        ap.error("--max-eps retained no deals")
    if args.val_deals >= len(deals):
        ap.error("--val-deals must be smaller than the retained deal count")
    print(
        f"retained: {len(deals)} deals "
        f"(loaded={loaded}, after-max-eps={after_eps}"
        f"{'' if args.max_eps is None else f' (max-eps={args.max_eps:g})'}, "
        f"subsample={args.subsample:g})",
        flush=True,
    )
    perm = rng.permutation(len(deals))
    val_idx = perm[: args.val_deals]
    train_idx = perm[args.val_deals:]
    # Suit-permutation augmentation applies to TRAINING only. Validation
    # must stay on the real distribution, or the held-out loss measures a
    # different task than deployment and is no longer comparable to any
    # earlier run.
    dataset = DMCDataset(
        deals, augment=args.augment, aug_seed=args.seed,
        paired_original=(args.sym_consistency > 0
                         or args.sym_teacher_weight > 0))
    val_dataset = DMCDataset(deals)
    print(f"buffer: {len(deals)} deals ({len(train_idx)} train / "
          f"{len(val_idx)} val){' [suit-augmented]' if args.augment else ''}",
          flush=True)

    if args.arch == "transformer":
        q_cfg = {"arch": "transformer", "d_model": 128, "layers": 4,
                 "heads": 4}
    elif args.arch == "gru":
        q_cfg = {"arch": "gru", "d_model": 512, "layers": 1,
                 "hidden": 2048, "blocks": 4}
    else:
        q_cfg = {"arch": "mlp", "hidden": 1024, "blocks": 4}
    b_cfg = {"hidden": 512, "blocks": 2}
    q_net = make_q_model(q_cfg, device)
    b_net = BeliefNet(**b_cfg).to(device)
    oracle_net = None
    oracle_ema = None
    oracle_opt = None
    oracle_cfg = {"kind": "q_oracle", "hidden": 1024, "blocks": 4}
    if args.oracle_out:
        oracle_net = QOracle(hidden=oracle_cfg["hidden"],
                             blocks=oracle_cfg["blocks"]).to(device)
        oracle_ema = EMA(oracle_net)
        oracle_opt = torch.optim.AdamW(oracle_net.parameters(), lr=args.lr,
                                       weight_decay=1e-4)
    params = sum(p.numel() for p in q_net.parameters())
    params += sum(p.numel() for p in b_net.parameters())
    if oracle_net is not None:
        params += sum(p.numel() for p in oracle_net.parameters())
    print(f"model params: {params / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(
        list(q_net.parameters()) + list(b_net.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scaler = torch.GradScaler("cuda", enabled=device == "cuda")
    q_ema, b_ema = EMA(q_net), EMA(b_net)
    sym_teacher = None
    oracle_target = None
    step0 = 0
    if args.resume_q and args.resume_b:
        ck = torch.load(args.resume_q, map_location=device, weights_only=True)
        # Checkpoints from before the 1.3 tactical block have a narrower
        # input layer. Grow it with zeros for the new columns: the expanded
        # net starts bit-identical to the old one, so an arm that masks the
        # new features is a true control rather than a different init. A
        # checkpoint already at the current width is untouched.
        q_kind = ck.get("config", {}).get("arch", "mlp")
        # Width BEFORE expanding: comparing after it always reads "unchanged"
        # and then restores an optimizer state shaped for the old layer.
        width_before = ck["raw_model"]["input.weight"].shape[1]
        expand_checkpoint_inputs(ck["raw_model"], q_kind)
        expand_checkpoint_inputs(ck["model"], q_kind)
        grew = ck["raw_model"]["input.weight"].shape[1] != width_before
        q_net.load_state_dict(ck["raw_model"])
        q_ema.shadow = ck["model"]
        q_cfg = ck["config"]
        if not grew:
            # The optimizer state is keyed by parameter shape; after an
            # expansion its moments describe the old layer and cannot be
            # restored. Starting Adam fresh is the honest option, and both
            # arms do it identically.
            opt.load_state_dict(ck["opt"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
        else:
            print("input layer expanded to the current feature layout; "
                  "optimizer state reset", flush=True)
        step0 = ck["step"]
        cb = torch.load(args.resume_b, map_location=device, weights_only=True)
        expand_checkpoint_inputs(cb["raw_model"], "belief")
        expand_checkpoint_inputs(cb["model"], "belief")
        b_net.load_state_dict(cb["raw_model"])
        b_ema.shadow = cb["model"]
        b_cfg = cb["config"]
        print(f"resumed at global step {step0}", flush=True)

    if args.oracle_target:
        oracle_target = load_oracle_model(args.oracle_target, device)
        oracle_target.requires_grad_(False)
        print(f"frozen oracle target: {args.oracle_target} "
              f"(lambda {args.oracle_lambda:g})", flush=True)

    if args.sym_teacher_q:
        from .dmc_model import load_q_model
        teacher_ck = torch.load(args.sym_teacher_q, map_location="cpu",
                                weights_only=True)
        teacher_cfg = teacher_ck.get("config", {})
        if teacher_cfg != q_cfg:
            raise ValueError("symmetry teacher architecture/config does not "
                             "match the student")
        sym_teacher = load_q_model(args.sym_teacher_q, device)
        sym_teacher.requires_grad_(False)
        print(f"frozen symmetry teacher: {args.sym_teacher_q} "
              f"(weight {args.sym_teacher_weight:g})", flush=True)

    def lr_at(step):
        c = min(1.0, step / max(args.total_steps, 1))
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (
            1 + math.cos(math.pi * c))

    t0 = time.time()
    running = {"q": 0.0, "belief": 0.0, "sym": 0.0, "teacher": 0.0,
               "oracle": 0.0, "oracle_abs_ret": 0.0}
    seen = 0
    done = 0
    q_net.train()
    b_net.train()
    if oracle_net is not None:
        oracle_net.train()
    epoch = 0
    while done < args.steps:
        order = rng.permutation(train_idx)
        # A fresh relabelling per deal per epoch: one pass sees one view of
        # each deal, the next pass a different one. Without this the run
        # would train on a single fixed permutation, which is a relabelled
        # dataset rather than an augmented one.
        dataset.aug_seed = (args.seed, epoch)
        epoch += 1
        for batch in iter_ply_batches(dataset, order, args.batch_plies,
                                      args.workers):
            t = batch_to_tensors(batch, device)
            if args.mask_new_features:
                mask_new_features(t)
            for g in opt.param_groups:
                g["lr"] = lr_at(step0 + done)
            if oracle_opt is not None:
                for g in oracle_opt.param_groups:
                    g["lr"] = lr_at(step0 + done)
            opt.zero_grad(set_to_none=True)
            if oracle_opt is not None:
                oracle_opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                q = q_forward(q_net, q_cfg.get("arch", "mlp"), t)
                offsets = b_net(t["obs"])
                oracle_q = (oracle_net(t["obs"], t["belief"], t["act"])
                            if oracle_net is not None else None)
                if oracle_target is not None:
                    with torch.no_grad():
                        q_star = oracle_target(t["obs"], t["belief"],
                                               t["act"])
                else:
                    q_star = None
                if args.sym_consistency or sym_teacher is not None:
                    original_t = {
                        key: t["orig_" + key]
                        for key in ("obs", "act", "raw", "n_raw",
                                    "glob", "hand")
                    }
                    q_original = q_forward(
                        q_net, q_cfg.get("arch", "mlp"), original_t)
                if sym_teacher is not None:
                    with torch.no_grad():
                        q_teacher = q_forward(
                            sym_teacher, q_cfg.get("arch", "mlp"), t)
            # Forced plies (one legal card) carry no ranking signal: the
            # argmax is decided by the rules, not the net. They still anchor
            # absolute Q calibration, which the MC objective depends on, so
            # the weight is a knob rather than a hard filter -- 1.0 keeps the
            # historical behaviour. Belief always trains on every ply: an
            # observer learns just as much from a forced card as a free one.
            w = torch.where(t["nlegal"] > 1,
                            torch.ones_like(t["ret"]),
                            torch.full_like(t["ret"], args.q_forced_weight))
            own = learner_rows(t, args.learner_team)
            if own is not None:
                w = w * own
                t["belief"] = t["belief"] * own[:, None, None]
            target = (args.oracle_lambda * q_star.float()
                      + (1 - args.oracle_lambda) * t["ret"]
                      if q_star is not None else t["ret"])
            err = F.huber_loss(q.float(), target, reduction="none")
            q_loss = (err * w).sum() / w.sum().clamp_min(1.0)
            b_loss = belief_nll(offsets, t["obs"], t["belief"])
            oracle_loss = (
                F.huber_loss(oracle_q.float(), t["ret"])
                if oracle_q is not None else q_loss.new_zeros(()))
            oracle_abs_ret = (
                (q_star.float() - t["ret"]).abs().mean()
                if q_star is not None else q_loss.new_zeros(()))
            sym_loss = (F.mse_loss(q.float(), q_original.float())
                        if args.sym_consistency else q_loss.new_zeros(()))
            teacher_loss = (
                F.mse_loss(q_original.float(), q_teacher.float())
                if sym_teacher is not None else q_loss.new_zeros(()))
            total = (q_loss + args.c_b * b_loss + oracle_loss
                     + args.sym_consistency * sym_loss
                     + args.sym_teacher_weight * teacher_loss)
            if not torch.isfinite(total):
                print("ERROR non-finite loss — hard stop", flush=True)
                raise SystemExit(3)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            if oracle_opt is not None:
                scaler.unscale_(oracle_opt)
            torch.nn.utils.clip_grad_norm_(
                list(q_net.parameters()) + list(b_net.parameters()), 1.0)
            if oracle_net is not None:
                torch.nn.utils.clip_grad_norm_(oracle_net.parameters(), 1.0)
            scaler.step(opt)
            if oracle_opt is not None:
                scaler.step(oracle_opt)
            scaler.update()
            q_ema.update(q_net)
            b_ema.update(b_net)
            if oracle_ema is not None:
                oracle_ema.update(oracle_net)
            n = len(t["obs"])
            running["q"] += float(q_loss.detach()) * n
            running["belief"] += float(b_loss.detach()) * n
            running["sym"] += float(sym_loss.detach()) * n
            running["teacher"] += float(teacher_loss.detach()) * n
            running["oracle"] += float(oracle_loss.detach()) * n
            running["oracle_abs_ret"] += float(oracle_abs_ret.detach()) * n
            seen += n
            done += 1
            if done % 200 == 0:
                print(f"  step {done}/{args.steps} "
                      f"q={running['q'] / max(seen, 1):.4f} "
                      f"belief={running['belief'] / max(seen, 1):.4f} "
                      f"sym={running['sym'] / max(seen, 1):.5f} "
                      f"teacher={running['teacher'] / max(seen, 1):.5f} "
                      f"oracle={running['oracle'] / max(seen, 1):.4f} "
                      f"lr={lr_at(step0 + done):.2e}", flush=True)
            if done >= args.steps:
                break

    val = validate(q_ema, b_ema, q_cfg, b_cfg, val_dataset, val_idx,
                   device, learner_team=args.learner_team)
    oracle_for_validation = None
    if oracle_ema is not None:
        oracle_for_validation = QOracle(hidden=oracle_cfg["hidden"],
                                        blocks=oracle_cfg["blocks"]).to(device)
        oracle_for_validation.load_state_dict(oracle_ema.shadow)
    elif oracle_target is not None:
        oracle_for_validation = oracle_target
    if oracle_for_validation is not None:
        val.update(validate_oracle(oracle_for_validation, val_dataset,
                                   val_idx, device))
    elapsed = time.time() - t0
    print(f"train done in {elapsed:.0f}s "
          f"({done / max(elapsed, 1e-9):.1f} steps/s): "
          f"q={running['q'] / max(seen, 1):.4f} "
          f"belief={running['belief'] / max(seen, 1):.4f} "
          f"oracle={running['oracle'] / max(seen, 1):.4f}", flush=True)
    oracle_summary = (f" oracle={val['oracle_mean']:.4f}"
                      if "oracle_mean" in val else "")
    print(f"val q={val['q_mean']:.4f} belief={val['belief_mean']:.4f} "
          f"(baseline {val['belief_base_mean']:.4f}){oracle_summary}",
          flush=True)
    print("VAL " + json.dumps({
        "steps": done, "global_step": step0 + done, "elapsed": elapsed,
        # The rate actually applied: the cosine is over GLOBAL steps, so a
        # resumed run's rate is not readable from its CLI alone.
        "lr_last": lr_at(step0 + done - 1) if done else None,
        "train_q": running["q"] / max(seen, 1),
        "train_belief": running["belief"] / max(seen, 1),
        "train_sym": running["sym"] / max(seen, 1),
        "train_sym_teacher": running["teacher"] / max(seen, 1),
        "train_oracle": running["oracle"] / max(seen, 1),
        "train_oracle_abs_ret": running["oracle_abs_ret"] / max(seen, 1),
        **val,
    }), flush=True)

    save_checkpoint(args.out_q, q_net, q_ema.shadow, opt, scaler,
                    step0 + done, q_cfg)
    save_checkpoint(args.out_b, b_net, b_ema.shadow, None, None,
                    step0 + done, b_cfg)
    print(f"saved {args.out_q} and {args.out_b}", flush=True)
    if oracle_net is not None:
        save_checkpoint(args.oracle_out, oracle_net, oracle_ema.shadow,
                        oracle_opt, scaler, step0 + done, oracle_cfg)
        print(f"saved {args.oracle_out}", flush=True)

    if args.choice_out:
        train_choice(args, deals, train_idx, val_idx, device, rng)


if __name__ == "__main__":
    main()

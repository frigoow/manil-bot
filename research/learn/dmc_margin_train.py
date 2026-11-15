"""M2 deal-outcome distribution net (manille_match_plan.md §4a, gate M2).

MarginNet: (hand multi-hot [32], declaration one-hot [5], seat-relative-to-
declarer one-hot [4]) -> 61 logits over signed margins [-30..+30] from the
holder's OWN team's perspective (index 30 = the 30-30 tie).

The `rel` input is what makes one net serve all three decisions. D1 is the
dealer choosing among 5 declarations, so it queries rel=0. But D2 (meegaan) is
called by a DEFENDER, who sees their own eight cards and not the declarer's --
a different conditioning entirely, answered by the same net at rel=1 or 3. D3
(terug meegaan) queries rel=0 or 2. Training all four seats jointly also lets
the card representation share statistics across ~4x the rows at no extra
rollout cost (see learn/dmc_margin_expand.py).

Output maps straight onto the equity solver's OutcomeDist: p_tie = P(0),
p_we_score[m] = P(+m), p_they_score[m] = P(-m).

Plain CE (the MLE for the per-(hand, declaration, rel) categorical), split by
HAND GROUP: every row descended from the same original deal -- all seats, all
5 declarations, all rollouts -- stays on one side of the split. Anything
looser leaks, because the four seats of a rollout are complementary.

  PYTHONPATH=. venv/bin/python -m learn.dmc_margin_train \
      --data data/margin/round_001_seats --out runs/margin_net001.pt
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dmc_margin_expand import SEAT_DTYPE

N_BINS = 61  # signed margins -30..+30; index 30 = tie
N_DECL = 5
N_REL = 4


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class MarginNet(nn.Module):
    # 128x1 is deliberate and measured: on 5000 hands x 20 rollouts, 512x3
    # overfits straight past the no-cards baseline (val NLL 3.795 at 6k steps,
    # 4.044 at 12k, baseline 3.909) while 128x1 reaches 3.758 with NO
    # train/val gap. Distinct HANDS are the binding resource here, not
    # capacity and not rollouts -- at equal row count, 4500 hands x 5
    # rollouts (6.008) beats 1125 hands x 20 (6.717). Revisit the size only
    # after a generation round with many more hands.
    #
    # cond (M5): add the 2-value "they announced" input -- whether either
    # defender's hand in this rollout would have called meegaan (computed
    # post-hoc by decision_layer.label_meegaan). It is what lets D3 react to
    # the announcement instead of reading an unconditioned distribution.
    def __init__(self, hidden=128, blocks=1, cond=False):
        super().__init__()
        self.cond = cond
        self.input = nn.Linear(32 + N_DECL + N_REL + (2 if cond else 0),
                               hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(blocks))
        self.head = nn.Linear(hidden, N_BINS)

    def forward(self, hand, decl, rel, ann=None):
        x = [hand, decl, rel]
        if self.cond:
            if ann is None:
                ann = hand.new_zeros((hand.shape[0], 2))
                ann[:, 0] = 1.0
            x.append(ann)
        z = torch.relu(self.input(torch.cat(x, dim=-1)))
        for block in self.blocks:
            z = block(z)
        return self.head(z)


def load_margin_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = MarginNet(**ck["config"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model.state_dict())
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                 alpha=1 - self.decay)


def load_rows(dirs):
    rows = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("shard_") and f.endswith(".bin"):
                rows.append(np.fromfile(os.path.join(d, f), dtype=SEAT_DTYPE))
    return np.concatenate(rows)


def rows_to_tensors(rows, rollouts, labels=None):
    """(hand multi-hot, decl one-hot, rel one-hot, margin index, group id)
    plus, when `labels` is given, the "they announced" one-hot (M5).

    The group is the original hand index, recovered from `rid`: rid counts
    rollouts within (hand, declaration), so rid // (5 * rollouts) is the deal
    every row in the group descends from."""
    hands = ((rows["hand"][:, None] >> np.arange(32, dtype=np.uint64)[None])
             & 1).astype(np.float32)
    decls = np.eye(N_DECL, dtype=np.float32)[rows["declaration"]]
    rels = np.eye(N_REL, dtype=np.float32)[rows["rel"]]
    targets = rows["margin"].astype(np.int64) + 30
    groups = rows["rid"] // (N_DECL * rollouts)
    anns = (None if labels is None
            else np.eye(2, dtype=np.float32)[labels.astype(np.int64)])
    return hands, decls, rels, targets, groups, anns


def on_policy_mask(rows, rollouts, declarations=None):
    """Which rows to train on so the net is conditioned the way play is.

    D1 needs COUNTERFACTUALS -- "what would clubs be worth on this hand" for
    all five declarations -- so every declarer row (rel=0) is kept. D2 and D3
    need the opposite: a defender sees a declaration that was CHOSEN, and the
    choice is itself evidence about the declarer's hand. Training those rows
    on all five declarations teaches the net that a declaration says nothing,
    which shows up downstream as defenders who are systematically optimistic
    and an equity table that sits above 0.5 for both sides at 0-0.

    So non-declarer rows are kept only at the declaration the chooser would
    actually make. `declarations` is one index per hand group (the chooser's
    pick); None falls back to the C++ trump rule, which never picks mull --
    fine for bootstrapping V, but it leaves no mull defender rows, so meegaan
    against a mull must wait for a chooser-derived pass."""
    groups = rows["rid"] // (N_DECL * rollouts)
    if declarations is None:
        import manille_ismcts as mi
        decl_rows = rows[rows["rel"] == 0]
        dg = decl_rows["rid"] // (N_DECL * rollouts)
        hand_of = {}
        for g, h in zip(dg, decl_rows["hand"]):
            hand_of.setdefault(int(g), int(h))
        n_groups = int(groups.max()) + 1
        declarations = np.zeros(n_groups, dtype=np.int64)
        for g in range(n_groups):
            declarations[g] = int(mi.apply_trump_rule(hand_of[g]))
    declarations = np.asarray(declarations)
    return (rows["rel"] == 0) | (rows["declaration"] == declarations[groups])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollouts", type=int, default=20,
                    help="rollouts per (hand, declaration) in the source "
                         "shards — sets the hand-group size")
    ap.add_argument("--val-groups-frac", type=float, default=0.1)
    ap.add_argument("--on-policy", action="store_true",
                    help="keep non-declarer rows only at the declaration the "
                         "chooser would make (see on_policy_mask) — required "
                         "for D2/D3 to be conditioned the way play is")
    ap.add_argument("--declarations", default=None,
                    help=".npy of one declaration index per hand group; "
                         "omit to use the C++ trump rule")
    ap.add_argument("--meegaan-labels", default=None,
                    help="M5: uint8 .npy, one 'they announced' bit per RAW "
                         "shard row (decision_layer.label_meegaan output), "
                         "aligned with the shard BEFORE the on-policy "
                         "filter; the filter is applied to it identically")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = load_rows(args.data)
    labels = (np.load(args.meegaan_labels) if args.meegaan_labels else None)
    if labels is not None and len(labels) != len(raw):
        raise SystemExit(f"labels {len(labels)} != raw rows {len(raw)} — "
                         "labels must be aligned with the shard before the "
                         "on-policy filter")
    if args.on_policy:
        decl_src = (np.load(args.declarations) if args.declarations else None)
        keep = on_policy_mask(raw, args.rollouts, decl_src)
        print(f"on-policy filter: {keep.sum()}/{len(keep)} rows "
              f"({keep[raw['rel'] != 0].mean():.1%} of non-declarer rows kept)",
              flush=True)
        raw = raw[keep]
        if labels is not None:
            labels = labels[keep]
    hands, decls, rels, targets, groups, anns = rows_to_tensors(
        raw, args.rollouts, labels)
    uniq = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    val_groups = rng.choice(uniq, size=max(1, int(
        len(uniq) * args.val_groups_frac)), replace=False)
    val_mask = np.isin(groups, val_groups)
    print(f"rows {len(targets)} ({(~val_mask).sum()} train / "
          f"{val_mask.sum()} val, {len(uniq)} hand groups)", flush=True)

    cond = anns is not None
    net = MarginNet(args.hidden, args.blocks, cond=cond).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(net)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)

    H = torch.from_numpy(hands).to(device)
    D = torch.from_numpy(decls).to(device)
    R = torch.from_numpy(rels).to(device)
    T = torch.from_numpy(targets).to(device)
    A = torch.from_numpy(anns).to(device) if cond else None
    train_idx = torch.from_numpy(np.where(~val_mask)[0]).to(device)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = train_idx[torch.randint(0, len(train_idx), (args.batch,),
                                      device=device)]
        loss = F.cross_entropy(
            net(H[idx], D[idx], R[idx], A[idx] if cond else None), T[idx])
        if not torch.isfinite(loss):
            print("ERROR non-finite loss — hard stop", flush=True)
            raise SystemExit(3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema.update(net)
        sched.step()
        if step % 1000 == 0:
            print(f"  step {step}/{args.steps} ce={float(loss):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # EMA-weight validation. The number that matters is NLL against the
    # per-(declaration, rel) empirical histogram of the TRAIN rows: that
    # baseline knows the declaration and the seat but nothing about the
    # cards, so beating it is exactly "the net reads the hand".
    val_net = MarginNet(args.hidden, args.blocks, cond=cond).to(device)
    val_net.load_state_dict(ema.shadow)
    val_net.eval()
    report = {}
    with torch.no_grad():
        vi = torch.from_numpy(np.where(val_mask)[0]).to(device)
        ti = torch.from_numpy(np.where(~val_mask)[0]).to(device)
        logits = val_net(H[vi], D[vi], R[vi], A[vi] if cond else None)
        nll = float(F.cross_entropy(logits, T[vi]))
        p = torch.softmax(logits, dim=-1)
        bins = torch.arange(-30, 31, device=device, dtype=torch.float32)
        pred_mean = (p * bins).sum(-1)
        true_mean = (T[vi] - 30).float()

        d_val = torch.from_numpy(raw["declaration"][val_mask]).to(device)
        r_val = torch.from_numpy(raw["rel"][val_mask]).to(device)
        d_tr = torch.from_numpy(raw["declaration"][~val_mask]).to(device)
        r_tr = torch.from_numpy(raw["rel"][~val_mask]).to(device)
        base_terms, cells = [], {}
        for d in range(N_DECL):
            for rl in range(N_REL):
                m = (d_val == d) & (r_val == rl)
                if not bool(m.any()):
                    continue
                hist = torch.bincount(T[ti][(d_tr == d) & (r_tr == rl)],
                                      minlength=N_BINS).float()
                hist = (hist + 1.0) / (hist.sum() + N_BINS)
                base_terms.append(-torch.log(hist[T[vi][m]]))
                cells[f"{d}/{rl}"] = {
                    "n": int(m.sum()),
                    "nll": float(F.cross_entropy(logits[m], T[vi][m])),
                    "base_nll": float(-torch.log(hist[T[vi][m]]).mean()),
                    "pred_mean": float(pred_mean[m].mean()),
                    "true_mean": float(true_mean[m].mean())}
        report = {"nll": nll,
                  "baseline_nll": float(torch.cat(base_terms).mean()),
                  "cells": cells,
                  "steps": args.steps,
                  "elapsed": time.time() - t0}
    print("VAL " + json.dumps(report), flush=True)
    torch.save({"model": ema.shadow,
                "config": {"hidden": args.hidden, "blocks": args.blocks,
                           "cond": cond},
                "val_nll": nll, "report": report}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

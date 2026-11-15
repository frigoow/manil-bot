"""Trump choice as a learned decision (Phase 2.0 / Phase 6 machinery).

Today the trump suit is a fixed heuristic (`apply_trump_rule`: longest
suit, ties → points → lowest index). Phase 6 makes it the first decision
of the hand, chosen by a network. ChoiceVNet is the action-as-input value
for that decision: V(hand, suit) -> expected final team points of the
choosing team, normalized (points - 30) / 30 — same DMC return target as
the play Q, evaluated over 4 suit candidates.

Bootstrap (this module's trainer): fit ChoiceVNet on the EXISTING v2
replay buffer — the deals' trumps were rule-chosen, but the returns are
real, so the net learns "value of the rule's choice". It has three uses:
warm start for the free-choice DMC phase, the regression artifact for the
sampler's soft choice-likelihood (softmax(V/temp) with small temp ≈ the
hard rule constraint it replaces), and a sanity read on how much the
choice matters at all (margin between best and second-best suit).

    PYTHONPATH=. venv/bin/python -m learn.dmc_trump_choice \
        --deals 200000 --steps 2000 --out runs/trump_choice_boot.pt
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manille_ismcts as mi
from .dmc_replay import load_dmc_deals
from .model import ResidualBlock

CTX_DIM = 32 + 4  # own hand one-hot + candidate suit one-hot


class ChoicePolicyNet(nn.Module):
    """pi(suit | hand) — 4 logits. Supervised mimic of apply_trump_rule:
    serves as the rule-faithful regression artifact for the sampler's
    soft choice-likelihood (softmax(logits/temp) ≈ hard constraint) and as
    the sane initialization for choice-DMC (starts rule-like)."""

    def __init__(self, hidden=256, blocks=2):
        super().__init__()
        self.input = nn.Linear(32, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(blocks))
        self.head = nn.Linear(hidden, 4)

    def forward(self, hand):
        z = torch.relu(self.input(hand))
        for block in self.blocks:
            z = block(z)
        return self.head(z)


class ChoiceVNet(nn.Module):
    """V(hand, suit) -> normalized expected team points. Argmax over the
    four suit candidates is the trump choice; softmax over the four
    values is the choice-likelihood for world weighting."""

    def __init__(self, hidden=256, blocks=2):
        super().__init__()
        self.input = nn.Linear(CTX_DIM, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(blocks))
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, hand, suit):
        z = torch.relu(self.input(torch.cat([hand, suit], dim=-1)))
        for block in self.blocks:
            z = block(z)
        return self.head(z).squeeze(-1)


def hand_suit_rows(deals):
    """Per-deal (hand[32], suit-candidates[4,4], target) arrays."""
    n = len(deals)
    hands = np.zeros((n, 32), dtype=np.float32)
    targets = np.zeros(n, dtype=np.float32)
    suits = np.eye(4, dtype=np.float32)
    for i, deal in enumerate(deals):
        mask = int(deal["hands"][int(deal["declarer"])])
        for c in range(32):
            if mask & (1 << c):
                hands[i, c] = 1.0
        team = int(deal["declarer"]) % 2
        pts = int(deal["pts0"]) if team == 0 else int(deal["pts1"])
        targets[i] = (pts - 30.0) / 30.0
    return hands, suits, targets


@torch.no_grad()
def mimic_accuracy(model, hands, rule_picks, device, batch=8192):
    """Fraction of held-out hands where the policy's argmax equals the
    deterministic rule pick."""
    agree = 0
    for lo in range(0, len(hands), batch):
        h = torch.from_numpy(hands[lo:lo + batch]).to(device)
        pick = model(h).argmax(dim=1).cpu().numpy()
        agree += int((pick == rule_picks[lo:lo + batch]).sum())
    return agree / len(hands)


@torch.no_grad()
def evaluate(model, hands, suits, targets, rule_picks, device, batch=8192):
    errs, margins, agree = [], [], 0
    for lo in range(0, len(hands), batch):
        h = torch.from_numpy(hands[lo:lo + batch]).to(device)
        s = torch.from_numpy(
            np.repeat(suits[None, :, :], h.shape[0], axis=0)).to(device)
        v = model(h.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 32),
                  s.reshape(-1, 4)).reshape(-1, 4)
        t = torch.from_numpy(targets[lo:lo + batch]).to(device)
        pick = v.argmax(dim=1)
        errs.append(F.huber_loss(
            v.gather(1, torch.from_numpy(rule_picks[lo:lo + batch]).to(
                device)[:, None]).squeeze(1), t, reduction="none"))
        top2 = v.topk(2, dim=1).values
        margins.append((top2[:, 0] - top2[:, 1]))
        agree += int((pick == torch.from_numpy(
            rule_picks[lo:lo + batch]).to(device)).sum())
    err = torch.cat(errs).mean().item()
    margin = torch.cat(margins).mean().item()
    return err, margin, agree / len(hands)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", default=None)
    ap.add_argument("--deals", type=int, default=200000)
    ap.add_argument("--val-deals", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/trump_choice_boot.pt")
    ap.add_argument("--objective", choices=["returns", "mimic"],
                    default="returns",
                    help="returns: fit V(hand, suit) to MC returns (weak "
                         "signal at 200k deals, measured). mimic: supervised "
                         "imitation of apply_trump_rule — the rule-faithful "
                         "regression artifact")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data is None:
        import glob as _glob
        args.data = sorted(_glob.glob("data/dmc/cycle_*"))
    deals = load_dmc_deals(args.data)
    if len(deals) > args.deals:
        deals = deals[-args.deals:]
    print(f"buffer: {len(deals)} deals", flush=True)

    hands, suits, targets = hand_suit_rows(deals)
    rule_picks = np.empty(len(deals), dtype=np.int64)
    for i, deal in enumerate(deals):
        hand_mask = np.int64(int(deal["hands"][int(deal["declarer"])]))
        rule_picks[i] = int(mi.apply_trump_rule(hand_mask))
    n_val = args.val_deals
    vh, vt, vr = hands[-n_val:], targets[-n_val:], rule_picks[-n_val:]
    hands, targets, rule_picks = (hands[:-n_val], targets[:-n_val],
                                  rule_picks[:-n_val])

    if args.objective == "mimic":
        model = ChoicePolicyNet().to(device)
        kind = "policy"
    else:
        model = ChoiceVNet().to(device)
        kind = "value"
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    def lr_at(step):
        c = min(1.0, step / max(args.steps, 1))
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (
            1 + math.cos(math.pi * c))

    t0 = time.time()
    n = len(hands)
    suits_t = torch.from_numpy(suits).to(device)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, args.batch)
        h = torch.from_numpy(hands[idx]).to(device)
        t = torch.from_numpy(targets[idx]).to(device)
        r = torch.from_numpy(rule_picks[idx]).to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        if args.objective == "mimic":
            loss = F.cross_entropy(model(h), r)
        else:
            v = model(
                h.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 32),
                suits_t.unsqueeze(0).expand(h.shape[0], -1, -1)
                .reshape(-1, 4),
            ).reshape(-1, 4)
            loss = F.huber_loss(v.gather(1, r[:, None]).squeeze(1), t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            if args.objective == "mimic":
                acc = mimic_accuracy(model, vh, vr, device)
                print(f"  step {step}: train {float(loss):.4f}  "
                      f"val rule-acc {acc * 100:.1f}%", flush=True)
            else:
                err, margin, agree = evaluate(model, vh, suits, vt, vr,
                                              device)
                print(f"  step {step}: train {float(loss):.4f}  "
                      f"val huber {err:.4f}  margin {margin:+.3f}  "
                      f"rule-agree {agree * 100:.1f}%", flush=True)

    if args.objective == "mimic":
        acc = mimic_accuracy(model, vh, vr, device)
        print(f"final ({time.time() - t0:.0f}s): val rule-acc "
              f"{acc * 100:.1f}%")
        err, margin, agree = 0.0, 0.0, acc
    else:
        err, margin, agree = evaluate(model, vh, suits, vt, vr, device)
        print(f"final ({time.time() - t0:.0f}s): val huber {err:.4f}  "
              f"margin {margin:+.3f}  rule-agree {agree * 100:.1f}%")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"kind": kind, "hidden": 256, "blocks": 2},
                "val_huber": err, "rule_agree": agree}, args.out)
    print(f"saved {args.out}")


def load_choice_model(path, device="cuda"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state.get("config", {"kind": "value", "hidden": 256, "blocks": 2})
    cls = ChoicePolicyNet if cfg.get("kind") == "policy" else ChoiceVNet
    model = cls(**{k: v for k, v in cfg.items() if k != "kind"})
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def suit_values(choice_model, hand_mask, device):
    """Suit scores (value or logits) for the four suits, from an int64
    hand bitmask."""
    h = torch.zeros(1, 32, device=device)
    for c in range(32):
        if int(hand_mask) & (1 << c):
            h[0, c] = 1.0
    if isinstance(choice_model, ChoicePolicyNet):
        return choice_model(h)[0].cpu().numpy()
    hands = h.expand(4, 32)
    suits = torch.eye(4, device=device)
    return choice_model(hands, suits).cpu().numpy()  # [4]


def choose_trump(choice_model, hand_mask, device, eps, rng):
    """The trump decision: argmax over suit values, or a uniform random
    suit with probability eps (exploration for the choice DMC stream)."""
    if eps > 0.0 and rng.random() < eps:
        return int(rng.integers(4))
    return int(np.argmax(suit_values(choice_model, hand_mask, device)))


@torch.no_grad()
def choice_log_likelihood(choice_model, hand_masks, observed_suit, device,
                          temp=0.1):
    """log P(observed_suit | hand) per hand: softmax over suit values at
    temperature temp. The world-weighting term that replaces the hard
    trump-rule constraint once trump is net-chosen (with the rule-
    mimicking bootstrap and small temp, ≈ the hard constraint)."""
    n = len(hand_masks)
    hands = torch.zeros(n, 32, device=device)
    for i, mask in enumerate(hand_masks):
        for c in range(32):
            if int(mask) & (1 << c):
                hands[i, c] = 1.0
    if isinstance(choice_model, ChoicePolicyNet):
        v = choice_model(hands) / max(temp, 1e-6)
    else:
        suits = torch.eye(4, device=device).unsqueeze(0).expand(n, 4, 4)
        v = choice_model(
            hands.unsqueeze(1).expand(n, 4, 32).reshape(-1, 32),
            suits.reshape(-1, 4),
        ).reshape(n, 4) / max(temp, 1e-6)
    logp = torch.log_softmax(v, dim=-1)
    return logp[:, int(observed_suit)].cpu().numpy()


if __name__ == "__main__":
    main()

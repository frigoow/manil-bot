"""Train BeliefGRUv2 (drop-in recurrent belief network for Manille).

Uses the exact [B, OBS_DIM] observation contract of BeliefNet (and the app's
manille_belief_v1.onnx / manille_belief_mull_v1.onnx), replacing the flat
512x2 ResNet MLP with a GRU over the 31-ply public history + residual head
over the full static/belief/tactical features.

Example:
    python -m learn.train_belief_gru \
        --data /path/to/fast_policy/suits /path/to/decl_suits/cycle_000 \
        --baseline artifacts/dmc_belief_best.pt \
        --out runs/belief_gru_suit_v2.pt \
        --epochs 12 --batch-deals 128
"""
import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import manille_ismcts as mi
from manille_env.card import permute_card_index, permute_hand_mask

from .dmc_model import BeliefGRUv2, load_belief_model
from .dmc_replay import load_dmc_deals
from .encoding import OBS_DIM
from .model import belief_probs, split_belief_inputs

ALL_SUIT_PERMS = list(itertools.permutations(range(4)))
# Precompute 24 x 32 card permutation LUTs for fast vectorized augmentation
PERM_CARD_LUT = np.array(
    [[permute_card_index(c, p) for c in range(32)] for p in ALL_SUIT_PERMS],
    dtype=np.int32,
)


def _encode_deal_belief(deal, perm_idx=None):
    """Encode one deal into (obs [32, OBS_DIM], tgt [32, 32] in {-1,0,1,2})."""
    hands = deal["hands"].astype(np.int64).copy()
    trump = int(deal["trump"])
    declarer = int(deal["declarer"])
    cards = deal["plies"]["card"].astype(np.int32).copy()

    if perm_idx is not None:
        p = ALL_SUIT_PERMS[perm_idx]
        for s in range(4):
            hands[s] = permute_hand_mask(int(hands[s]), p)
        if trump >= 0:
            trump = p[trump]
        cards = PERM_CARD_LUT[perm_idx][cards]

    traj = mi.encode_q_trajectory(
        np.ascontiguousarray(hands, dtype=np.int64),
        trump,
        declarer,
        np.ascontiguousarray(cards, dtype=np.int32),
    )
    obs = traj["obs"]  # [32, OBS_DIM] float32
    players = traj["players"]  # [32] int32

    # True holder per card (0..3)
    holder = np.empty(32, dtype=np.int8)
    for s in range(4):
        m = int(hands[s])
        for c in range(32):
            if (m >> c) & 1:
                holder[c] = s

    # Relative unseen target per ply: -1 if seen (own hand or played), else 0..2
    tgt = np.full((32, 32), -1, dtype=np.int8)
    played_mask = 0
    for t in range(32):
        seat = int(players[t])
        own_mask = int(hands[seat]) & ~played_mask
        seen = own_mask | played_mask
        for c in range(32):
            if not ((seen >> c) & 1):
                tgt[t, c] = (int(holder[c]) - seat) % 4 - 1
        played_mask |= 1 << int(cards[t])
    return obs, tgt


class DealBeliefDataset(Dataset):
    def __init__(self, deals, augment=False, seed=0):
        self.deals = deals
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.deals)

    def __getitem__(self, idx):
        perm_idx = int(np.random.randint(0, 24)) if self.augment else None
        obs, tgt = _encode_deal_belief(self.deals[idx], perm_idx=perm_idx)
        return obs, tgt


def belief_nll_loss(offsets, obs, tgt):
    """Mean per-unseen-card NLL over a batch of plies [N, OBS_DIM], tgt [N, 32]."""
    prior, mask = split_belief_inputs(obs)
    logits = torch.log(prior.clamp_min(1e-9)) + offsets
    logits = logits.masked_fill(mask < 0.5, -1e9)
    log_p = torch.log_softmax(logits, dim=1)  # [N, 3, 32]
    valid = tgt >= 0
    tgt_safe = tgt.clamp_min(0).long().unsqueeze(1)  # [N, 1, 32]
    nll = -log_p.gather(1, tgt_safe).squeeze(1)  # [N, 32]
    return nll[valid].mean(), nll, valid


@torch.no_grad()
def evaluate_val(model, val_obs, val_tgt, device, batch=4096):
    model.eval()
    n = len(val_obs)
    tot_nll = 0.0
    tot_cnt = 0
    tr_nll = np.zeros(8, dtype=np.float64)
    tr_cnt = np.zeros(8, dtype=np.int64)
    tricks = np.tile(np.arange(32) // 4, n // 32)
    for s in range(0, n, batch):
        ob = torch.from_numpy(val_obs[s : s + batch]).to(device, non_blocking=True)
        tg = torch.from_numpy(val_tgt[s : s + batch]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            off = model(ob)
        _, nll, valid = belief_nll_loss(off.float(), ob.float(), tg)
        nll_np = nll.cpu().numpy()
        val_np = valid.cpu().numpy()
        tot_nll += float(nll_np[val_np].sum())
        tot_cnt += int(val_np.sum())
        tr_b = tricks[s : s + len(ob)]
        for t in range(8):
            m = (tr_b == t)[:, None] & val_np
            if m.any():
                tr_nll[t] += float(nll_np[m].sum())
                tr_cnt[t] += int(m.sum())
    per_tr = [tr_nll[t] / max(1, tr_cnt[t]) for t in range(7)]
    return tot_nll / max(1, tot_cnt), per_tr


class BeliefONNXWrapper(nn.Module):
    """Exact manille_belief_v1.onnx contract: obs [B, 2492] -> probs [B, 3, 32]."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, obs):
        prior, mask = split_belief_inputs(obs)
        offsets = self.core(obs)
        return belief_probs(offsets, prior, mask)


def export_onnx(model, out_onnx, sample_obs=None):
    model_cpu = model.cpu().eval()
    wrapper = BeliefONNXWrapper(model_cpu).eval()
    dummy = (torch.from_numpy(sample_obs[:1].copy())
             if sample_obs is not None else torch.zeros(1, OBS_DIM, dtype=torch.float32))
    torch.onnx.export(
        wrapper,
        (dummy,),
        out_onnx,
        input_names=["obs"],
        output_names=["probs"],
        dynamic_axes={"obs": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(out_onnx, so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    got = sess.run(None, {"obs": dummy.numpy()})[0]
    diff = float(np.max(np.abs(ref - got)))
    # Benchmark 1-core single-sample latency
    x1 = dummy[:1].numpy()
    for _ in range(20):
        sess.run(None, {"obs": x1})
    t0 = time.perf_counter()
    iters = 500
    for _ in range(iters):
        sess.run(None, {"obs": x1})
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    sz_mb = os.path.getsize(out_onnx) / (1024 * 1024)
    print(f"Exported {out_onnx} ({sz_mb:.2f} MB), max_diff={diff:.2e}, 1-core={ms:.3f} ms/call",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-onnx", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--d-tok", type=int, default=192)
    ap.add_argument("--d-gru", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--deals-per-epoch", type=int, default=100000)
    ap.add_argument("--val-deals", type=int, default=1024)
    ap.add_argument("--batch-deals", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    deals = load_dmc_deals(args.data)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(deals))
    val_deals = deals[perm[: args.val_deals]]
    tr_deals = deals[perm[args.val_deals :]]
    print(f"Loaded {len(deals)} deals ({len(tr_deals)} train, {len(val_deals)} val) in {time.time()-t0:.1f}s",
          flush=True)

    # Pre-encode validation deals once
    t1 = time.time()
    v_obs, v_tgt = [], []
    for d in val_deals:
        o, tg = _encode_deal_belief(d)
        v_obs.append(o)
        v_tgt.append(tg)
    val_obs = np.concatenate(v_obs, axis=0)
    val_tgt = np.concatenate(v_tgt, axis=0)
    print(f"Encoded {len(val_deals)} val deals ({len(val_obs)} plies) in {time.time()-t1:.1f}s",
          flush=True)

    if args.baseline:
        base_m = load_belief_model(args.baseline, args.device)
        b_ce, b_tr = evaluate_val(base_m, val_obs, val_tgt, args.device)
        print(f"Baseline ({args.baseline}): val_ce={b_ce:.4f}  T1..T7=" +
              " ".join(f"{x:.3f}" for x in b_tr), flush=True)
        del base_m

    torch.manual_seed(args.seed)
    cfg = dict(
        arch="gru_v2", obs_dim=OBS_DIM, d_tok=args.d_tok, d_gru=args.d_gru,
        hidden=args.hidden, blocks=args.blocks,
    )
    model = BeliefGRUv2(
        obs_dim=OBS_DIM, d_tok=args.d_tok, d_gru=args.d_gru,
        hidden=args.hidden, blocks=args.blocks,
    ).to(args.device)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        print(f"Resumed weights from {args.resume} (prev val_ce={ck.get('val_ce')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BeliefGRUv2 params={n_params/1e6:.2f}M  cfg={cfg}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_ep = max(1, min(len(tr_deals), args.deals_per_epoch) // args.batch_deals)
    total_steps = args.epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    best_ce = 999.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    step = 0
    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        n_use = min(len(tr_deals), args.deals_per_epoch)
        sub = tr_deals[rng.choice(len(tr_deals), size=n_use, replace=False)]
        ds = DealBeliefDataset(sub, augment=not args.no_augment, seed=args.seed + ep)
        dl = DataLoader(ds, batch_size=args.batch_deals, shuffle=True,
                        num_workers=args.workers, pin_memory=(args.device == "cuda"),
                        drop_last=True, persistent_workers=False)
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for obs_b, tgt_b in dl:
            # [B, 32, OBS_DIM] -> [B*32, OBS_DIM]
            ob = obs_b.reshape(-1, OBS_DIM).to(args.device, non_blocking=True)
            tg = tgt_b.reshape(-1, 32).to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                off = model(ob)
                loss, _, _ = belief_nll_loss(off.float(), ob.float(), tg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            if step < total_steps:
                sched.step()
                step += 1
            tr_loss_sum += float(loss)
            tr_batches += 1

        val_ce, val_tr = evaluate_val(model, val_obs, val_tgt, args.device)
        mark = ""
        if val_ce < best_ce:
            best_ce = val_ce
            mark = " *BEST*"
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": cfg, "val_ce": val_ce, "epoch": ep}, args.out)
        print(f"ep {ep:2d}/{args.epochs}  tr={tr_loss_sum/max(1,tr_batches):.4f}  "
              f"val={val_ce:.4f}  T1..T7={' '.join(f'{x:.3f}' for x in val_tr)}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({time.time()-t_ep:.1f}s){mark}",
              flush=True)

    print(f"Done. Best val CE = {best_ce:.4f} saved to {args.out}", flush=True)
    if args.export_onnx:
        best = load_belief_model(args.out, "cpu")
        export_onnx(best, args.export_onnx, val_obs)


if __name__ == "__main__":
    main()

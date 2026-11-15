"""Fast learned play-out policy (cpp/fast_policy.h): trainer and loader.

A small MLP imitates a strong player's card (the DMC blueprint's greedy
play, from self-play shards) from the acting seat's own information. The
rule agent's play-out then uses it as the continuation policy for every
simulated seat instead of the hand-written rule policy. Features come from
the C++ extractor (`mi.fast_policy_features`) -- one implementation, used by
training here and by inference in the play-out.

    python -m learn.fast_policy --data data/fast_policy/suits data/fast_policy/mull \\
        --out artifacts/fast_policy.pt --h1 256 --h2 128
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import manille_ismcts as mi

F_DIM = mi.FAST_POLICY_DIM
F_BINARY = mi.FAST_POLICY_BINARY


class FastPolicy(nn.Module):
    def __init__(self, h1=256, h2=128):
        super().__init__()
        self.l1 = nn.Linear(F_DIM, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 32)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


def to_native(model):
    """The C++ net (mi.FastPolicyNet) holding this model's weights."""
    g = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return mi.FastPolicyNet(g(model.l1.weight), g(model.l1.bias), g(model.l2.weight),
                            g(model.l2.bias), g(model.l3.weight), g(model.l3.bias))


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = FastPolicy(ck["config"]["h1"], ck["config"]["h2"])
    model.load_state_dict(ck["model"])
    return model.eval()


def load_native(path):
    return to_native(load_model(path))


def export_binary(path, out):
    """The deploy file cpp/fast_policy.h::load_binary reads (the phone ships
    it as an asset): "MFPN", version, F_DIM, h1, h2, then the float32 weights
    in torch.nn.Linear's [out][in] order."""
    import struct
    model = load_model(path)
    g = lambda t: t.detach().cpu().numpy().astype("<f4").tobytes()
    with open(out, "wb") as f:
        f.write(b"MFPN" + struct.pack("<4i", 1, F_DIM, model.l1.out_features,
                                      model.l2.out_features))
        for layer in (model.l1, model.l2, model.l3):
            f.write(g(layer.weight))
            f.write(g(layer.bias))


def build_dataset(dirs, max_deals=None, seed=0):
    """Choice plies of every deal: packed binary features, dense, legal mask
    bits, the played card, the deal index and the trump (for splits/reports)."""
    from .dmc_replay import load_dmc_deals
    deals = load_dmc_deals(dirs)
    if max_deals is not None and len(deals) > max_deals:
        deals = deals[np.random.default_rng(seed).permutation(len(deals))[:max_deals]]
    xs, ds, ls, ys, gi, tr = [], [], [], [], [], []
    for i, deal in enumerate(deals):
        cards = deal["plies"]["card"].astype(np.int32)
        X, D, L, _ = mi.fast_policy_features(deal["hands"], int(deal["trump"]),
                                             int(deal["declarer"]), cards)
        legal = ((L[:, None] >> np.arange(32)[None, :]) & 1).astype(bool)
        keep = legal.sum(axis=1) > 1
        xs.append(np.packbits(X[keep], axis=1))
        ds.append(D[keep])
        ls.append(np.packbits(legal[keep], axis=1))
        ys.append(cards[keep].astype(np.int64))
        gi.append(np.full(int(keep.sum()), i, dtype=np.int32))
        tr.append(np.full(int(keep.sum()), int(deal["trump"]), dtype=np.int8))
    cat = np.concatenate
    return dict(x=cat(xs), d=cat(ds), l=cat(ls), y=cat(ys), deal=cat(gi), trump=cat(tr),
                n_deals=len(deals))


def batch_tensors(data, idx, device):
    x = np.unpackbits(data["x"][idx], axis=1, count=F_BINARY)
    x = np.concatenate([x.astype(np.float32), data["d"][idx]], axis=1)
    legal = np.unpackbits(data["l"][idx], axis=1, count=32).astype(bool)
    return (torch.from_numpy(x).to(device), torch.from_numpy(legal).to(device),
            torch.from_numpy(data["y"][idx]).to(device))


def evaluate(model, data, idx, device, batch=16384):
    model.eval()
    hit = np.zeros(len(idx), dtype=bool)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            x, legal, y = batch_tensors(data, j, device)
            logits = model(x).masked_fill(~legal, -1e9)
            hit[s:s + batch] = (logits.argmax(dim=1) == y).cpu().numpy()
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-deals", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    data = build_dataset(args.data, args.max_deals, args.seed)
    n = len(data["y"])
    print(f"{data['n_deals']} deals, {n} choice plies ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(args.seed)
    val_deals = rng.random(data["n_deals"]) < args.val_frac  # split by deal
    is_val = val_deals[data["deal"]]
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]

    torch.manual_seed(args.seed)
    model = FastPolicy(args.h1, args.h2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * (len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05)
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        tot, cnt = 0.0, 0
        for s in range(0, len(perm) - args.batch + 1, args.batch):
            x, legal, y = batch_tensors(data, np.sort(perm[s:s + args.batch]), args.device)
            loss = nn.functional.cross_entropy(model(x).masked_fill(~legal, -1e9), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < steps:
                sched.step()
                step += 1
            tot += float(loss)
            cnt += 1
        hit = evaluate(model, data, va_idx, args.device)
        tv = data["trump"][va_idx]
        print(f"epoch {ep + 1}: loss {tot / max(cnt, 1):.4f}  val acc {hit.mean():.4f}  "
              f"suit {hit[tv >= 0].mean() if (tv >= 0).any() else float('nan'):.4f}  "
              f"mull {hit[tv < 0].mean() if (tv < 0).any() else float('nan'):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {"h1": args.h1, "h2": args.h2, "f_dim": F_DIM},
                "val_acc": float(hit.mean())}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

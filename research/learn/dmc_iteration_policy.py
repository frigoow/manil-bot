"""Policy contract for paired-advantage policy iteration
(manille_policy_iteration_plan.md §5.1, §8.3).

One scorer interface serves live evaluation, trajectory play and every
branch continuation, so the policy that generates labels is by construction
the policy that is evaluated. It sees only canonical player-visible tensors
(the C++ q-observation and candidate features); it is never handed a hand.

Two adapters share one implementation because they share one topology:

  raw_q  the frozen DMC QGRU blueprint. Its bounded scorer output IS Q, so
         argmax reproduces QNetAgent exactly, and logit_scale * Q = 15 Q is
         the round-0 training anchor z_0 = 30 Q / 2 of plan §8.1.
  actor  a standalone QGRU-topology actor (checkpoint kind
         `dmc_iteration_actor`). Same bounded scorer, same stored logit
         scale 15, so round 0 is exactly 15 Q_0 — but its output is a policy
         logit, not calibrated expected points, and `load_q_model` refuses
         the checkpoint so nothing can mistake it for a Q evaluator.

Greedy choice is argmax of the UNSCALED scorer output with first-index tie
breaking over the encoder's legal ordering — exactly QNetAgent's rule, so
multiplying by 15 in float32 can never create a tie the raw agent lacks.
"""
import hashlib
import json
import os

import numpy as np
import torch

import manille_ismcts as mi

from tournament import Agent, game_to_cpp_arrays

from .dmc_model import QGRU, load_q_model, make_q_model
from .dmc_tokens import GLOBAL_DIM, MAX_PLAYS, RAW_TOKEN_DIM, tokens_batch

ACTOR_KIND = "dmc_iteration_actor"
LOGIT_SCALE = 15.0
POLICY_CONTRACT_VERSION = 1
# Unique observation rows per trunk forward: bounds the GRU activation
# transient (rows x 31 x d_model) on a GPU shared by several workers.
CHUNK_ROWS = 512


def strict_fp32():
    """Disable TF32 for cuDNN and matmuls in this process.

    PyTorch leaves cuDNN TF32 on by default, which silently runs the GRU
    at ~1e-3 relative precision: measured 4e-4 max error on the blueprint
    scorer vs a float64 reference (2.4e-7 with TF32 off, matching CPU
    fp32). At that noise, greedy near-ties flip with batch composition,
    so labels, audits and evaluation would not share one policy. Every
    policy load goes through here."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def encoder_contract():
    """Dimensions of the installed canonical encoder and token schema.

    Read from the live extension rather than copied from documentation:
    those numbers have gone stale before (plan §3)."""
    return {
        "obs_dim": int(mi.OBS_DIM),
        "act_dim": int(mi.ACT_DIM),
        "raw_token_dim": int(RAW_TOKEN_DIM),
        "global_dim": int(GLOBAL_DIM),
        "max_plays": int(MAX_PLAYS),
        "tokens": "learn.dmc_tokens.tokens_batch",
    }


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _json_hash(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class GRUPolicy:
    """Deterministic greedy policy over a QGRU-topology scorer.

    `score_rows` is the batched primitive; single-state helpers wrap it so
    batched and single inference cannot drift apart in code (numerical
    agreement is still tested, since kernels can differ by batch size)."""

    def __init__(self, model, kind, checkpoint_sha256, device="cpu",
                 logit_scale=LOGIT_SCALE, meta=None, precision="fp32"):
        if not isinstance(model, QGRU):
            # Plan §8.3: a source that does not fit the actor contract fails
            # preflight instead of silently changing architecture.
            raise TypeError(f"policy iteration requires a QGRU, got "
                            f"{type(model).__name__}")
        if kind not in ("raw_q", "actor"):
            raise ValueError(f"unknown policy kind {kind!r}")
        self.model = model
        self.kind = kind
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = device
        self.logit_scale = float(logit_scale)
        self.meta = dict(meta or {})
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unknown precision {precision!r}")
        if precision != "fp32" and not str(device).startswith("cuda"):
            raise ValueError(f"{precision} inference is a CUDA tensor-core "
                             "path")
        self.precision = precision
        fp = {
            "kind": kind,
            "checkpoint_sha256": checkpoint_sha256,
            "logit_scale": self.logit_scale,
            "encoder": encoder_contract(),
            "contract_version": POLICY_CONTRACT_VERSION,
        }
        if precision != "fp32":
            # Only non-default precision enters the hash, so every existing
            # fp32 fingerprint (and dataset manifest) is unchanged.
            fp["precision"] = precision
        self._fingerprint = _json_hash(fp)

    def with_precision(self, precision):
        """Same weights, different inference precision (its own
        fingerprint, so memos and manifests can never mix the two)."""
        return GRUPolicy(self.model, self.kind, self.checkpoint_sha256,
                         self.device, self.logit_scale, self.meta, precision)

    def fingerprint(self):
        """Checkpoint hash + policy kind/config + encoder version. Every
        memo and every shard is keyed on this."""
        return self._fingerprint

    def spec(self):
        return {"kind": self.kind,
                "checkpoint_sha256": self.checkpoint_sha256,
                "logit_scale": self.logit_scale,
                "encoder": encoder_contract(),
                "contract_version": POLICY_CONTRACT_VERSION,
                "fingerprint": self._fingerprint, **self.meta,
                **({"precision": self.precision}
                   if self.precision != "fp32" else {})}

    def score_rows(self, obs, n_plays, act, counts):
        """Scorer output for every legal candidate of U states.

        obs [U, OBS_DIM] canonical q-observations, n_plays [U] plays so far,
        act [sum(counts), ACT_DIM] candidate features grouped per state in
        the encoder's legal order. Returns float32 [sum(counts)]."""
        counts = np.asarray(counts, dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(counts)])
        out = []
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=(torch.float16 if self.precision == "fp16"
                               else torch.bfloat16),
                enabled=self.precision != "fp32"):
            for lo in range(0, len(obs), CHUNK_ROWS):
                hi = min(lo + CHUNK_ROWS, len(obs))
                raw, n_raw, glob, _ = tokens_batch(obs[lo:hi],
                                                   n_plays[lo:hi])
                last = self.model.trunk(
                    torch.from_numpy(raw).to(self.device),
                    torch.from_numpy(n_raw).to(self.device))
                reps = torch.as_tensor(counts[lo:hi], device=self.device)
                q = self.model.head(
                    torch.repeat_interleave(last, reps, dim=0),
                    torch.repeat_interleave(
                        torch.from_numpy(glob).to(self.device), reps, dim=0),
                    torch.from_numpy(np.ascontiguousarray(
                        act[offs[lo]:offs[hi]])).to(self.device))
                out.append(q.float().cpu().numpy())
        out = np.concatenate(out) if out else np.zeros(0, np.float32)
        if self.precision != "fp32" and not np.all(np.isfinite(out)):
            # fp16 has a narrow range; an overflow must fail loudly rather
            # than turn into an arbitrary greedy choice.
            raise FloatingPointError(f"non-finite {self.precision} scores")
        return out

    def score_legal(self, obs, n_plays, act):
        """One state: scores for its n legal candidates (act [n, ACT_DIM])."""
        return self.score_rows(obs[None, :], np.array([n_plays]), act,
                               [len(act)])

    def logits(self, scores):
        return self.logit_scale * np.asarray(scores, dtype=np.float64)

    @staticmethod
    def argmax(scores):
        """Index into the encoder's legal ordering; first max wins, as in
        QNetAgent (np.argmax)."""
        return int(np.argmax(scores))


def load_raw_q_policy(path, device="cpu"):
    """The frozen DMC blueprint as a policy (round-0 source)."""
    strict_fp32()
    model = load_q_model(path, device)
    return GRUPolicy(model, "raw_q", sha256_file(path), device=device,
                     meta={"path": os.path.abspath(path)})


def actor_config(q_config, logit_scale=LOGIT_SCALE):
    """Actor checkpoint config from its initializing source's geometry."""
    if q_config.get("arch") != "gru":
        raise TypeError("policy-iteration actor requires a QGRU source "
                        f"(arch={q_config.get('arch')!r})")
    return {
        "kind": ACTOR_KIND,
        "arch": "gru",
        "d_model": q_config.get("d_model", 256),
        "layers": q_config.get("layers", 1),
        "hidden": q_config.get("hidden", 1024),
        "blocks": q_config.get("blocks", 2),
        "logit_scale": float(logit_scale),
        "encoder": encoder_contract(),
    }


def save_actor(path, model, config, metadata=None, training_state=None):
    """Atomic actor checkpoint write.

    `metadata` carries provenance (source hash, round, dataset hash, eta).
    `training_state` (optimizer/RNG/step) is only for resumption; the
    exported inference artifact omits it (see export_actor)."""
    if config.get("kind") != ACTOR_KIND:
        raise ValueError("actor config must carry kind=" + ACTOR_KIND)
    payload = {"model": {k: v.detach().cpu()
                         for k, v in model.state_dict().items()},
               "config": config,
               "metadata": metadata or {}}
    if training_state is not None:
        payload["training_state"] = training_state
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def export_actor(src, dst):
    """Inference-only artifact: weights + config + provenance metadata."""
    state = torch.load(src, map_location="cpu", weights_only=False)
    if state.get("config", {}).get("kind") != ACTOR_KIND:
        raise ValueError(f"{src} is not a {ACTOR_KIND} checkpoint")
    tmp = dst + ".tmp"
    torch.save({"model": state["model"], "config": state["config"],
                "metadata": state.get("metadata", {})}, tmp)
    os.replace(tmp, dst)


def load_actor_model(path, device="cpu"):
    """(QGRU, config, metadata) for an actor checkpoint; refuses anything
    else. Loads with map_location='cpu' so a CPU deployment never touches
    CUDA."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    if config.get("kind") != ACTOR_KIND:
        raise ValueError(f"{path} is not a {ACTOR_KIND} checkpoint "
                         f"(kind={config.get('kind')!r})")
    want = encoder_contract()
    if config.get("encoder") != want:
        raise ValueError(f"{path}: encoder contract {config.get('encoder')} "
                         f"does not match installed {want}")
    model = make_q_model(config, device="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, config, state.get("metadata", {})


def load_actor_policy(path, device="cpu"):
    strict_fp32()
    model, config, metadata = load_actor_model(path, device)
    return GRUPolicy(model, "actor", sha256_file(path), device=device,
                     logit_scale=config["logit_scale"],
                     meta={"path": os.path.abspath(path),
                           "round": metadata.get("round")})


def load_policy(spec, device="cpu"):
    """Policy from a spec {"kind": "raw_q"|"actor", "path": ...}."""
    strict_fp32()
    if spec["kind"] == "raw_q":
        return load_raw_q_policy(spec["path"], device)
    if spec["kind"] == "actor":
        return load_actor_policy(spec["path"], device)
    raise ValueError(f"unknown policy kind {spec['kind']!r}")


def actor_from_q(q_path, device="cpu"):
    """A fresh actor initialized from a raw-Q checkpoint: bit-identical
    weights, so its logits are exactly 15 Q_0 (plan §8.3)."""
    q_state = torch.load(q_path, map_location="cpu", weights_only=True)
    config = actor_config(q_state.get("config", {}))
    model = load_q_model(q_path, "cpu")
    return model.to(device), config


# --- evaluation-harness agent ------------------------------------------------

class IterationPolicyAgent(Agent):
    """Greedy agent over a GRUPolicy (eval spec type `dmc_iter`)."""

    def __init__(self, policy):
        self.policy = policy
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def scores(self, game, player_id, played_cards, played_players):
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        enc = mi.encode_q_observation(*arrays, self.dealer, player_id)
        n = int(enc["n_legal"])
        n_plays = len(played_cards) + len(game.current_trick)
        return enc["legal"][:n], self.policy.score_legal(
            enc["obs"], n_plays, enc["act"][:n])

    def choose_card(self, game, player_id, played_cards, played_players):
        from manille_env.card import Card

        legal, scores = self.scores(game, player_id, played_cards,
                                    played_players)
        return Card.from_index(int(legal[self.policy.argmax(scores)]))

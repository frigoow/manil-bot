"""Depth-limited SPARTA probe: candidate -> blueprint plays to the end of the
current trick (+ `leaf_depth` more tricks) -> Q-net leaf instead of a full
rollout. Exact minimax leaves at trick >= exact_trick are kept (they are
cheaper than a forward). This is the "shallow Q-leaf SPARTA" item from
SOL_HANDOFF §3.6, built as a probe for the phone-budget question: how much
of the champion's gain survives ~10x fewer network rows per decision?

Leaf value in a world: the seat to move evaluates max_a Q(s_seat, a) on ITS
OWN information state in that world (true holdings of the particle), mapped
to the searching team's points: 30 + 30 q if same team else 30 - 30 q.
"""
import numpy as np

import manille_ismcts as mi
from manille_env.card import Card
from manille_env.game_logic import GamePhase
from tournament import game_to_cpp_arrays

from .dmc_model import QGRU, QTransformer
from .dmc_sparta import SpartaDMCAgent, q_eval_batch
from .sparta import inject_world


def q_leaf_rollouts(worlds, declarer, q_model, device, stop_tricks,
                    exact_leaves=True, exact_trick=6, q_memo=None,
                    leaf_memo=None, memo_stats=None, leaf_seat=None):
    """worlds: list of (game, pc, pp); stop_tricks[k]: trick index at which
    world k is leaf-evaluated. Returns team points [K, 2] (team 0 = seats
    0/2)."""
    final = {}
    active = []
    for k, (g, pc, pp) in enumerate(worlds):
        if g.phase != GamePhase.PLAYING:
            final[k] = np.array(g.hand_points, dtype=np.float64)
        else:
            active.append(k)
    memo = q_memo if q_memo is not None else {}
    while active:
        # 1) exact leaves (cheap at <= 12 cards; memoized per decision)
        if exact_leaves:
            solved = {}
            for k in active:
                game, pc, pp = worlds[k]
                trick = (len(pc) + len(game.current_trick)) // 4
                if trick < exact_trick:
                    continue
                masks = np.zeros(4, dtype=np.int64)
                for p in range(4):
                    for c in game.hands[p]:
                        masks[p] |= np.int64(1) << c.to_index()
                key = (masks.tobytes(), game.current_player,
                       tuple((p, c.to_index()) for p, c in game.current_trick),
                       tuple(game.hand_points))
                if leaf_memo is not None and key in leaf_memo:
                    solved[k] = leaf_memo[key].copy()
                    if memo_stats is not None:
                        memo_stats["leaf_hit"] += 1
                    continue
                arrays = game_to_cpp_arrays(game, pc, pp)
                res = mi.exact_solve(*arrays, declarer, masks)
                vals = res["pts0"]
                v = max(vals) if game.current_player % 2 == 0 else min(vals)
                solved[k] = np.array([v, 60.0 - v])
                if leaf_memo is not None:
                    leaf_memo[key] = solved[k].copy()
                    if memo_stats is not None:
                        memo_stats["leaf_miss"] += 1
            final.update(solved)
            active = [k for k in active if k not in solved]
            if not active:
                break
        # 2) one batched Q forward for every active world (step or leaf)
        rows, metas, keys, need = [], [], [], []
        for k in active:
            game, pc, pp = worlds[k]
            arrays = game_to_cpp_arrays(game, pc, pp)
            enc = mi.encode_q_observation(*arrays, declarer,
                                          game.current_player)
            n = int(enc["n_legal"])
            rows.append((enc["obs"], enc["act"][:n], enc["legal"][:n]))
            metas.append((len(pc) + len(game.current_trick), 0))
            key = enc["obs"].tobytes()
            keys.append(key)
            if key not in memo:
                memo[key] = None
                need.append(len(rows) - 1)
        if memo_stats is not None:
            memo_stats["q_hit"] += len(rows) - len(need)
            memo_stats["q_miss"] += len(need)
        if need:
            counts = [rows[j][1].shape[0] for j in need]
            act = np.concatenate([rows[j][1] for j in need], axis=0)
            if isinstance(q_model, (QTransformer, QGRU)):
                q = q_eval_batch(q_model, device,
                                 np.stack([rows[j][0] for j in need]), act,
                                 token_meta=[metas[j] for j in need],
                                 counts=counts)
            else:
                obs = np.repeat(np.stack([rows[j][0] for j in need]),
                                counts, axis=0)
                q = q_eval_batch(q_model, device, obs, act)
            q = q.float().cpu().numpy()
            pos = 0
            for j, n in zip(need, counts):
                memo[keys[j]] = q[pos:pos + n]
                pos += n
        still = []
        for j, k in enumerate(active):
            game, pc, pp = worlds[k]
            q_j = memo[keys[j]]
            trick = (len(pc) + len(game.current_trick)) // 4
            at_leaf = (trick >= stop_tricks[k] if leaf_seat is None
                       else game.current_player == leaf_seat)
            if at_leaf:
                v = 30.0 + 30.0 * float(np.max(q_j))  # seat-to-move's team
                if game.current_player % 2 == 0:
                    final[k] = np.array([v, 60.0 - v])
                else:
                    final[k] = np.array([60.0 - v, v])
                continue
            card = Card.from_index(int(rows[j][2][int(np.argmax(q_j))]))
            pid = game.current_player
            pre = list(game.current_trick)
            if not game.play_card(pid, card):
                raise ValueError("Q-leaf rollout played an illegal card")
            if len(pre) == 3:
                for tp_, tc_ in pre + [(pid, card)]:
                    pc.append(tc_.to_index())
                    pp.append(tp_)
            if game.phase == GamePhase.PLAYING:
                still.append(k)
            else:
                final[k] = np.array(game.hand_points, dtype=np.float64)
        active = still
    return np.array([final[k] for k in range(len(worlds))], dtype=np.float64)


class SpartaQLeafAgent(SpartaDMCAgent):
    def __init__(self, *args, leaf_depth=0, **kw):
        super().__init__(*args, **kw)
        # leaf_depth: tricks completed after the current one before the Q
        # leaf. 0 = leaf once the current trick resolves; -1 = leaf right
        # after the candidate card; large = full rollout (champion).
        self.leaf_depth = leaf_depth

    def _arm_values(self, game, player_id, played_cards, played_players,
                    cards, particles, my_team, trump, base_seed,
                    seed_offsets, override=None):
        cur_trick = (len(played_cards) + len(game.current_trick)) // 4
        if self.leaf_depth == "me":
            stop = 99  # stop condition is the seat, not the trick
        elif self.leaf_depth >= 0:
            stop = cur_trick + 1 + self.leaf_depth
        else:
            stop = cur_trick  # leaf immediately (same trick index)
        worlds = []
        for card in cards:
            for k in range(len(particles)):
                w_game = inject_world(
                    self.dealer, trump, particles[k], game.current_trick,
                    game.hand_points, game.tricks_won, game.current_player)
                pc_w, pp_w = list(played_cards), list(played_players)
                pre = list(w_game.current_trick)
                if not w_game.play_card(player_id, card):
                    raise ValueError("candidate illegal in a world whose "
                                     "hand matches mine — injector bug")
                if len(pre) == 3:
                    for tp_, tc_ in pre + [(player_id, card)]:
                        pc_w.append(tc_.to_index())
                        pp_w.append(tp_)
                worlds.append((w_game, pc_w, pp_w))
        if self.leaf_depth != "me" and self.leaf_depth < 0:
            # "immediately": the trick may have just resolved -> index moved
            stops = [max(stop, (len(pc) + len(g.current_trick)) // 4)
                     for g, pc, pp in worlds]
            stops = [0] * len(worlds)  # any active world leafs at once
        else:
            stops = [stop] * len(worlds)
        pts = q_leaf_rollouts(worlds, self.dealer, self.q_model, self.device,
                              stops, self.exact_leaves, self.exact_trick,
                              q_memo=self._q_memo, leaf_memo=self._leaf_memo,
                              memo_stats=self.memo_stats,
                              leaf_seat=(player_id if self.leaf_depth == "me"
                                         else None))
        return pts[:, my_team].reshape(len(cards), len(particles))

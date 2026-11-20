"""Rule-based Manille agent: card counting + hand-written heuristics, no search.

All public knowledge comes from the canonical C++ tracker
(`mi.encode_q_observation`): own hand, played cards per seat, established
voids, masters, unseen counts, per-play obligation flags, and per-candidate
"wins now / wins outright / opponent may beat / partner winning" answered by
the engine. On top of that, the agent folds the history flags the encoder
already emits into tighter per-seat caps:

* F_FOLLOWED_NO_BEAT under an in-suit winner -> holds nothing higher there;
* F_DUCKED_TRUMP                             -> holds no higher trump;
* obliged, void, not trumping, no trump on the table -> void in trump.

Play is chosen by scoring each legal card as
    EV(card) = (2·P(team wins trick) − 1)·(points at stake) − keep_value(card)
where keep_value is a heuristic estimate of the card's later worth (masters
and trumps win future tricks; stranded honours leak their points).
"""
import numpy as np

import manille_ismcts as mi
from manille_env.card import Card, Suit, get_legal_moves
from tournament import Agent, game_to_cpp_arrays
from learn import encoding as E

PTS = [0, 0, 0, 1, 2, 3, 4, 5]
RANK_STR = ['7', '8', '9', 'J', 'Q', 'K', 'A', '10']
SUIT_STR = ['D', 'H', 'C', 'S']


def suit_of(c):
    return c >> 3


def rank_of(c):
    return c & 7


def cstr(c):
    return RANK_STR[rank_of(c)] + SUIT_STR[suit_of(c)]


CARDS = [Card.from_index(i) for i in range(32)]
_FAST_NETS = {}  # fast play-out policy nets by path, shared by every agent in the process


def _winner(trick, trump_suit):
    """Winner of a (partial) trick of (seat, Card), via the engine's
    Card.beats."""
    ws, wc = trick[0]
    for p, c in trick[1:]:
        if c.beats(wc, trump_suit):
            ws, wc = p, c
    return ws, wc


def outranks(u, x, trump):
    """Card ordering for *probability estimates only* (would u, played after
    x, beat x when x is the winner). Trick resolution itself always comes from
    the engine via the encoder's action features."""
    if trump is not None and suit_of(u) == trump and suit_of(x) != trump:
        return True
    return suit_of(u) == suit_of(x) and rank_of(u) > rank_of(x)


DEFAULTS = dict(
    trick_rest=1.5,        # expected points each not-yet-played card adds
    keep_win_value=7.0,    # value of a future trick win carried by a card
    keep_pts_weight=0.7,   # how much a stranded honour's points count later
    pull_trumps=True,
    pull_min_trumps=2,
    pull_min_trumps_def=3,
    cash_threshold=0.15,   # P(safe) needed to cash a master
    lead_pts_w=1.0,
    guard_w=0.0,
    sim=True,
    bare_honour_lead_pen=0.0,
    planner=True,
    signal_mode=0,             # discard signals: 0 off, 1 direct, 2 opposite colour
    signal_final=True,         # the play-out may not overrule a signal play
    signal_ruff=True,          # (opposite) also ask for a suit I can ruff
    signal_max_pts=1,          # biggest card that still counts as "small"
    signal_cheap=False,        # only signal when nothing is worth vetting
    signal_read='lead',        # 'lead' = lead the asked suit, 'infer' = reweight worlds
    signal_boost=3.0,
    plan_est_v2=False,
    nt_est_low_run=False,      # no trump: establish from a touching run with its lowest card
    nt_lock_cash_from_trick=0, # no trump, from this trick on (0 = off): the play-out may only
                               # swap a "cash master" lead for another master
    def_trump_lock=False,      # defending a suit: the play-out may not turn a side-suit lead
                               # into a trump lead (user feedback: weak trump leads)
    def_trump_lock_weak=False, # ... only from a weak holding: at most 2 trumps, at most 3 points
    def_trump_margin=0.0,      # softer than the lock (0 = off): such a trump lead must beat the
                               # lead tier's card by this many points before the play-out switches
    plan_est_v2_ruff=0.25,
    plan_force_trump=False,
    playout=True,              # multi-trick roll-out with the rule policy
    playout_worlds=384,
    playout_cpp=True,          # C++ play-out (False = Python reference)
    trump_rule_known=True,     # False when trump did not come from the C++ rule (GUI)
    enum_max=0,                # enumerate all consistent deals when at most this many
    playout_lik='belief',      # world likelihood: 'rule' policy, 'q' (SPARTA's Q-net)
                               # or 'belief' (belief net + per-seat rule likelihood)
    lik_belief_model='artifacts/dmc_belief_best.pt',
    lik_belief_w=0.75,         # exponent on the belief net's world probability
    lik_partner_eps=0.5,       # per unexplained partner play (agent-like policy)
    lik_opp_eps=0.7,           # per unexplained opponent play
    lik_device='cpu',
    lik_knowledge=False,       # rules read belief-weighted (resampled) worlds too
    root_q_w=0.4,              # blend a Q-net's value into the play-out values
    root_q_model='artifacts/dmc_q_blueprint_best.pt',
    root_q_device='cuda',
    root_q_sym=False,          # suit-relabelling-averaged Q (2-6 views, one batch)
    # no-trump (mull) deals: the trump nets never saw trump=-1, so the blend
    # and the belief weights switch to the no-trump lineage (the GUI's
    # chooser-conditioned mull nets)
    mull_root_q_model='artifacts/dmc_q_mull_cond.pt',
    mull_lik_belief_model='artifacts/dmc_belief_mull_cond.pt',
    nt_tiers=False,            # no-trump lead tiers: cash opponent-proof cards, no lead
                               # toward partner through a void opponent
    decl_tilt=None,            # equity policy file: weight worlds by P(declaration | hands)
    decl_tilt_declaration=True,  # the declaration term itself (the belief net already
                               # conditions on trump and declarer; the doubles it never sees)
    double_tilt=False,         # ... and by P(meegaan / terug calls and passes | hands);
                               # 'called' = only on deals where a meegaan was called
    decl_temp=0.01,            # softmax temperature on the P(match win) scale
    decl_symmetrize=False,     # average the margin net over suit relabellings (24x cost)
    decl_human_seats=(),       # seats whose calls are human: no likelihood for them
    match_equity=None,         # policy file: rank play-out cards by P(match win), not points
    playout_nt_rules=31,       # no-trump card play in the play-out (bits, cpp/rule_playout.h)
    playout_tr_rules=0,        # trump card play in the play-out (bits, cpp/rule_playout.h)
    mull_root_q_w=None,        # blend weight at no trump (None = root_q_w): the
                               # play-out's rule policy is weaker there, the net stronger
    playout_q_model='artifacts/dmc_q_gru.pt',
    playout_q_tau=0.1,
    playout_q_device='cuda',
    playout_margin=0.0,        # points the play-out must gain to overrule
    playout_max_cards=8,       # only use it from this hand size down
    playout_agent_like=True,   # play-out leads with the agent's planner tiers
    playout_lik_agent=False,   # ... and uses it as the play likelihood too
    playout_honest=False,      # seats decide from own hand + public record only
    playout_z=1.0,             # paired-test z the gain must clear (0 = margin only)
    playout_lik_eps=0.5,       # world likelihood per unexplained play (1 = off)
    trump_keep_bonus=2.0,
    plan_skip=(3,),
    plan_exit_sim=True,
    plan_pull_min_opp=0.3,     # expected opponent trumps worth drawing
    plan_majority_margin=0.0,
    plan_to_declarer=True,
    plan_ruff=0.5,             # P(partner ruffs, not overruffed) to lead for it
    plan_cash=0.3,             # base P(safe) to cash a master
    plan_cash_risk=0.0,        # + per point at risk above 4
    plan_est_ruff=0.4,         # max opp ruff risk for establishing a suit
    plan_partner_top=0.45,
    vet_w=1.0,             # soft vetting inference (1 = off)
    vet_partner=False,
    ruff_first=0.0,
    cash_risk_w=0.0,
    pull_needs_reason=False,
    sim_vet=False,
    sim_lead=1,
    sim_worlds=40,
    win_own_pts_w=1.0,
    to_declarer_bonus=3.0,
    ruff_probs=True,
    cheapest_of_masters=False,
    partner_after_f=0.6,
    ruff_from_worlds=True,
    ruff_future=1.3,
    trump_power=0.85,
    trump_power_slope=0.3,
    declarer_pull=True,
    declarer_pull_min=3,
    endgame_master_trump=True,
    n_worlds=400,          # trump-rule-aware location sampling (0 = off)
    guard_power=0.5,
    low_pull_min=3,
    low_pull_bonus=2.0,
    trump_lead_pen=3.0,
    guard_lead_bonus=1.0,
    bare_honour_pen=2.0,
    partner_top_w=2.0,
    opp_ruff_pen=3.0,
    partner_ruff_bonus=3.0,
    singleton_bonus=1.0,
    length_w=0.2,
    playout_net=None,      # fast learned play-out policy (learn/fast_policy.py): path
    playout_net_mix=1.0,   # ... its weight against the rule play-out, per world
    playout_net_plies=0,   # > 0: hybrid instead -- the net plays the first this many
                           # plies after the candidate, the rule policy the rest
    playout_net_trick_end=False,  # ... and the net finishes the trick it is in
    playout_bp_mix=0.0,    # probe: weight of blueprint-rollout values (all seats greedy
                           # on the Q net, SPARTA's continuation) in the play-out part
    playout_bp_worlds=64,  # ... on this many of the play-out worlds (both estimators)
    playout_bp_model=None, # None = the root-Q net of the deal type
    playout_bp_device='cuda',
    ruff_convention=False, # partnership agreement (user): the opening leader, defending
                           # with a trump, leads a side singleton at trick 1; partner
                           # returns that suit while the opener may ruff. Planner
                           # preference + modelled in the play-out (tr bit 32), so the
                           # overrule can still decline it
    through_fix=True,      # trump through to a declaring partner needs 2+ trumps: a lone
                           # trump is my ruff (+0.34 +- 0.15 per changed lead, blueprint
                           # rollouts; drawing with my own master instead measured -0.33)
    last_ruff_low=True,    # last to play with the highest trump left in the game: win
                           # with the lowest trump that also wins instead (user request:
                           # the master keeps its trick for later)
    playout_sir=False,     # weight-first deduplication + deterministic stratified
                           # resampling to <= playout_worlds unique worlds
    lik_net_partner_w=0.0, # honest fast_policy action log-likelihood weight for partner
                           # on suit deals (replaces world_mismatches when > 0)
    lik_net_opp_w=0.0,     # ... and for opponents on suit deals
    lik_net_tau=1.0,       # temperature on fast_policy logits for history likelihood
    mull_endgame_ab_cards=0,  # exact double-dummy alpha-beta on mull at <= this hand size
    mull_endgame_q_w=-1.0, # root Q weight when mull AB runs (< 0 = normal qw)
    mull_endgame_z=-1.0,   # z threshold when mull AB runs (< 0 = playout_z)
    suit_net_z=-1.0,       # z threshold on suit deals with fast_policy (< 0 = playout_z)
    playout_race_k=0,      # two-stage candidate racing Stage-1 world count (0 = off)
    playout_race_z=2.5,    # Stage-1 pruning z-bound in paired SEs
    suit_root_q_w_late=-1.0,  # late-deal Q blend weight on suit deals (< 0 = off)
    worlds_belief_sampled=False,  # True when worlds were drawn from belief-weighted DP
    native=False,          # run the deploy path in C++ (cpp/rule_agent.cpp) when the
                           # params allow it (native_supported); nets stay in Python
)

# Params the native agent hard-codes: any other value keeps the Python path.
NATIVE_FIXED = dict(
    sim=True, planner=True, signal_mode=0, vet_w=1.0, enum_max=0, lik_knowledge=False,
    playout_lik='belief', playout_cpp=True, playout_honest=False, playout_agent_like=True,
    playout_lik_agent=False, match_equity=None, cheapest_of_masters=False,
    playout_bp_mix=0.0,
    guard_w=0.0, ruff_first=0.0, pull_needs_reason=False, sim_lead=1, ruff_probs=True,
    ruff_from_worlds=True,
)
# Params copied into mi.RuleParams (plan_skip and mull_root_q_w converted).
NATIVE_FIELDS = (
    'trick_rest keep_win_value keep_pts_weight win_own_pts_w partner_after_f sim_worlds '
    'sim_vet guard_power trump_power trump_power_slope ruff_future trump_keep_bonus '
    'plan_pull_min_opp plan_majority_margin plan_to_declarer plan_ruff plan_force_trump '
    'plan_cash plan_cash_risk plan_est_ruff plan_est_v2 plan_est_v2_ruff plan_partner_top '
    'nt_est_low_run nt_lock_cash_from_trick def_trump_lock def_trump_lock_weak '
    'def_trump_margin '
    'plan_exit_sim nt_tiers trump_lead_pen to_declarer_bonus bare_honour_lead_pen '
    'pull_trumps pull_min_trumps pull_min_trumps_def declarer_pull declarer_pull_min '
    'endgame_master_trump cash_threshold lead_pts_w low_pull_min low_pull_bonus '
    'guard_lead_bonus bare_honour_pen partner_top_w opp_ruff_pen partner_ruff_bonus '
    'singleton_bonus length_w playout playout_worlds playout_max_cards playout_margin '
    'ruff_convention through_fix last_ruff_low playout_z playout_nt_rules playout_tr_rules '
    'lik_belief_w lik_partner_eps lik_opp_eps root_q_w '
    'playout_sir lik_net_partner_w lik_net_opp_w lik_net_tau '
    'mull_endgame_ab_cards mull_endgame_q_w mull_endgame_z suit_net_z '
    'playout_race_k playout_race_z suit_root_q_w_late worlds_belief_sampled').split()


class Knowledge:
    """Public-information view built from one encoder call. Seats are
    relative: 0 = me, 1 = next, 2 = partner, 3 = previous."""

    def __init__(self, enc, worlds=None, me=0, min_worlds=20, vet_w=1.0,
                 vet_seats=(1, 3), boost=None, world_w=None):
        self.boost = boost      # (suit, factor): partner signalled for this suit
        self.world_w = world_w  # per-world likelihood weights (aligned with worlds)
        self.W_raw = None       # weighted pool before resampling, for the play-out
        self.w_raw = None
        self.vet_w = vet_w
        self.vet_seats = vet_seats
        self.worlds = worlds
        self.me = me
        self.min_worlds = min_worlds
        obs = enc["obs"]
        self.obs = obs
        self.n_legal = int(enc["n_legal"])
        self.legal = [int(c) for c in enc["legal"][:self.n_legal]]
        self.act = {int(c): enc["act"][i] for i, c in enumerate(self.legal)}
        t = obs[E.TRUMP_OFF:E.TRUMP_OFF + 4]
        self.trump = int(np.argmax(t)) if t.max() > 0 else None
        decl = obs[E.DECLARER_OFF:E.DECLARER_OFF + 4]
        self.declarer = int(np.argmax(decl)) if decl.max() > 0 else None
        self.hand = [c for c in range(32) if obs[E.OWN_HAND_OFF + c] > 0]
        self.cap = [[7] * 4 for _ in range(4)]
        for s in range(4):
            for r in range(4):
                if obs[E.VOID_OFF + r * 4 + s] > 0:
                    self.cap[r][s] = -1
        self.played_by = [0] * 4
        seen = set(self.hand)
        self.trick = []          # [(rel seat, card)] on the table now
        self._read_history(seen)
        self.unseen = [c for c in range(32) if c not in seen]
        self.size = [len(self.hand)] + [8 - self.played_by[r] for r in (1, 2, 3)]
        self._locate()

    def _read_history(self, seen):
        obs = self.obs
        trick = []
        self.plies = []          # (rel seat, card, led suit, pos, partner winning)
        for i in range(32):
            p = obs[E.HISTORY_OFF + i * E.PLY_DIM:E.HISTORY_OFF + (i + 1) * E.PLY_DIM]
            if p[0:4].max() == 0:
                break
            seat = int(np.argmax(p[0:4]))
            card = int(np.argmax(p[4:36]))
            led = int(np.argmax(p[36:40]))
            pos = int(np.argmax(p[40:44]))
            f = p[E.PLY_FLAG_OFF:E.PLY_FLAG_OFF + E.N_FLAGS]
            if pos == 0:
                trick = []
            winner_card = None
            if pos > 0 and p[44:48].max() > 0:
                wseat = int(np.argmax(p[44:48]))
                winner_card = next(c for s, c in trick if s == wseat)
            if winner_card is not None:
                if f[E.F_FOLLOWED_NO_BEAT] and suit_of(winner_card) == led:
                    self.cap[seat][led] = min(self.cap[seat][led], rank_of(winner_card))
                if f[E.F_DUCKED_TRUMP] and self.trump is not None:
                    self.cap[seat][self.trump] = min(self.cap[seat][self.trump],
                                                     rank_of(winner_card))
                if (f[E.F_OBLIGATION] and f[E.F_VOID_ESTABLISHED]
                        and not f[E.F_TRUMPED] and self.trump is not None
                        and suit_of(winner_card) != self.trump):
                    self.cap[seat][self.trump] = -1
            self.plies.append((seat, card, led, pos, bool(f[E.F_PARTNER_WINNING])))
            trick.append((seat, card))
            seen.add(card)
            self.played_by[seat] += 1
            if f[E.F_TRICK_COMPLETED]:
                trick = []
        self.trick = trick
        self.cap[0] = [7] * 4  # never constrain myself

    def _locate(self):
        others = [1, 2, 3]
        poss = {c: [p for p in others if self.cap[p][suit_of(c)] >= rank_of(c)]
                for c in self.unseen}
        # A seat whose possible cards exactly fill its hand holds all of them;
        # a seat already filled by certain cards holds nothing else.
        for _ in range(4):
            changed = False
            for p in others:
                mine = [c for c in self.unseen if p in poss[c]]
                fixed = [c for c in mine if poss[c] == [p]]
                if len(mine) == self.size[p] and len(fixed) < len(mine):
                    for c in mine:
                        if poss[c] != [p]:
                            poss[c] = [p]
                            changed = True
                elif len(fixed) == self.size[p] and len(mine) > len(fixed):
                    for c in mine:
                        if poss[c] != [p]:
                            poss[c] = [q for q in poss[c] if q != p]
                            changed = True
            if not changed:
                break
        self.poss = poss
        self.prob = {}
        self.W = None
        if self.worlds is not None and len(self.worlds):
            self._locate_from_worlds()
        if self.W is None:
            for c, hs in poss.items():
                w = {p: max(self.size[p], 1e-3) for p in hs}
                z = sum(w.values()) or 1.0
                self.prob[c] = {p: w[p] / z for p in hs}

    def _locate_from_worlds(self):
        """Location marginals from the trump-constrained sampler's worlds
        (voids + the declarer's hand consistent with the trump rule), kept
        only where they also respect the rank caps read from the flags."""
        W = np.asarray(self.worlds, dtype=np.int64)
        W = np.stack([W[:, (self.me + r) % 4] for r in range(4)], axis=1)
        keep = np.ones(len(W), dtype=bool)
        for r in (1, 2, 3):
            forbid = 0
            for su in range(4):
                for rk in range(self.cap[r][su] + 1, 8):
                    forbid |= 1 << (su * 8 + rk)
            if forbid:
                keep &= (W[:, r] & forbid) == 0
        if keep.sum() < self.min_worlds:
            return
        W = W[keep]
        if self.world_w is not None:
            # The rules read plain means over W, so hand them a posterior
            # sample; the play-out keeps the unresampled pool and its
            # weights (importance weights are less noisy than duplicates).
            w = np.asarray(self.world_w, dtype=float)[keep]
            self.W_raw, self.w_raw = W, w
            W = W[np.random.default_rng().choice(len(W), size=len(W), p=w / w.sum())]
        if self.vet_w < 1.0:
            W = self._vet_reweight(W)
        if self.boost is not None:
            W = self._signal_reweight(W)
        self.W = W
        for c in self.unseen:
            bit = np.int64(1) << c
            fr = [(p, float(((W[:, p] & bit) != 0).mean())) for p in (1, 2, 3)]
            self.prob[c] = {p: f for p, f in fr if f > 0}

    def _signal_reweight(self, W):
        """Partner asked for suit w: he holds its top unseen card, or is void
        in it (and can ruff). Those worlds become `factor` times likelier."""
        w, factor = self.boost
        unseen = [c for c in self.unseen if suit_of(c) == w]
        if not unseen:
            return W
        top = max(unseen, key=rank_of)
        m = 0
        for c in unseen:
            m |= 1 << c
        fits = ((W[:, 2] & np.int64(1 << top)) != 0) | ((W[:, 2] & np.int64(m)) == 0)
        wts = np.where(fits, float(factor), 1.0)
        if wts.min() == wts.max():
            return W
        idx = np.random.default_rng().choice(len(W), size=len(W), p=wts / wts.sum())
        return W[idx]

    def _vet_reweight(self, W):
        """Soft inference: a player last to play on his partner's winning
        trick vets his biggest card. Worlds in which he could legally have
        thrown a bigger side card than the one he played are down-weighted,
        then the world set is resampled by weight."""
        trump = self.trump
        trump_bits = 0
        if trump is not None:
            for r in range(8):
                trump_bits |= 1 << (trump * 8 + r)
        wts = np.ones(len(W))
        for i, (seat, card, led, pos, pw) in enumerate(self.plies):
            if seat not in self.vet_seats or pos != 3 or not pw:
                continue
            later = 0
            for s2, c2, _, _, _ in self.plies[i + 1:]:
                if s2 == seat:
                    later |= 1 << c2
            if suit_of(card) == led:
                opt = 0xFF << (led * 8)
            else:
                opt = 0xFFFFFFFF
            opt &= ~trump_bits
            big = 0
            for c in range(32):
                if PTS[rank_of(c)] > PTS[rank_of(card)]:
                    big |= 1 << c
            then = W[:, seat] | np.int64(later)
            bad = (then & np.int64(opt & big)) != 0
            wts = np.where(bad, wts * self.vet_w, wts)
        if wts.min() == wts.max():
            return W
        idx = np.random.default_rng().choice(len(W), size=len(W), p=wts / wts.sum())
        return W[idx]

    # -- queries ---------------------------------------------------------
    def p_holds(self, c, p):
        return self.prob.get(c, {}).get(p, 0.0)

    def p_none(self, cards, p):
        if self.W is not None:
            m = 0
            for c in cards:
                m |= 1 << c
            if not m:
                return 1.0
            return float(((self.W[:, p] & np.int64(m)) == 0).mean())
        q = 1.0
        for c in cards:
            q *= 1.0 - self.p_holds(c, p)
        return q

    def p_ruff(self, s, o):
        """P(seat o is void in suit s and holds a trump)."""
        if self.trump is None or s == self.trump:
            return 0.0
        sm = 0
        for c in self.unseen_suit(s):
            sm |= 1 << c
        tm = 0
        for c in self.unseen_suit(self.trump):
            tm |= 1 << c
        if not tm:
            return 0.0
        if self.W is not None:
            w = self.W[:, o]
            return float((((w & np.int64(sm)) == 0) & ((w & np.int64(tm)) != 0)).mean())
        return self.p_none(self.unseen_suit(s), o) * (1.0 - self.p_none(self.unseen_suit(self.trump), o))

    def partner_forced_pts(self, s):
        """Expected points partner must follow with on a lead of suit s
        (his cheapest card there, 0 if void), from the worlds."""
        if self.W is None:
            return 0.0
        cards = self.unseen_suit(s)
        if not cards:
            return 0.0
        tot = 0.0
        for w in self.W[:, 2]:
            held = [PTS[rank_of(c)] for c in cards if int(w) >> c & 1]
            tot += min(held) if held else 0
        return tot / len(self.W)

    def unseen_suit(self, s):
        return [c for c in self.unseen if suit_of(c) == s]

    def higher_unseen(self, c):
        return [u for u in self.unseen
                if suit_of(u) == suit_of(c) and rank_of(u) > rank_of(c)]

    def is_master(self, c):
        return not self.higher_unseen(c)

    def is_boss(self, c):
        """No opponent can beat c in its suit: my master, or both opponents
        are known to hold nothing above it (void, or they followed without
        beating a lower card). Partner may hold the higher cards."""
        s, r = suit_of(c), rank_of(c)
        return self.is_master(c) or (self.cap[1][s] < r and self.cap[3][s] < r)

    def p_beats(self, x, o, led):
        """P(seat o, still to play in a trick led in `led`, can beat x)."""
        trump = self.trump
        sx = suit_of(x)
        higher = self.higher_unseen(x)
        if sx == led:
            p_high = 1.0 - self.p_none(higher, o)
            if trump is not None and sx != trump:
                p_void = self.p_none(self.unseen_suit(led), o)
                p_trump = 1.0 - self.p_none(self.unseen_suit(trump), o)
                return min(1.0, p_high + p_void * p_trump)
            return p_high
        if trump is not None and sx == trump:
            p_void = self.p_none(self.unseen_suit(led), o)
            return p_void * (1.0 - self.p_none(higher, o))
        return 1.0

    def current_winner(self):
        """Card currently winning the trick on the table (ordering only)."""
        if not self.trick:
            return None, None
        ws, wc = self.trick[0]
        for s, c in self.trick[1:]:
            if outranks(c, wc, self.trump):
                ws, wc = s, c
        return ws, wc


class RuleAgent(Agent):
    def __init__(self, verbose=False, **params):
        self.p = dict(DEFAULTS)
        self.p.update(params)
        self.verbose = verbose
        self.log = []
        self.last_values = None

    def new_game(self, game, dealer, player_id):
        self.me = player_id
        self.dealer = dealer
        self.last_values = None
        # match score as (declaring team, other team), for the declaration
        # likelihood; callers that play matches set it before each deal
        self.decl_score = getattr(game, 'decl_score', (0, 0))
        self._human_declared = False

    def _sampler_declarer(self):
        """Declarer passed to the world sampler: the dealer when trump was
        picked by the C++ trump rule (the sampler then keeps only declarer
        hands the rule would have declared), -1 = unconstrained when trump
        came from elsewhere (a human or the GUI's equity chooser) -- the
        rule constraint would then be wrong, and can make the sampler fail."""
        return self.dealer if self.p['trump_rule_known'] else -1

    def native_supported(self):
        return all(self.p[k] == v for k, v in NATIVE_FIXED.items())

    def _native_params(self):
        if getattr(self, '_nparams', None) is None:
            q = mi.RuleParams()
            for f in NATIVE_FIELDS:
                setattr(q, f, self.p[f])
            q.plan_skip_mask = sum(1 << t for t in self.p['plan_skip'])
            w = self.p['mull_root_q_w']
            q.mull_root_q_w = -1.0 if w is None else w
            q.playout_net_mix = self.p['playout_net_mix']
            q.playout_net_plies = self.p['playout_net_plies']
            q.playout_net_trick_end = self.p['playout_net_trick_end']
            self._nparams = q
        return self._nparams

    def belief_probs(self, t, arrays, player_id):
        """The belief net's 3x32 holder probabilities (seats me+1..me+3),
        float32 -- the numbers dmc_belief_log_weights takes the log of."""
        import torch
        from .model import belief_probs, split_belief_inputs
        path = self.p['mull_lik_belief_model' if t < 0 else 'lik_belief_model']
        cache = self.__dict__.setdefault('_belief_model', {})
        if path not in cache:
            from .dmc_model import load_belief_model
            cache[path] = load_belief_model(path, self.p['lik_device'])
        obs = mi.encode_observation(*arrays, self.dealer, player_id)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.p['lik_device'])
            probs = belief_probs(cache[path](obs_t), *split_belief_inputs(obs_t))
        return probs.squeeze(0).cpu().numpy().astype(np.float32)

    def native_choose(self, game, player_id, played_cards, played_players, arrays, worlds):
        """cpp/rule_agent.cpp on the given n_worlds sample (absolute seats)."""
        t = -1 if game.trump_suit is None else int(game.trump_suit)
        history = list(zip(played_players, played_cards)) + \
            [(p, c.to_index()) for p, c in game.current_trick]
        hs = np.array([s for s, _ in history], dtype=np.int32)
        hc = np.array([c for _, c in history], dtype=np.int32)
        probs = self.belief_probs(t, arrays, player_id)
        q = None
        if self._q_weight(t < 0) > 0:
            self._blend_root_q({}, game, player_id, played_cards, played_players)
            q = np.zeros(32, dtype=np.float32)
            for c, v in self.last_q.items():
                q[c] = v
        W = np.ascontiguousarray(worlds if worlds is not None else np.zeros((0, 4)),
                                 dtype=np.int64)
        extra = None
        if self.p['decl_tilt'] and len(W):
            # per input world; the C++ keeps it aligned through its filter
            extra = np.ascontiguousarray(
                self.decl_log_weights(game, t, W, history, player_id), dtype=np.float64)
        net = None
        if self.p['playout_net']:
            if self.p['playout_net'] not in _FAST_NETS:
                from .fast_policy import load_native
                _FAST_NETS[self.p['playout_net']] = load_native(self.p['playout_net'])
            net = _FAST_NETS[self.p['playout_net']]
        card, why, vals = mi.rule_agent_choose(*arrays, self.dealer, player_id, W, hs, hc,
                                               probs, q, self._native_params(), extra, net)
        self.last_values = dict(vals) if vals else None
        return card, why

    def choose_card(self, game, player_id, played_cards, played_players):
        legal = game.get_legal_moves(player_id)
        self.last_values = None  # play-out mean points per card (for GUI hints)
        if len(legal) == 1:
            return legal[0]
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        if self.p['native'] and self.native_supported():
            try:
                worlds = mi.sample_determinizations(
                    *arrays, player_id, self.p['n_worlds'], self._sampler_declarer(), True)
            except RuntimeError:
                worlds = None
            card, why = self.native_choose(game, player_id, played_cards, played_players,
                                           arrays, worlds)
            if self.verbose:
                self.log.append(f"P{player_id} -> {cstr(card)}  ({why}) [native]")
            return Card.from_index(card)
        enc = mi.encode_q_observation(*arrays, self.dealer, player_id)
        worlds = None
        if self.p['enum_max'] > 0:
            # late in the hand the consistent deals are few: list each once
            # (exact) instead of sampling hundreds of copies of them
            en = mi.enumerate_determinizations(*arrays, player_id, self.p['enum_max'],
                                               self._sampler_declarer())
            if len(en):
                worlds = en[np.random.default_rng().permutation(len(en))]
        if worlds is None and self.p['n_worlds'] > 0:
            try:
                worlds = mi.sample_determinizations(
                    *arrays, player_id, self.p['n_worlds'], self._sampler_declarer(), True)
            except RuntimeError:
                worlds = None
        world_w = None
        if (worlds is not None and len(worlds) and self.p['lik_knowledge']
                and self.p['playout_lik'] == 'belief'):
            hist0 = list(zip(played_players, played_cards)) + \
                [(p, c.to_index()) for p, c in game.current_trick]
            t0 = -1 if game.trump_suit is None else int(game.trump_suit)
            world_w = self.belief_world_weights(t0, arrays, worlds, hist0, player_id, game)
        k = Knowledge(enc, worlds, player_id, vet_w=self.p['vet_w'],
                      vet_seats=(1, 2, 3) if self.p['vet_partner'] else (1, 3),
                      world_w=world_w)
        sig = None
        if self.p['signal_mode']:
            if k.trick:
                sig = self._signal_send(k)
            elif self.p['signal_read'] == 'infer':
                asked = self._signal_asked(k)
                if asked is not None:
                    k = Knowledge(enc, worlds, player_id, vet_w=self.p['vet_w'],
                                  vet_seats=(1, 2, 3) if self.p['vet_partner'] else (1, 3),
                                  boost=(asked, self.p['signal_boost']))
            else:
                sig = self._signal_read(k)
        if sig is not None:
            choice, why = sig
        elif k.trick:
            choice, why = self._follow(k)
        else:
            choice, why = self._lead(k)
        if self.p['cheapest_of_masters']:
            low = self._lowest_equivalent_master(k, choice)
            if low != choice:
                why += f"; {cstr(low)} is equivalent"
                choice = low
        lock = self.p['signal_final']
        signal_locked = sig is not None and (
            lock is True or (lock == 'send' and k.trick) or (lock == 'read' and not k.trick))
        if (self.p['playout'] and k.W is not None and len(k.legal) > 1
                and len(k.hand) <= self.p['playout_max_cards'] and not signal_locked):
            from .rule_playout import evaluate_candidates, world_weights
            n = min(len(k.W), self.p['playout_worlds'])
            W = k.W[:n] if k.W_raw is None else k.W_raw[:n]
            worlds_abs = np.stack([W[:, (s - player_id) % 4] for s in range(4)], axis=1)
            history = list(zip(played_players, played_cards)) + \
                [(p, c.to_index()) for p, c in game.current_trick]
            honest = int(self.p['playout_honest'])
            agent_like = self.p['playout_agent_like']
            lik_agent = agent_like and self.p['playout_lik_agent']
            if self.p['playout_cpp']:
                # C++ port (cpp/rule_playout.cpp), parity-tested against the
                # Python reference in tests/test_rule_playout_parity.py
                t = -1 if k.trump is None else k.trump
                hs = np.array([s for s, _ in history], dtype=np.int32)
                hc = np.array([c for _, c in history], dtype=np.int32)
                worlds_abs = np.ascontiguousarray(worlds_abs, dtype=np.int64)
                if self.p['playout_lik_eps'] < 1.0 and history and self.p['playout_lik'] == 'rule':
                    wts = mi.rule_playout_weights(t, self.dealer, player_id, worlds_abs,
                                                  hs, hc, self.p['playout_lik_eps'],
                                                  honest, lik_agent)
                else:
                    wts = np.ones(len(worlds_abs))
                net, plies = None, self.p['playout_net_plies']
                if self.p['playout_net']:
                    if self.p['playout_net'] not in _FAST_NETS:
                        from .fast_policy import load_native
                        _FAST_NETS[self.p['playout_net']] = load_native(self.p['playout_net'])
                    net = _FAST_NETS[self.p['playout_net']]
                hybrid = net is not None and plies > 0
                arr = mi.rule_playout_evaluate(t, self.dealer, player_id, list(k.legal),
                                               worlds_abs, hs, hc, honest, agent_like,
                                               self.p['playout_nt_rules'],
                                               self.p['playout_tr_rules']
                                               | (32 if self.p['ruff_convention'] else 0),
                                               net if hybrid else None, plies,
                                               self.p['playout_net_trick_end'])
                if self.p['playout_net'] and not hybrid:
                    nets = _FAST_NETS
                    if self.p['playout_net'] not in nets:
                        from .fast_policy import load_native
                        nets[self.p['playout_net']] = load_native(self.p['playout_net'])
                    nm = self.p['playout_net_mix']
                    arr = (1 - nm) * arr + nm * mi.fast_policy_evaluate(
                        nets[self.p['playout_net']], t, self.dealer, player_id, list(k.legal),
                        worlds_abs, hs, hc)
                per = {c: arr[i] for i, c in enumerate(k.legal)}
            else:
                if self.p['playout_lik_eps'] < 1.0 and history and self.p['playout_lik'] == 'rule':
                    wts = world_weights(k.trump, self.dealer, player_id, worlds_abs,
                                        history, self.p['playout_lik_eps'], honest,
                                        lik_agent)
                else:
                    wts = np.ones(len(worlds_abs))
                per = evaluate_candidates(k.trump, self.dealer, player_id, k.legal,
                                          worlds_abs, history, honest, agent_like,
                                          self.p['playout_nt_rules'],
                                          self.p['playout_tr_rules']
                                          | (32 if self.p['ruff_convention'] else 0))
            bp = self.p['playout_bp_mix']
            if bp > 0:
                m = min(n, self.p['playout_bp_worlds'])
                worlds_abs = np.ascontiguousarray(worlds_abs[:m])
                wts = np.ones(m)
                bpv = self._bp_rollouts(game, player_id, played_cards, played_players,
                                        list(k.legal), worlds_abs)
                per = {c: (1 - bp) * np.asarray(per[c])[:m] + bp * bpv[i]
                       for i, c in enumerate(k.legal)}
            if self.p['playout_lik'] == 'q' and history:
                wts = self.q_world_weights(k, worlds_abs, history, player_id)
            if k.W_raw is not None:
                wts = k.w_raw[:n]
            elif self.p['playout_lik'] == 'belief':
                t0 = -1 if k.trump is None else k.trump
                wts = self.belief_world_weights(t0, arrays, worlds_abs, history, player_id,
                                                game)
            if self._q_weight(k.trump is None) > 0:
                per = self._blend_root_q(per, game, player_id, played_cards, played_players)
            if self.p['match_equity']:
                per = self._to_equity(per, game, player_id)
            wts = wts / wts.sum()
            vals = {c: float(np.dot(wts, v)) for c, v in per.items()}
            self.last_values = dict(vals)
            pool = vals
            lock = self.p['nt_lock_cash_from_trick']
            if (lock and k.trump is None and not k.trick and why.startswith("plan: cash master")
                    and 9 - len(k.hand) >= lock):
                # late in a mull the play-out undervalues cashing: only
                # another master may replace the cash
                pool = {c: v for c, v in vals.items() if c == choice or k.is_master(c)}
            def_pen = 0.0
            if ((self.p['def_trump_lock'] or self.p['def_trump_margin'] > 0)
                    and k.trump is not None and not k.trick
                    and k.declarer not in (0, 2) and suit_of(choice) != k.trump
                    and (not self.p['def_trump_lock_weak'] or (
                        sum(1 for c in k.hand if suit_of(c) == k.trump) <= 2
                        and sum(PTS[rank_of(c)] for c in k.hand if suit_of(c) == k.trump) <= 3))):
                # defenders leading a small trump from a weak holding was
                # the play-out's doing, twice the blueprint's rate at trick 1:
                # blocked, or it must clear a margin
                if self.p['def_trump_lock']:
                    pool = {c: v for c, v in pool.items() if suit_of(c) != k.trump}
                else:
                    def_pen = self.p['def_trump_margin']
                    pool = {c: v - (def_pen if suit_of(c) == k.trump else 0.0) for c, v in pool.items()}
            best = max(pool, key=pool.get)
            if best != choice:
                d = per[best] - per[choice]
                m = float(np.dot(wts, d))
                n_eff = 1.0 / float(np.dot(wts, wts))
                se = float(np.sqrt(max(np.dot(wts, (d - m) ** 2), 1e-12) / n_eff))
                z = self.p['playout_z']
                need = self.p['playout_margin'] + (def_pen if suit_of(best) == k.trump else 0.0)
                if m > need and (z <= 0 or m > z * se):
                    why += (f"; play-out prefers {cstr(best)} {vals[best]:.1f} vs "
                            f"{vals[choice]:.1f} (+{m:.2f}±{se:.2f})")
                    choice = best
        if self.p['last_ruff_low'] and len(k.trick) == 3 and k.trump is not None \
                and suit_of(choice) == k.trump and k.is_master(choice):
            # last to play, about to spend the highest trump left in the game:
            # a lower trump that also takes the trick saves it for later
            low = [c for c in k.legal if suit_of(c) == k.trump and rank_of(c) < rank_of(choice)
                   and k.act[c][E.ACT_WINS_OFF] > 0]
            if low:
                why += f"; {cstr(min(low, key=rank_of))} wins too, master trump kept"
                choice = min(low, key=rank_of)
        if self.verbose:
            self.log.append(f"P{player_id} {'lead' if not k.trick else 'pos%d' % len(k.trick)}"
                            f" -> {cstr(choice)}  ({why})")
        return Card.from_index(choice)

    def q_world_weights(self, k, worlds_abs, history, player_id):
        """SPARTA's world weights: each world scored by the log-likelihood,
        under softmax(Q/tau) of a DMC Q-net, of every observed play by the
        other seats (learn/dmc_sparta.py::plpw_q_log_weights). A world that
        replays no observed decision carries no information, weight 1."""
        from .dmc_sparta import plpw_q_log_weights
        if getattr(self, '_q_model', None) is None:
            from .dmc_model import load_q_model
            self._q_model = load_q_model(self.p['playout_q_model'], self.p['playout_q_device'])
        obs_players = np.array([s for s, _ in history], dtype=np.int32)
        obs_cards = np.array([c for _, c in history], dtype=np.int32)
        t = -1 if k.trump is None else k.trump
        logw = plpw_q_log_weights(self._q_model, self.p['playout_q_device'],
                                  np.ascontiguousarray(worlds_abs, dtype=np.int64),
                                  obs_players, obs_cards, t, self.dealer, player_id,
                                  self.p['playout_q_tau'])
        logw = np.where(np.isfinite(logw), logw, 0.0)
        return np.exp(logw - logw.max())

    def _bp_rollouts(self, game, player_id, played_cards, played_players, cards, worlds_abs):
        """My team's final points [card][world] with every seat following the
        Q blueprint greedily after my card (learn/dmc_sparta.py rollouts)."""
        from .dmc_model import load_q_model
        from .dmc_sparta import q_blueprint_rollouts
        from .sparta import inject_world
        mull = game.trump_suit is None
        path = self.p['playout_bp_model'] or self.p['mull_root_q_model' if mull else 'root_q_model']
        dev = self.p['playout_bp_device']
        cache = self.__dict__.setdefault('_bp_model', {})
        if (path, dev) not in cache:
            cache[(path, dev)] = load_q_model(path, dev)
        trump = -1 if mull else int(game.trump_suit)
        worlds = []
        for card in cards:
            cobj = Card.from_index(card)
            for w in worlds_abs:
                g = inject_world(self.dealer, trump, w, game.current_trick, game.hand_points,
                                 game.tricks_won, game.current_player)
                pc, pp = list(played_cards), list(played_players)
                pre = list(g.current_trick)
                if not g.play_card(player_id, cobj):
                    raise ValueError("candidate illegal in a sampled world")
                if len(pre) == 3:
                    for s_, c_ in pre + [(player_id, cobj)]:
                        pc.append(c_.to_index())
                        pp.append(s_)
                worlds.append((g, pc, pp))
        pts = q_blueprint_rollouts(worlds, self.dealer, cache[(path, dev)], dev, None, 0.0,
                                   q_memo={})
        return pts[:, player_id % 2].reshape(len(cards), len(worlds_abs))

    def _q_weight(self, mull):
        w = self.p['mull_root_q_w'] if mull else None
        return self.p['root_q_w'] if w is None else w

    def _blend_root_q(self, per, game, player_id, played_cards, played_players):
        """Mix a DMC Q-net's estimate of each card into its per-world
        play-out values: (1-w) * play-out + w * (30 Q + 30). Q is the net's
        expected final team points (scaled (pts-30)/30) from this info set,
        the same quantity the play-out estimates; the shift is constant
        across worlds, so the paired test still sees the same noise."""
        mull = game.trump_suit is None
        path = self.p['mull_root_q_model' if mull else 'root_q_model']
        cache = self.__dict__.setdefault('_root_q', {})
        if path not in cache:
            from .dmc_agent import QNetAgent, SymmetrizedQNetAgent
            from .dmc_model import load_q_model
            dev = self.p['root_q_device']
            # sym: average Q over the rule-safe suit relabellings (the
            # "Blueprint+" wrapper, +0.44 over the raw net as a policy);
            # trump deals only -- the no-trump orbit is 24 views, ungated
            sym = self.p['root_q_sym'] and not mull
            cls = SymmetrizedQNetAgent if sym else QNetAgent
            cache[path] = cls(load_q_model(path, dev), dev)
        agent = cache[path]
        agent.dealer = self.dealer
        legal, q = agent.q_values(game, player_id, played_cards, played_players)
        qv = {int(c): float(x) for c, x in zip(legal, q)}
        self.last_q = qv  # net value per card, (pts - 30) / 30 (diagnostics)
        w = self._q_weight(mull)
        return {c: (1 - w) * v + w * (30.0 * qv[c] + 30.0) for c, v in per.items()}

    def belief_world_weights(self, t, arrays, worlds_abs, history, player_id, game=None):
        """World weights from the DMC belief net's per-card marginals
        (exponent lik_belief_w) times a per-seat rule-policy likelihood:
        each partner play the agent-like roll-out policy would not have made
        costs lik_partner_eps, each such opponent play lik_opp_eps. Weights
        and eps were fitted on logged decisions by the log-score of the true
        deal (belief benchmark, 2026-11-11): +0.69 nats over uniform vs
        +0.18 for the plain rule likelihood."""
        from .dmc_hybrid import dmc_belief_log_weights
        path = self.p['mull_lik_belief_model' if t < 0 else 'lik_belief_model']
        cache = self.__dict__.setdefault('_belief_model', {})
        if path not in cache:
            from .dmc_model import load_belief_model
            cache[path] = load_belief_model(path, self.p['lik_device'])
        worlds_abs = np.ascontiguousarray(worlds_abs, dtype=np.int64)
        obs = mi.encode_observation(*arrays, self.dealer, player_id)
        logw = self.p['lik_belief_w'] * dmc_belief_log_weights(
            cache[path], self.p['lik_device'], obs, worlds_abs, arrays, player_id)
        if history:
            hs = np.array([s for s, _ in history], dtype=np.int32)
            hc = np.array([c for _, c in history], dtype=np.int32)
            mm = mi.rule_playout_mismatches(t, self.dealer, player_id, worlds_abs, hs, hc,
                                            0, True)
            partner = (player_id + 2) % 4
            opp = [(player_id + 1) % 4, (player_id + 3) % 4]
            logw = logw + mm[:, partner] * np.log(self.p['lik_partner_eps']) \
                + mm[:, opp].sum(axis=1) * np.log(self.p['lik_opp_eps'])
        if self.p['decl_tilt'] and game is not None:
            logw = logw + self.decl_log_weights(game, t, worlds_abs, history, player_id)
        return np.exp(logw - logw.max())

    def _to_equity(self, per, game, player_id):
        """Per-world team points -> P(match win) x 100, from the decision
        layer's equity table V (the same table the declarations and doubles
        are priced with). Far from 101 this is nearly linear in points and
        changes nothing; near the end a sure small win beats a risky big
        one, and a doubled deal is played for the match, not the margin."""
        from .decision_layer import equity_vector
        if getattr(self, '_V', None) is None:
            import torch
            self._V = np.asarray(torch.load(self.p['match_equity'], map_location='cpu',
                                            weights_only=False)['V'])
        team = player_id % 2
        us, them = game.team_scores[team].score, game.team_scores[1 - team].score
        next_bit = 1 if (game.dealer + 1) % 2 == team else 0
        eq = equity_vector(self._V, us, them, next_bit, game.multiplier)
        grid = np.arange(-30, 31)
        return {c: 100.0 * np.interp(np.asarray(v) - 30.0, grid, eq) for c, v in per.items()}

    def set_declaration_context(self, us, them, declarer_is_bot=True):
        """Match score from the DECLARING team's view, and whether the
        declaration came from the bot policy the tilt models (the GUI calls
        this after new_game; same contract as SpartaDMCAgent). A human's
        declaration gets no likelihood: the tilt models the wrong agent."""
        self.decl_score = (int(us), int(them))
        self._human_declared = not declarer_is_bot

    # -- what the declaration and the doubles say about the hands ----------
    def _decl_model(self):
        if getattr(self, '_decl', None) is None:
            from .decision_layer import EquityChoiceModel
            self._decl = EquityChoiceModel(self.p['decl_tilt'], 'cpu',
                                           temp=self.p['decl_temp'])
            # the served chooser averages its margin net over the 24 suit
            # relabellings; for a likelihood over hundreds of worlds per
            # move that is 24x the phone budget, so it is optional here
            self._decl.policy.oracle.symmetrize = self.p['decl_symmetrize']
        return self._decl

    def decl_log_weights(self, game, t, worlds_abs, history, player_id):
        """log P(declaration, meegaan, terug | the world's hands) under the
        equity decision layer (learn/decision_layer.py, the policy that
        makes these calls for bot seats). A dealer who declared hearts holds
        hearts; one who mulled holds masters everywhere; a defender who said
        meegaan is strong, and a pass is evidence too. Replaces the hard
        trump-rule constraint whenever declarations are not the C++ rule's
        (trump_rule_known=False). The chooser is an argmax, so its
        likelihood is softened: softmax / sigmoid at decl_temp on the
        P(match win) scale, as in SPARTA's M6.3 tilt. Seats whose call
        came from a human are left out (decl_human_seats)."""
        from .decision_layer import base_multiplier, equity_vector
        model = self._decl_model()
        pol = model.policy
        played = np.zeros(4, dtype=np.int64)
        for s, c in history:
            played[s] |= np.int64(1) << np.int64(c)
        init = np.asarray(worlds_abs, dtype=np.int64) | played[None, :]
        decl = 4 if t < 0 else int(t)
        d = self.dealer
        us, them = self.decl_score            # declaring team, other team
        human = set(self.p['decl_human_seats'])
        logw = np.zeros(len(init))
        if (self.p['decl_tilt_declaration'] and d != player_id and d not in human
                and not self._human_declared):
            logw += model.log_likelihood(init[:, d], decl, us=us, them=them)
        if not self.p['double_tilt']:
            return logw
        if self.p['double_tilt'] == 'called' and not game.doubled:
            # a meegaan is strong evidence about the caller's hand; "nobody
            # doubled" is weak and only costs effective worlds (declared
            # matches, seed 70: +1.14 multiplied pts on doubled deals,
            # -0.03 on the rest)
            return logw
        T = self.p['decl_temp']
        b = base_multiplier(decl)
        n = len(init)

        def logsig(x):
            return -np.logaddexp(0.0, -x)

        # meegaan: defenders asked from the dealer's left; first yes takes it
        pm_seat = getattr(game, 'meegaan_seat', None)
        pt = pol._rates(us, them)[1][decl] if pol.price_terug else 0.0
        ev = {k: equity_vector(pol.V, them, us, 1, k, pol.target) for k in (b, 2 * b, 4 * b)}
        lp, lq = [], []                           # log P(call), log P(pass) per defender
        for off in (1, 3):
            seat = (d + off) % 4
            dist = pol.oracle.dist(init[:, seat].astype(np.uint64), np.full(n, decl),
                                   np.full(n, off))
            delta = (1 - pt) * dist @ ev[2 * b] + pt * dist @ ev[4 * b] - dist @ ev[b]
            lp.append(logsig(delta / T))
            lq.append(logsig(-delta / T))
        if not pol.enable_meegaan or human & {(d + 1) % 4, (d + 3) % 4}:
            return logw
        if not game.doubled:
            logw += lq[0] + lq[1]
        elif pm_seat == (d + 1) % 4:
            logw += lp[0]
        elif pm_seat == (d + 3) % 4:
            logw += lq[0] + lp[1]
        else:                                     # doubled, caller unknown
            logw += np.log(-np.expm1(np.minimum(lq[0] + lq[1], -1e-12)))
        if not game.doubled or not pol.enable_terug or human & {d, (d + 2) % 4}:
            return logw
        # terug: the trump-makers answer, dealer first, conditioned on the call
        tr_seat = getattr(game, 'terug_seat', None)
        evt = {k: equity_vector(pol.V, us, them, 0, k, pol.target) for k in (2 * b, 4 * b)}
        lp, lq = [], []
        for off in (0, 2):
            seat = (d + off) % 4
            dist = pol.oracle.dist(init[:, seat].astype(np.uint64), np.full(n, decl),
                                   np.full(n, off), announced=np.ones(n, dtype=np.int64))
            delta = dist @ evt[4 * b] - dist @ evt[2 * b]
            lp.append(logsig(delta / T))
            lq.append(logsig(-delta / T))
        if not game.redoubled:
            logw += lq[0] + lq[1]
        elif tr_seat == d:
            logw += lp[0]
        elif tr_seat == (d + 2) % 4:
            logw += lq[0] + lp[1]
        else:
            logw += np.log(-np.expm1(np.minimum(lq[0] + lq[1], -1e-12)))
        return logw

    @staticmethod
    def _lowest_equivalent_master(k, c):
        """A master's touching run (nothing live between the cards) holds
        interchangeable winners: play the lowest of the run. The higher card
        only banks now the points the lower one banks later, and it is the
        one that can never be beaten."""
        if not k.is_master(c):
            return c
        # Cards on the table are live: an opponent's ace on the table
        # separates my manille from my king.
        live = set(k.unseen) | {x for _, x in k.trick}
        best = c
        for d in k.legal:
            if (suit_of(d) == suit_of(c) and rank_of(d) < rank_of(best)
                    and not any((suit_of(c) << 3 | r) in live
                                for r in range(rank_of(d) + 1, rank_of(c)))):
                best = d
        return best

    def simulate_trick(self, k, cand):
        """Play out the rest of the current trick over the sampled worlds,
        the other seats answering with simple human rules (legal sets from
        the engine): side winning -> lowest card (the last seat vets its
        biggest side card); otherwise the cheapest winner, else the cheapest
        card. Returns (P(our side wins), E[signed trick points])."""
        trump_suit = None if k.trump is None else Suit(k.trump)
        W = k.W
        n = min(len(W), self.p['sim_worlds'])
        base = [(p, CARDS[c]) for p, c in k.trick] + [(0, CARDS[cand])]
        leader = base[0][0]
        after = [(leader + i) % 4 for i in range(len(base), 4)]
        wins = 0
        signed = 0.0
        for wi in range(n):
            trick = list(base)
            hands = {seat: [CARDS[c] for c in range(32) if int(W[wi, seat]) >> c & 1]
                     for seat in after}
            for j, seat in enumerate(after):
                play = self._respond(trick, seat, hands, after[j + 1:], trump_suit,
                                     vet=self.p['sim_vet'])
                if play is None:
                    continue
                trick.append((seat, play))
                hands[seat] = [c for c in hands[seat] if c != play]
            ws, _ = _winner(trick, trump_suit)
            pts = sum(c.points for _, c in trick)
            if ws % 2 == 0:
                wins += 1
                signed += pts
            else:
                signed -= pts
        return wins / n, signed / n

    @staticmethod
    def _respond(trick, seat, hands, later, trump_suit, vet=True):
        """How a competent player answers in the play-out. The world's hands
        are all known here, so 'can someone behind me still beat it' is
        answered exactly for this world (legal sets from the engine)."""
        led = trick[0][1].suit
        legal = get_legal_moves(hands[seat], led, trick, trump_suit, seat)
        if not legal:
            return None

        def opp_behind_beats(t):
            ws, wc = _winner(t, trump_suit)
            for o in later:
                if o % 2 == ws % 2:
                    continue
                for c in get_legal_moves(hands[o], led, t, trump_suit, o):
                    if c.beats(wc, trump_suit):
                        return True
            return False

        cheap = lambda c: (c.points, c.suit == trump_suit, c.rank)
        ws, wc = _winner(trick, trump_suit)
        if ws % 2 == seat % 2 and not opp_behind_beats(trick):
            # our trick is safe: vet the biggest side card
            if vet:
                return max(legal, key=lambda c: (c.suit != trump_suit, c.points, -c.rank))
            return min(legal, key=cheap)
        secure = []
        for c in legal:
            t = trick + [(seat, c)]
            w, _ = _winner(t, trump_suit)
            if w % 2 == seat % 2 and not opp_behind_beats(t):
                secure.append(c)
        if secure:
            return min(secure, key=lambda c: (c.suit == trump_suit, c.rank))
        winners = [c for c in legal if c.beats(wc, trump_suit)]
        if ws % 2 != seat % 2 and winners:
            return min(winners, key=lambda c: (c.suit == trump_suit, c.rank))
        return min(legal, key=cheap)

    def _pull_ok(self, k):
        """Drawing trumps protects side winners: only worth it with a trump
        majority on our side and a side master to protect."""
        P = self.p
        if not P['pull_needs_reason']:
            return True
        t = k.trump
        mine = sum(1 for c in k.hand if suit_of(c) == t)
        unseen = k.unseen_suit(t)
        opp = sum(k.p_holds(u, 1) + k.p_holds(u, 3) for u in unseen)
        par = sum(k.p_holds(u, 2) for u in unseen)
        majority = mine + par > opp
        side_master = any(suit_of(c) != t and k.is_master(c) for c in k.hand)
        return majority and side_master

    # -- card valuation --------------------------------------------------
    def power(self, k, c):
        """Rough P(card c wins a future trick)."""
        trump = k.trump
        s = suit_of(c)
        higher = k.higher_unseen(c)
        opps = (1, 3)
        if trump is not None and s == trump:
            if not higher:
                return 0.95
            n_opp = sum(k.p_holds(u, o) for u in higher for o in opps)
            return max(0.15, self.p['trump_power'] - self.p['trump_power_slope'] * n_opp)
        n_higher_opp = sum(k.p_holds(u, o) for u in higher for o in opps)
        mine_in_suit = sum(1 for h in k.hand if suit_of(h) == s)
        if trump is not None:
            if self.p['ruff_from_worlds']:
                # P(some opponent is void here and can ruff), from the
                # trump-rule-aware worlds; inflated because voids only grow.
                safe = 1.0
                for o in opps:
                    safe *= 1.0 - k.p_ruff(s, o)
                ruff_risk = min(1.0, (1.0 - safe) * self.p['ruff_future'])
            else:
                opp_trump = max(1.0 - k.p_none(k.unseen_suit(trump), o) for o in opps)
                suit_left = len(k.unseen_suit(s)) + mine_in_suit
                ruff_risk = opp_trump * (0.25 if suit_left >= 5 else
                                         0.5 if suit_left >= 3 else 0.75)
        else:
            ruff_risk = 0.0
        if not higher:
            return 0.9 * (1 - ruff_risk)
        if mine_in_suit - 1 >= len(higher):
            # guarded honour: the guards wait out the higher cards
            return self.p['guard_power'] * (1 - ruff_risk) / (1 + 0.5 * n_higher_opp)
        return 0.1 / (1 + n_higher_opp)

    def keep_value(self, k, c):
        pw = self.power(k, c)
        v = (pw * self.p['keep_win_value']
             + (2 * pw - 1) * PTS[rank_of(c)] * self.p['keep_pts_weight'])
        if self.p['guard_w']:
            v += self.p['guard_w'] * self._guard_duty(k, c)
        if k.trump is not None and suit_of(c) == k.trump:
            # a trump also ruffs: its value is not only its rank
            v += self.p['trump_keep_bonus']
        return v

    def _guard_duty(self, k, c):
        """Points of the honour that c is the last guard of: with c gone the
        honour is bare (blote aas) and falls to whoever leads the master."""
        s = suit_of(c)
        if k.trump is not None and s == k.trump:
            return 0.0
        same = [h for h in k.hand if suit_of(h) == s and h != c]
        if len(same) != 1:
            return 0.0
        h = same[0]
        if rank_of(h) < rank_of(c) or PTS[rank_of(h)] < 3 or k.is_master(h):
            return 0.0
        return float(PTS[rank_of(h)])

    # -- following -------------------------------------------------------
    def _follow(self, k):
        trump = k.trump
        led = suit_of(k.trick[0][1])
        leader = k.trick[0][0]
        n = len(k.trick)
        after = [(leader + i) % 4 for i in range(n + 1, 4)]
        opps_after = [s for s in after if s in (1, 3)]
        partner_after = 2 in after
        ws, wc = k.current_winner()
        trick_pts = sum(PTS[rank_of(c)] for _, c in k.trick)
        rest = len(after) * self.p['trick_rest']

        best, best_ev, why = None, None, ""
        for c in k.legal:
            a = k.act[c]
            wins_now = a[E.ACT_WINS_OFF] > 0
            partner_winning = a[E.ACT_PARTNER_WINNING_OFF] > 0
            if wins_now or partner_winning:
                win_card = c if wins_now else wc
                if wins_now and a[E.ACT_SURE_WIN_OFF] > 0:
                    p_safe = 1.0
                elif wins_now and a[E.ACT_OPP_CAN_BEAT_OFF] == 0:
                    p_safe = 1.0
                else:
                    p_safe = 1.0
                    for o in opps_after:
                        p_safe *= 1.0 - k.p_beats(win_card, o, led)
                p_win = p_safe + ((1 - p_safe) * 0.3 if partner_after else 0.0)
            else:
                p_win = 0.0
                if partner_after:
                    # p_beats honours the follow-suit obligation: partner
                    # can only ruff when void in the led suit.
                    p_part = k.p_beats(wc, 2, led)
                    p_win = p_part * (self.p['partner_after_f'] ** len(opps_after))
            own = PTS[rank_of(c)] * (self.p['win_own_pts_w'] if wins_now else 1.0)
            stake = trick_pts + own + rest
            if self.p['sim'] and k.W is not None:
                p_win, signed = self.simulate_trick(k, c)
                ev = signed - self.keep_value(k, c)
            else:
                ev = (2 * p_win - 1) * stake - self.keep_value(k, c)
            if best_ev is None or ev > best_ev + 1e-9:
                best, best_ev = c, ev
                why = f"p_win={p_win:.2f} stake={stake:.1f} ev={ev:.2f}"
        return best, why

    # -- signalling (partnership convention) -----------------------------
    def _signal_target(self, s):
        """Suit a discard of suit s asks for: itself (direct) or its
        same-colour partner suit (opposite: hearts ask for diamonds, spades
        for clubs, and back)."""
        return s if self.p['signal_mode'] == 1 else s ^ 1

    def _signal_send(self, k):
        """Partner is winning and I cannot follow: partner leads next, so
        instead of vetting, discard a small card that asks for the suit I
        want -- one where I hold the master (A/10), or (opposite only) one I
        am void in while holding trumps, so I can buy it."""
        trump = k.trump
        led = suit_of(k.trick[0][1])
        ws, _ = k.current_winner()
        if ws != 2 or any(suit_of(c) == led for c in k.legal):
            return None
        if self.p['signal_cheap'] and any(
                PTS[rank_of(c)] >= 2 and not k.is_master(c)
                and (trump is None or suit_of(c) != trump) for c in k.legal):
            return None  # something worth vetting: a signal would cost points
        my_trumps = [c for c in k.hand if trump is not None and suit_of(c) == trump]
        wants = []
        for s in range(4):
            if s == trump:
                continue
            mine = [c for c in k.hand if suit_of(c) == s]
            if mine:
                top = max(mine, key=rank_of)
                if k.is_master(top) and PTS[rank_of(top)] >= 4:
                    wants.append((10 + PTS[rank_of(top)], s))
            elif my_trumps and k.unseen_suit(s) and self.p['signal_ruff']:
                wants.append((5, s))
        wants.sort(reverse=True)
        for _, w in wants:
            src = w if self.p['signal_mode'] == 1 else w ^ 1
            if src == trump:
                continue
            pool = sorted([c for c in k.legal if suit_of(c) == src], key=rank_of)
            if not pool:
                continue
            if src == w and len(pool) < 2:
                continue  # keep the master itself
            low = pool[0]
            if PTS[rank_of(low)] > self.p['signal_max_pts'] or (k.is_master(low) and src != w):
                continue
            return low, f"signal: asks partner for {SUIT_STR[w]}"
        return None

    def _signal_asked(self, k):
        """Suit partner asked for on the trick I just won, or None."""
        trump = k.trump
        if len(k.plies) < 4 or len(k.plies) % 4:
            return None
        for seat, card, led, pos, pw in k.plies[-4:]:
            if seat != 2:
                continue
            s = suit_of(card)
            if (not pw or s == led or s == trump
                    or PTS[rank_of(card)] > self.p['signal_max_pts']):
                return None
            w = self._signal_target(s)
            return None if w == trump else w
        return None

    def _signal_read(self, k):
        """I won the last trick; if partner discarded a small side card on
        it while I was winning, lead the suit it asks for."""
        trump = k.trump
        if len(k.plies) < 4 or len(k.plies) % 4:
            return None
        for seat, card, led, pos, pw in k.plies[-4:]:
            if seat != 2:
                continue
            s = suit_of(card)
            if (not pw or s == led or s == trump
                    or PTS[rank_of(card)] > self.p['signal_max_pts']):
                return None
            w = self._signal_target(s)
            if w == trump:
                return None
            mine = sorted([c for c in k.legal if suit_of(c) == w], key=rank_of)
            if not mine:
                return None
            return mine[0], f"signal read: partner asked for {SUIT_STR[w]}"
        return None

    # -- leading ---------------------------------------------------------
    def _lead(self, k):
        if self.p['planner'] and k.W is not None:
            return self._lead_plan(k)
        return self._lead_old(k)

    def _suit_view(self, k, s):
        """Everything the planner wants to know about one side suit."""
        mine = sorted([c for c in k.hand if suit_of(c) == s], key=rank_of)
        unseen = k.unseen_suit(s)
        p_opp_ruff = 1.0 - (1 - k.p_ruff(s, 1)) * (1 - k.p_ruff(s, 3))
        if k.trump is None and self.p['nt_tiers']:
            # no trump: a card no opponent can beat is cashed like a master
            # (the suit both opponents showed out of is ours)
            masters = [c for c in mine if k.is_boss(c)]
        else:
            masters = [c for c in mine if k.is_master(c)]
        top = max(unseen, key=rank_of) if unseen else None
        return dict(
            mine=mine, unseen=unseen, masters=masters,
            p_opp_ruff=p_opp_ruff,
            p_partner_ruff=k.p_ruff(s, 2) * (1 - k.p_ruff(s, 3)),
            p_partner_top=k.p_holds(top, 2) if top is not None else 0.0,
            # opponents still holding this suit: the fewer, the sooner it is ruffed
            opp_len=sum(k.p_holds(u, 1) + k.p_holds(u, 3) for u in unseen),
        )

    def _lead_plan(self, k):
        """Whole-hand lead plan, in the order a good player thinks:
        endgame trump control, draw trumps (only with a reason), partner
        ruff, cash safe masters, establish a suit, lead to partner's
        strength, safe exit."""
        P = self.p
        trump = k.trump
        legal = k.legal
        opps = (1, 3)
        side = [s for s in range(4) if s != trump and any(suit_of(c) == s for c in legal)]
        view = {s: self._suit_view(k, s) for s in side}
        my_tr = sorted([c for c in legal if trump is not None and suit_of(c) == trump],
                       key=rank_of)
        opp_tr = 0.0
        par_tr = 0.0
        if trump is not None:
            for u in k.unseen_suit(trump):
                opp_tr += k.p_holds(u, 1) + k.p_holds(u, 3)
                par_tr += k.p_holds(u, 2)

        skip = P['plan_skip']
        # 0. ruff convention: the opener leads a side singleton at trick 1,
        #    partner returns that suit while the opener can still ruff
        if P['ruff_convention'] and trump is not None:
            if not k.plies and k.declarer not in (0, 2) and my_tr:
                singles = [c for c in legal if suit_of(c) != trump
                           and sum(1 for h in k.hand if suit_of(h) == suit_of(c)) == 1]
                if singles:
                    c = min(singles, key=lambda c: (PTS[rank_of(c)], rank_of(c)))
                    return c, "plan: ruff convention, singleton lead"
            if k.plies and k.plies[0][0] == 2:
                s0 = suit_of(k.plies[0][1])
                followed = any(p[0] == 2 and suit_of(p[1]) == s0 for p in k.plies[1:])
                if s0 != trump and not followed and k.cap[2][trump] >= 0:
                    back = [c for c in legal if suit_of(c) == s0]
                    if back:
                        return min(back, key=rank_of), "plan: ruff convention, return partner's suit"

        # 1. endgame: a master trump first keeps control of the last tricks
        if 1 not in skip and (my_tr and opp_tr > 0.2 and len(k.hand) <= 3 and k.is_master(my_tr[-1])):
            return my_tr[-1], "plan: endgame master trump"

        # 2. draw trumps -- only with a majority and side winners to protect
        if 2 not in skip and my_tr and opp_tr > P['plan_pull_min_opp']:
            majority = len(my_tr) + par_tr > opp_tr + P['plan_majority_margin']
            winners = any(view[s]['masters'] for s in side) or any(
                len(view[s]['mine']) >= 3 for s in side)
            we_declare = k.declarer in (0, 2)
            if majority and winners and (we_declare or len(my_tr) >= 3):
                if k.is_master(my_tr[-1]):
                    return my_tr[-1], f"plan: draw trumps (master) opp~{opp_tr:.1f}"
                if k.declarer == 0 and len(my_tr) >= 3:
                    return my_tr[0], f"plan: declarer draws low opp~{opp_tr:.1f}"
            # partner declared: lead a low trump through to his length
            if (k.declarer == 2 and P['plan_to_declarer']
                    and not k.is_master(my_tr[0]) and PTS[rank_of(my_tr[0])] <= 2):
                if not P['through_fix'] or len(my_tr) >= 2:
                    return my_tr[0], "plan: trump through to declaring partner"

        # 3. partner ruff
        if 3 not in skip and P['plan_ruff'] and par_tr > 0:
            best = None
            for s in side:
                v = view[s]
                if v['p_partner_ruff'] >= P['plan_ruff'] and (best is None or v['p_partner_ruff'] > best[0]):
                    best = (v['p_partner_ruff'], v['mine'][0])
            if best is not None:
                return best[1], f"plan: partner ruff p={best[0]:.2f}"

        # 3b. one enemy trump left and both opponents void in a side suit:
        #     a pointless card there forces that trump out for nothing
        if P['plan_force_trump'] and trump is not None:
            ut = k.unseen_suit(trump)
            if len(ut) == 1 and (k.p_holds(ut[0], 1) + k.p_holds(ut[0], 3)) > 0.7 \
                    and (my_tr or par_tr > 0.5):
                best = None
                for s in side:
                    v = view[s]
                    low = v['mine'][0]
                    both_void = k.p_none(v['unseen'], 1) * k.p_none(v['unseen'], 3) > 0.9 \
                        if v['unseen'] else True
                    if both_void and PTS[rank_of(low)] <= 1 and not k.is_master(low):
                        if best is None or PTS[rank_of(low)] < PTS[rank_of(best)]:
                            best = low
                if best is not None:
                    return best, "plan: force out the last enemy trump"

        # 4. cash safe masters, biggest first (the manille while everyone
        #    still follows suit)
        cash = []
        for s in side:
            v = view[s]
            if not v['masters']:
                continue
            c = v['masters'][-1]
            p_safe = 1.0 - v['p_opp_ruff']
            risk = PTS[rank_of(c)] + k.partner_forced_pts(s)
            need = min(0.95, P['plan_cash'] + P['plan_cash_risk'] * (risk - 4))
            if p_safe >= need:
                cash.append((p_safe * (PTS[rank_of(c)] + 3), c, p_safe))
        if cash and 4 not in skip:
            cash.sort(reverse=True)
            return cash[0][1], f"plan: cash master p_safe={cash[0][2]:.2f}"

        # 5. establish: honour behind missing higher cards, with guards;
        #    lead low (or the top of a touching sequence) to force them out
        est = []
        for s in side:
            v = view[s]
            mine = v['mine']
            if len(mine) < 2 or v['p_opp_ruff'] > P['plan_est_ruff']:
                continue
            hon = [c for c in mine if PTS[rank_of(c)] >= 2 and not k.is_master(c)]
            if not hon:
                continue
            h = hon[-1]
            n_higher = len(k.higher_unseen(h))
            if len(mine) - 1 < n_higher:
                continue
            run = [h]
            r = rank_of(h) - 1
            while r >= 0 and ((s << 3) | r) in mine:
                run.append((s << 3) | r)
                r -= 1
            if P['plan_est_v2']:
                # only one card to drive out, a way back in, and nobody to
                # ruff the established card
                if n_higher != 1 or v['p_opp_ruff'] > P['plan_est_v2_ruff']:
                    continue
                entry = any(c not in mine and (k.is_master(c) or
                            (trump is not None and suit_of(c) == trump))
                            for c in k.hand)
                if not entry:
                    continue
                # the lowest card of the touching run still forces the one
                # higher card out under must-beat (K from A-K drives the 10)
                lead = run[-1] if len(run) >= 2 else mine[0]
            elif trump is None and P['nt_est_low_run']:
                # Q from A-K-Q forces the manille as well and costs less
                lead = run[-1] if len(run) >= 2 else mine[0]
            else:
                # top of a touching sequence (K-Q, Q-J) drives out the higher
                # card without giving a cheap trick; otherwise lead low
                lead = h if len(run) >= 2 else mine[0]
            score = PTS[rank_of(h)] + len(mine) - 2 * n_higher - 3 * v['p_opp_ruff']
            est.append((score, lead, h))
        if est and 5 not in skip:
            est.sort(reverse=True)
            return est[0][1], f"plan: establish {cstr(est[0][2])}"

        # 6. toward partner's strength
        tow = []
        for s in side:
            v = view[s]
            if v['p_partner_top'] >= P['plan_partner_top'] and v['p_opp_ruff'] < 0.5:
                if (trump is None and P['nt_tiers'] and v['p_partner_top'] < 0.9
                        and (k.cap[1][s] < 0) != (k.cap[3][s] < 0)):
                    # no trump, one opponent out of the suit: if the other
                    # one holds the top card his partner vets on it for free
                    continue
                c = v['mine'][0]
                if PTS[rank_of(c)] <= 1 or len(v['mine']) == 1:
                    tow.append((v['p_partner_top'] - 0.2 * PTS[rank_of(c)], c))
        if tow and 6 not in skip:
            tow.sort(reverse=True)
            return tow[0][1], f"plan: toward partner p_top={tow[0][0]:.2f}"

        # 7. safe exit: cheapest card where losing the trick costs least
        if P['plan_exit_sim']:
            return self._lead_sim(k)
        best, best_sc = None, None
        for c in legal:
            s = suit_of(c)
            if trump is not None and s == trump:
                sc = -P['trump_lead_pen'] - PTS[rank_of(c)] - 2 * self.power(k, c)
            else:
                v = view[s]
                mine = v['mine']
                sc = -PTS[rank_of(c)] * 1.5
                sc -= 4 * v['p_opp_ruff']
                if c != mine[0]:
                    sc -= 1.0
                # leading away from a guarded honour gives it up
                if len(mine) == 2 and c == mine[0] and PTS[rank_of(mine[1])] >= 3 \
                        and not k.is_master(mine[1]):
                    sc -= 1.5
                sc += 0.3 * v['opp_len']
                sc -= 0.5 * self.power(k, c) * 3
            if best_sc is None or sc > best_sc:
                best, best_sc = c, sc
        return best, f"plan: safe exit sc={best_sc:.2f}"

    def _lead_old(self, k):
        trump = k.trump
        P = self.p
        opps = (1, 3)

        if trump is not None and P['ruff_first']:
            # partner void + holding trump, the opponent behind him not:
            # lead the lowest card of that suit for partner to ruff.
            best = None
            for s_ in range(4):
                if s_ == trump:
                    continue
                mine = sorted([c for c in k.legal if suit_of(c) == s_], key=rank_of)
                if not mine:
                    continue
                pr = k.p_ruff(s_, 2) * (1 - k.p_ruff(s_, 3))
                if pr >= P['ruff_first'] and (best is None or pr > best[0]):
                    best = (pr, mine[0])
            if best is not None:
                return best[1], f"lead to partner's ruff p={best[0]:.2f}"

        if trump is not None and P['pull_trumps'] and self._pull_ok(k):
            my_trumps = sorted([c for c in k.legal if suit_of(c) == trump], key=rank_of)
            opp_may_trump = any(k.p_holds(u, o) > 0
                                for u in k.unseen_suit(trump) for o in opps)
            we_declare = k.declarer in (0, 2)
            need = P['pull_min_trumps'] if we_declare else P['pull_min_trumps_def']
            if (my_trumps and opp_may_trump and k.is_master(my_trumps[-1])
                    and len(my_trumps) >= need):
                return my_trumps[-1], "pull trumps with master"
            if (P['declarer_pull'] and k.declarer == 0 and opp_may_trump
                    and len(my_trumps) >= P['declarer_pull_min']):
                return my_trumps[0], "declarer pulls trumps low"
            if (P['endgame_master_trump'] and my_trumps and opp_may_trump
                    and len(k.hand) <= 3 and k.is_master(my_trumps[-1])):
                return my_trumps[-1], "endgame master trump"

        if self.p['sim_lead'] == 2 and k.W is not None:
            return self._lead_sim(k)
        cands = []
        for c in k.legal:
            if trump is not None and suit_of(c) == trump:
                continue
            if not k.is_master(c):
                continue
            p_safe = 1.0
            for o in opps:
                p_safe *= 1.0 - k.p_beats(c, o, suit_of(c))
            need = P['cash_threshold']
            if P['cash_risk_w']:
                # points that go if it is ruffed: the card plus what partner
                # is forced to follow with
                risk = PTS[rank_of(c)] + k.partner_forced_pts(suit_of(c))
                need = min(0.95, P['cash_threshold'] + P['cash_risk_w'] * (risk - 4))
            if p_safe >= need:
                cands.append((PTS[rank_of(c)] + 2 * p_safe, c, p_safe))
        if cands:
            cands.sort(reverse=True)
            return cands[0][1], f"cash master p_safe={cands[0][2]:.2f}"

        if self.p['sim_lead'] and k.W is not None:
            return self._lead_sim(k)
        return self._lead_low(k)

    def _lead_sim(self, k):
        """Generic lead: this trick played out over the worlds, plus the
        longer-term lead rules the one-trick view cannot see."""
        P = self.p
        trump = k.trump
        best, best_ev, why = None, None, ""
        for c in k.legal:
            p_win, signed = self.simulate_trick(k, c)
            ev = signed - self.keep_value(k, c)
            s_ = suit_of(c)
            mine = sorted([h for h in k.hand if suit_of(h) == s_], key=rank_of)
            tag = ""
            if trump is not None and s_ == trump:
                # declarer's partner: lead a low trump through to his length
                if (k.declarer == 2 and c == mine[0]
                        and (not P['through_fix'] or len(mine) >= 2)
                        and any(k.p_holds(u, o) > 0
                                for u in k.unseen_suit(trump) for o in (1, 3))):
                    ev += P['to_declarer_bonus']
                    tag = " trump-to-declarer"
            else:
                below = (s_ << 3) | (rank_of(c) - 1) if rank_of(c) > 0 else None
                seq_top = below is not None and below in k.hand
                # an honour without length or a touching card below just
                # gets captured
                if (PTS[rank_of(c)] >= 2 and not k.is_master(c)
                        and len(mine) <= 2 and not seq_top):
                    ev -= P['bare_honour_lead_pen']
                    tag = " unsupported-honour"
            if best_ev is None or ev > best_ev:
                best, best_ev = c, ev
                why = f"sim lead p_win={p_win:.2f} ev={ev:.2f}{tag}"
        return best, why

    def _lead_low(self, k):
        """No safe master to cash: lead the lowest card of the best suit."""
        trump = k.trump
        P = self.p
        opps = (1, 3)
        we_declare = k.declarer in (0, 2)
        best, best_sc, why = None, None, ""
        by_suit = {}
        for c in k.legal:
            by_suit.setdefault(suit_of(c), []).append(c)
        for s, cards in by_suit.items():
            cards.sort(key=rank_of)
            c = cards[0]
            n_mine = len(cards)
            unseen_s = k.unseen_suit(s)
            sc = -P['lead_pts_w'] * PTS[rank_of(c)]
            reasons = []
            if trump is not None and s == trump:
                opp_trumps = any(k.p_holds(u, o) > 0 for u in unseen_s for o in opps)
                if we_declare and n_mine >= P['low_pull_min'] and opp_trumps:
                    sc += P['low_pull_bonus']
                    reasons.append("pull")
                elif (k.declarer == 2 and opp_trumps and P['to_declarer_bonus']
                      and (not P['through_fix'] or n_mine >= 2)):
                    # lead trumps through to the declarer's long trumps
                    sc += P['to_declarer_bonus'] - P['trump_lead_pen'] * 0
                    reasons.append("trump-to-declarer")
                else:
                    sc -= P['trump_lead_pen']
            else:
                # an honour in this suit that is not master but has guards:
                # leading small draws the higher cards out.
                if n_mine >= 2 and any(PTS[rank_of(h)] >= 3 for h in cards[1:]):
                    sc += P['guard_lead_bonus']
                    reasons.append("draw")
                if n_mine == 1 and PTS[rank_of(c)] >= 3:
                    sc -= P['bare_honour_pen']
                top = max(unseen_s, key=rank_of) if unseen_s else None
                if top is not None and not k.is_master(cards[-1]):
                    sc += P['partner_top_w'] * k.p_holds(top, 2)
                if trump is not None and P['ruff_probs']:
                    # weight by P(void here AND holding a trump), from the worlds
                    p_opp = 1.0 - (1 - k.p_ruff(s, 1)) * (1 - k.p_ruff(s, 3))
                    p_par = k.p_ruff(s, 2)
                    sc -= P['opp_ruff_pen'] * p_opp
                    sc += P['partner_ruff_bonus'] * p_par * (1 - k.p_ruff(s, 3))
                    if p_opp > 0.3:
                        reasons.append(f"opp-ruff{p_opp:.1f}")
                    if p_par > 0.3:
                        reasons.append(f"partner-ruff{p_par:.1f}")
                elif trump is not None:
                    tr = k.unseen_suit(trump)
                    for o in opps:
                        if k.cap[o][s] < 0 and k.cap[o][trump] >= 0 and tr:
                            sc -= P['opp_ruff_pen']
                            reasons.append("opp-void")
                    if k.cap[2][s] < 0 and k.cap[2][trump] >= 0 and tr:
                        sc += P['partner_ruff_bonus']
                        reasons.append("partner-ruff")
                if trump is not None:
                    my_trumps = sum(1 for h in k.hand if suit_of(h) == trump)
                    if n_mine == 1 and my_trumps > 0 and PTS[rank_of(c)] <= 1:
                        sc += P['singleton_bonus']
                        reasons.append("make-void")
                sc += P['length_w'] * len(unseen_s)
            if best_sc is None or sc > best_sc:
                best, best_sc, why = c, sc, "low lead " + ",".join(reasons) + f" sc={sc:.2f}"
        return best, why

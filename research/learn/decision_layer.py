"""D1/D2/D3 as argmax of expected match equity (manille_match_plan.md §4c).

Three decisions, one margin net and one equity table:

  D1  the dealer picks a suit or mull        -- queries rel=0
  D2  a defender calls meegaan (x2)          -- queries rel=1 or 3
  D3  the trump-makers answer terug (x2)     -- queries rel=0 or 2

There is no search and no new training loop here: every decision is a lookup
of P(signed margin | my hand, declaration, my seat) against
V(us, them, we_deal). What makes that legitimate is that the multiplier and
the 101 threshold act on the *distribution*, not the mean -- at 94-70 you do
not want the best expected margin, you want the best P(margin >= 7), and a
trailing team wants the variance that a doubled mull provides while a leading
team does not. Expected points cannot express any of that.

The circularity the plan names ("the fixed point that matters"): the per-deal
outcome distribution depends on the declaration policy, which is derived from
V, which is solved from the outcome distribution. `fixed_point` alternates
until the policy stops moving, which takes a handful of sweeps and is minutes
of CPU.

Known approximations, all deliberate and all measurable later:

* The aggregate OutcomeDist fed to the solver is computed at ONE representative
  match state (default 0-0), while the real policy varies with the score. The
  outer iteration absorbs most of this; a state-conditional table would be the
  refinement if the M3 gate says it matters.
* D1 anticipates the opponents' meegaan and its own terug through scalar
  per-declaration rates rather than by integrating over their hands.
* D3's conditioning on "they announced" is only present when the margin net
  was trained with the extra input (see `label_meegaan`). Without it, D3 reads
  an unconditioned distribution and will under-react.
"""
import itertools

import numpy as np

from manille_env.card import permute_hand_mask

from .match_equity import TARGET, OutcomeDist, solve_match_equity
from .match_harness import MULL, DeclarationPolicy, hand_mask

N_BINS = 61
MARGINS = np.arange(-30, 31)
MAX_MULT = 8


class MarginOracle:
    """P(signed margin | hand, declaration, rel) from the trained net.

    `symmetrize` averages the net over all 24 relabellings of the suits.
    The margin distribution is exactly suit-symmetric -- P(m | hand, d) ==
    P(m | sigma(hand), sigma(d)) for every permutation sigma, with mull
    fixed -- but nothing in the encoding or the training data enforces it,
    and the trained net misses it badly: measured on the served policy at
    20k random hands, declaring spades 24.4% of the time against hearts
    17.6% where symmetry forces both to 20.9%, and 18.6% of (hand, sigma)
    pairs changing the declaration outright (scripts/suit_symmetry_probe.py).

    Averaging over the orbit is a projection onto the space of symmetric
    functions, which is where the true distribution lives, so it cannot
    increase squared error against the truth -- the fix is safe in a way
    most inference-time tricks are not. It costs 24 forwards of a small net
    on a once-per-deal decision.
    """

    def __init__(self, net, device="cuda", cond=False, symmetrize=False,
                 chunk=16384):
        self.net = net
        self.device = device
        self.cond = cond  # net was trained with the "they announced" input
        self.symmetrize = symmetrize
        self.chunk = chunk  # bounds the 24x row blow-up on batch call sites

    def dist(self, hand_masks, declarations, rels, announced=None):
        """-> [N, 61] rows summing to 1, margins -30..+30 from the holder's
        own team."""
        hm = np.asarray(hand_masks, dtype=np.uint64).reshape(-1)
        d = np.asarray(declarations, dtype=np.int64).reshape(-1)
        r = np.asarray(rels, dtype=np.int64).reshape(-1)
        a = (np.zeros(len(hm), dtype=np.int64) if announced is None
             else np.asarray(announced, dtype=np.int64).reshape(-1))
        if not self.symmetrize:
            return self._forward(hm, d, r, a)

        # Relabel hand and declaration together; mull (4) has no suit and
        # stays put. rel and announced are suit-free.
        acc = np.zeros((len(hm), N_BINS))
        for perm in itertools.permutations(range(4)):
            pm = np.array([permute_hand_mask(int(m), perm) for m in hm],
                          dtype=np.uint64)
            pd = np.where(d == MULL, MULL, np.take(perm, np.minimum(d, 3)))
            acc += self._forward(pm, pd, r, a)
        return acc / 24.0

    # One-hot tables built once: np.eye leaks ~2.6 KB per call under numpy
    # 2.4.0, and dist() runs it 72 times per symmetrized query -- a 20k-match
    # simulation grew each worker by ~20 GB and took the desktop down twice
    # (2026-11-14). Indexing a cached table does not leak.
    _EYE = {n: np.eye(n, dtype=np.float32) for n in (2, 4, 5)}

    def _forward(self, hm, d, r, a):
        """One raw pass of the net, chunked to bound peak memory."""
        import torch

        out = np.empty((len(hm), N_BINS))
        for lo in range(0, len(hm), self.chunk):
            sl = slice(lo, lo + self.chunk)
            hand = ((hm[sl, None] >> np.arange(32, dtype=np.uint64)[None]) & 1)
            args = [
                torch.from_numpy(hand.astype(np.float32)).to(self.device),
                torch.from_numpy(self._EYE[5][d[sl]]).to(self.device),
                torch.from_numpy(self._EYE[4][r[sl]]).to(self.device),
            ]
            if self.cond:
                args.append(torch.from_numpy(
                    self._EYE[2][a[sl]]).to(self.device))
            with torch.no_grad():
                out[sl] = torch.softmax(
                    self.net(*args), dim=-1).float().cpu().numpy()
        return out


def equity_vector(V, us, them, next_bit, mult, target=TARGET):
    """V after a deal, as a vector over signed margins -30..+30.

    A tie (margin 0) scores nothing and simply passes the deal, so it is a
    lookup at the same score with the dealing bit flipped -- V already has the
    tie self-loop solved into it, so this is not circular."""
    out = np.empty(N_BINS)
    for i, m in enumerate(MARGINS):
        pts = int(m) * mult
        if pts > 0:
            out[i] = V[min(us + pts, target), them, next_bit]
        elif pts < 0:
            out[i] = V[us, min(them - pts, target), next_bit]
        else:
            out[i] = V[us, them, next_bit]
    return out


def base_multiplier(declaration):
    return 2 if declaration == MULL else 1


def _symmetrize_rates(rates):
    """Average a per-declaration rate array over the four suits, leaving mull
    alone. Last axis is the 5 declarations; earlier axes (the score grid of a
    ResponseRates table) pass through untouched.

    These rates are empirical measurements of how often a symmetric opponent
    answers each declaration, so suit symmetry forces the four suit entries to
    be equal and any spread between them is estimator noise plus the margin
    net's own suit bias. The spread is not small: the shipped policy prices
    terug at 0.261 after a hearts declaration and 0.187 after spades, which is
    enough on its own to skew D1's argmax. Symmetrizing the oracle without
    symmetrizing these just moves the bias somewhere else -- it did, when this
    was first wired: the spade over-declaration came back as a hearts one.
    Mull is genuinely different (separate model, doubled stake) and keeps its
    own rate."""
    out = np.array(rates, dtype=np.float64, copy=True)
    out[..., :4] = out[..., :4].mean(axis=-1, keepdims=True)
    return out


class EquityPolicy(DeclarationPolicy):
    """The decision model. `p_meegaan[d]` / `p_terug[d]` are the response
    rates D1 assumes; `fixed_point` re-estimates them each sweep.

    `enable_terug` gates whether WE call terug. `price_terug` gates whether
    D2 (and D1's anticipation) price the risk that THEY will. They are not
    the same thing: M5's gate bundled them (handoff §9), and arm C of the
    three-arm table -- priced, never called -- is what the GUI should ship
    against opponents who can redouble. `price_terug` defaults to
    `enable_terug`, preserving the pre-split behaviour."""

    def __init__(self, oracle, V, p_meegaan=None, p_terug=None,
                 enable_meegaan=True, enable_terug=True, price_terug=None,
                 mull_bias=0.0, target=TARGET, rates=None,
                 symmetrize_rates=False):
        self.oracle = oracle
        self.V = V
        self.target = target
        # Optional ResponseRates table. When present it REPLACES the flat
        # p_meegaan/p_terug constants at pricing time (handoff §11): those
        # were measured once at 0-0 and are wrong at every other score, with
        # opposite signs leading and trailing. The flat arrays stay for
        # back-compat and as the seed the table is built from.
        self.rates = rates
        self.enable_meegaan = enable_meegaan
        self.enable_terug = enable_terug
        self.price_terug = enable_terug if price_terug is None else price_terug
        # Additive tilt on the mull branch's equity, in P(win) units. D1 is
        # an argmax over five predicted distributions whose errors are
        # correlated, so it systematically over-picks whatever the net
        # overestimates -- a winner's curse we measured directly (d_we sits
        # +0.865 above what the defender heads imply). Mull is not symmetric
        # with the four suits (different model, different training
        # distribution, doubled stake), so its share of that error does not
        # cancel against theirs. This exists to find out whether the raw
        # argmax sits at the optimum; nothing proves it does.
        self.mull_bias = float(mull_bias)
        self.p_meegaan = np.zeros(5) if p_meegaan is None else np.asarray(p_meegaan)
        self.p_terug = np.zeros(5) if p_terug is None else np.asarray(p_terug)
        if symmetrize_rates:
            self.p_meegaan = _symmetrize_rates(self.p_meegaan)
            self.p_terug = _symmetrize_rates(self.p_terug)
            if self.rates is not None:
                self.rates = ResponseRates(
                    self.rates.step,
                    _symmetrize_rates(self.rates.p_meegaan),
                    _symmetrize_rates(self.rates.p_terug))

    def _ev(self, p, us, them, next_bit, mult):
        return float(p @ equity_vector(self.V, us, them, next_bit, mult,
                                       self.target))

    def _rates(self, us, them):
        """(p_meegaan[5], p_terug[5]) for a DECLARING team sitting at
        (us, them). Both are properties of the declarer's state, so a
        defender pricing the threat must query the mirrored pair."""
        if self.rates is not None:
            return self.rates.at(us, them)
        return self.p_meegaan, self.p_terug

    def decl_evs(self, hand_masks, us, them):
        """-> [N, 5] expected match equity per declaration -- D1's argmax
        inputs, vectorized. Also the M6.3 sampler tilt: a softmax over this
        vector is the chooser's declaration likelihood."""
        masks = np.asarray(hand_masks).reshape(-1)
        n = len(masks)
        dists = self.oracle.dist(
            np.repeat(masks, 5), np.tile(np.arange(5), n),
            np.zeros(5 * n, dtype=np.int64)).reshape(n, 5, N_BINS)
        out = np.empty((n, 5))
        pm_all, pt_all = self._rates(us, them)
        for d in range(5):
            b = base_multiplier(d)
            pm = pm_all[d] if self.enable_meegaan else 0.0
            pt = pt_all[d] if self.price_terug else 0.0
            # The declaring team deals this hand, so the next one is theirs.
            evs = np.stack([equity_vector(self.V, us, them, 0, k,
                                          self.target)
                            for k in (b, 2 * b, 4 * b)])
            w = np.array([1 - pm, pm * (1 - pt), pm * pt])
            out[:, d] = dists[:, d] @ (w @ evs)
        if self.mull_bias:
            out[:, MULL] += self.mull_bias
        return out

    def declare(self, hand_mask, us, them):
        return int(np.argmax(self.decl_evs([hand_mask], us, them)[0]))

    def meegaan(self, hand_mask, declaration, rel, us, them):
        if not self.enable_meegaan:
            return False
        b = base_multiplier(declaration)
        p = self.oracle.dist([hand_mask], [declaration], [rel])[0]
        # We are the DEFENDERS here, so `us, them` is our own pair and the
        # team that would answer terug sits at the mirror of it.
        pt = (self._rates(them, us)[1][declaration]
              if self.price_terug else 0.0)
        # The defending team does not deal this hand, so the next one is ours.
        v_pass = self._ev(p, us, them, 1, b)
        v_call = ((1 - pt) * self._ev(p, us, them, 1, 2 * b)
                  + pt * self._ev(p, us, them, 1, 4 * b))
        return v_call > v_pass

    def terug(self, hand_mask, declaration, rel, us, them):
        if not self.enable_terug:
            return False
        b = base_multiplier(declaration)
        # Conditioned on the announcement when the net can express it: a
        # meegaan is public evidence that the defenders are strong, which
        # shifts our own margin distribution down.
        p = self.oracle.dist([hand_mask], [declaration], [rel],
                             announced=[1])[0]
        return (self._ev(p, us, them, 0, 4 * b)
                > self._ev(p, us, them, 0, 2 * b))


class ResponseRates:
    """P(meegaan) and P(terug | meegaan) per declaration, as a function of the
    DECLARING team's score, on a coarse grid with nearest-point lookup.

    Replaces the flat constants that `fixed_point` measures once at 0-0 and
    that `decl_evs` then applied at every score (handoff §11). The
    announcement policies were always score-aware; only D1's model of them was
    frozen, and it is wrong in opposite directions at the two ends of the
    match -- leading, it under-prices a double the opponent takes 97% of the
    time; trailing, it counts on one they never give."""

    def __init__(self, step, p_meegaan, p_terug):
        self.step = int(step)
        self.p_meegaan = np.asarray(p_meegaan, dtype=np.float64)  # [G, G, 5]
        self.p_terug = np.asarray(p_terug, dtype=np.float64)

    def _i(self, s):
        return int(np.clip(round(float(s) / self.step), 0,
                           self.p_meegaan.shape[0] - 1))

    def at(self, us, them):
        i, j = self._i(us), self._i(them)
        return self.p_meegaan[i, j], self.p_terug[i, j]


def response_rate_table(policy, deals, step=5, target=TARGET, sweeps=3,
                        tol=0.005, verbose=True):
    """Measure the response rates a symmetric opponent produces at each score.

    Nothing is retrained and V is untouched: the announcement policies already
    take the score, so this is measurement, not fitting. Each grid point runs
    its own small fixed point, because the declarations at that score depend
    on the rates assumed there."""
    import copy

    masks = np.asarray(_masks(deals), dtype=np.uint64)
    n = (target - 1) // step + 1
    pm_tab = np.zeros((n, n, 5))
    pt_tab = np.zeros((n, n, 5))
    probe = copy.copy(policy)

    for i in range(n):
        for j in range(n):
            us, them = i * step, j * step
            pm, pt = policy.p_meegaan.copy(), policy.p_terug.copy()
            for _ in range(sweeps):
                probe.rates = None       # the probe prices with its own flats
                probe.p_meegaan, probe.p_terug = pm, pt
                decls = probe.decl_evs(masks, us, them).argmax(1)
                pm2, pt2 = _response_rates(probe, deals, decls, (us, them))
                done = (np.abs(pm2 - pm).max() < tol
                        and np.abs(pt2 - pt).max() < tol)
                pm, pt = pm2, pt2
                if done:
                    break
            pm_tab[i, j], pt_tab[i, j] = pm, pt
        if verbose:
            print(f"  rates row {i + 1}/{n} (us={i * step})", flush=True)
    return ResponseRates(step, pm_tab, pt_tab)


def aggregate_outcome_dist(oracle, policy, deals, state=(0, 0), rel_us=0):
    """Marginalise the per-hand margin distributions into the single
    OutcomeDist the solver takes, with each deal's multiplier folded in.

    `rel_us=0` gives the distribution for a deal WE declare; `rel_us=1` gives
    a deal THEY declare, seen from our side."""
    us, them = state
    p_we = np.zeros(30 * MAX_MULT + 1)
    p_they = np.zeros(30 * MAX_MULT + 1)
    p_tie = 0.0

    # The declaration is always chosen from the DECLARER's hand, but the
    # distribution must be read from the hand of the seat we are asking
    # about. When they declare (rel_us != 0) those are different hands and
    # conflating them asks "what margin does a defender holding the
    # declarer's cards get", which is meaningless and quietly puts V near 1
    # on both sides.
    #
    # Multipliers are resolved PER DEAL, from the real hands, not from a
    # scalar per-declaration rate. The scalar form is right in
    # EquityPolicy.declare -- D1 cannot see the opponents' cards, so a rate is
    # all it has -- but it is wrong here, because this function runs offline
    # with all four hands available and the call is CORRELATED with the
    # outcome. A flat rate destroys that correlation in opposite directions on
    # the two sides: when we declare the opponents call it on their good
    # outcomes (our bad ones), and when they declare we call it on our good
    # ones. Opposite-signed bias on the two distributions is exactly a
    # mirror-gap generator -- it took V(0,0) from 0.503 to 0.527 and the
    # mirror gap from 0.006 to 0.055 as soon as doubling was switched on.
    decls, own, mults = [], [], []
    for hands, declarer in deals:
        # Scores as the DECLARER's team sees them: when they declare, our
        # (us, them) is their (them, us).
        d_us, d_them = (us, them) if rel_us == 0 else (them, us)
        dec_mask = hand_mask(hands[declarer])
        d = policy.declare(dec_mask, d_us, d_them)
        mult = base_multiplier(d)
        if policy.enable_meegaan:
            called = any(
                policy.meegaan(hand_mask(hands[(declarer + off) % 4]), d, off,
                               d_them, d_us)
                for off in (1, 3))
            if called:
                mult *= 2
                if policy.enable_terug and any(
                        policy.terug(hand_mask(hands[(declarer + off) % 4]),
                                     d, off, d_us, d_them)
                        for off in (0, 2)):
                    mult *= 2
        decls.append(d)
        mults.append(mult)
        own.append(hand_mask(hands[(declarer + rel_us) % 4]))

    dists = oracle.dist(own, decls, [rel_us] * len(decls))
    for k, p in zip(mults, dists):
        for i, mg in enumerate(MARGINS):
            q = p[i]
            if q == 0.0:
                continue
            pts = int(mg) * k
            if pts > 0:
                p_we[pts] += q
            elif pts < 0:
                p_they[-pts] += q
            else:
                p_tie += q
    n = len(decls)
    p_we /= n
    p_they /= n
    p_tie /= n
    # rel_us=1 already signs the margin from our side, so no swap is needed.
    return OutcomeDist(p_we, p_they, p_tie)


def fixed_point(oracle, deals, n_sweeps=6, state=(0, 0), target=TARGET,
                enable_meegaan=True, enable_terug=True, verbose=True):
    """Alternate policy -> outcome distribution -> V -> policy until the
    declarations stop changing. Returns (policy, V)."""
    from .match_harness import RuleTrumpNeverMull

    policy = RuleTrumpNeverMull()
    policy.p_meegaan = np.zeros(5)
    policy.p_terug = np.zeros(5)
    policy.enable_meegaan = False
    policy.enable_terug = False

    prev = None
    V = None
    for sweep in range(n_sweeps):
        d_we = aggregate_outcome_dist(oracle, policy, deals, state, rel_us=0)
        d_they = aggregate_outcome_dist(oracle, policy, deals, state, rel_us=1)
        V = solve_match_equity(d_we, d_they, target=target)

        nxt = EquityPolicy(oracle, V, policy.p_meegaan, policy.p_terug,
                           enable_meegaan, enable_terug, target=target)
        decls = np.array([nxt.declare(m, *state)
                          for m in _masks(deals)])
        if enable_meegaan:
            nxt.p_meegaan, nxt.p_terug = _response_rates(
                nxt, deals, decls, state)
        moved = 1.0 if prev is None else float(np.mean(decls != prev))
        if verbose:
            print(f"  sweep {sweep}: mull {np.mean(decls == MULL):5.1%}, "
                  f"declarations moved {moved:.1%}, "
                  f"meegaan {nxt.p_meegaan.mean():.1%}", flush=True)
        prev, policy = decls, nxt
        if moved == 0.0:
            break
    return policy, V


def _masks(deals):
    out = []
    for hands, declarer in deals:
        m = 0
        for c in hands[declarer]:
            m |= 1 << c.to_index()
        out.append(m)
    return out


def _seat_masks(deals, off):
    """Hand mask of the seat `off` places after each deal's declarer."""
    out = np.empty(len(deals), dtype=np.uint64)
    for i, (hands, declarer) in enumerate(deals):
        m = 0
        for c in hands[(declarer + off) % 4]:
            m |= 1 << c.to_index()
        out[i] = m
    return out


def _announce_decisions(policy, p, decls, us, them, next_bit, call_side):
    """Vectorized D2/D3 rule over a [N, 61] block of margin distributions.

    `call_side=True` is meegaan (pass vs call, pricing the terug reply);
    False is terug (answer vs decline). Same comparisons as the scalar
    methods, grouped by declaration because both the base stake and the
    assumed reply rate depend on it."""
    out = np.zeros(len(decls), dtype=bool)
    _, pt_all = policy._rates(them, us) if call_side else (None, None)
    for d in range(5):
        sel = decls == d
        if not sel.any():
            continue
        b = base_multiplier(d)
        ev = {k: equity_vector(policy.V, us, them, next_bit, k, policy.target)
              for k in (b, 2 * b, 4 * b)}
        block = p[sel]
        if call_side:
            pt = pt_all[d] if policy.price_terug else 0.0
            v_stay = block @ ev[b]
            v_act = (1 - pt) * (block @ ev[2 * b]) + pt * (block @ ev[4 * b])
        else:
            v_stay = block @ ev[2 * b]
            v_act = block @ ev[4 * b]
        out[sel] = v_act > v_stay
    return out


def _response_rates(policy, deals, decls, state):
    """Per-declaration P(meegaan) and P(terug | meegaan) under `policy`, by
    running the actual D2/D3 rules over the real seats of each deal.

    Batched: two oracle calls instead of one per seat per deal. The per-call
    path is not merely slow -- each single-row call fragments the heap by
    ~8 KB that glibc never returns, and at 441 grid points x 3 sweeps that
    OOM-killed a table build at 28.6 GB (2026-07-30). Batching the same
    20,000 rows costs 4 MB instead of 171 MB."""
    us, them = state
    decls = np.asarray(decls, dtype=np.int64)
    n = len(deals)
    mee = np.zeros((5, 2))
    ter = np.zeros((5, 2))

    called = np.zeros(n, dtype=bool)
    if policy.enable_meegaan:
        # Both defenders of every deal in one block; a deal is doubled if
        # either seat calls, matching the scalar loop's short-circuit.
        masks = np.concatenate([_seat_masks(deals, o) for o in (1, 3)])
        p = policy.oracle.dist(masks, np.tile(decls, 2),
                               np.repeat([1, 3], n))
        hit = _announce_decisions(policy, p, np.tile(decls, 2), them, us, 1,
                                  True)
        called = hit[:n] | hit[n:]

    answered = np.zeros(n, dtype=bool)
    if policy.enable_terug and called.any():
        idx = np.flatnonzero(called)
        sub = [deals[i] for i in idx]
        d_sub = decls[idx]
        masks = np.concatenate([_seat_masks(sub, o) for o in (0, 2)])
        p = policy.oracle.dist(masks, np.tile(d_sub, 2),
                               np.repeat([0, 2], len(idx)),
                               announced=np.ones(2 * len(idx), dtype=np.int64))
        hit = _announce_decisions(policy, p, np.tile(d_sub, 2), us, them, 0,
                                  False)
        answered[idx] = hit[:len(idx)] | hit[len(idx):]

    for d in range(5):
        sel = decls == d
        mee[d] = [called[sel].sum(), sel.sum()]
        ter[d] = [answered[sel].sum(), called[sel].sum()]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_m = np.where(mee[:, 1] > 0, mee[:, 0] / np.maximum(mee[:, 1], 1), 0.0)
        p_t = np.where(ter[:, 1] > 0, ter[:, 0] / np.maximum(ter[:, 1], 1), 0.0)
    return p_m, p_t


def equity_choice_log_likelihood(policy, hand_masks, observed_decl,
                                 us=0, them=0, temp=0.01):
    """M6.3: log P(observed_decl | hand) per hand -- softmax over D1's
    expected-equity vector at temperature `temp`. The world-weighting term
    that replaces the hard trump-rule constraint once declarations come
    from the equity chooser; extends Phase 6's choice_log_likelihood to
    five declarations. `observed_decl` is 0..3 for suits and -1 or 4 for
    mull (the two conventions used on either side of the engine boundary).

    The chooser is a deterministic argmax, so its honest likelihood is a
    delta; `temp` is the softening knob, calibrated on the EV scale
    (P(win), gaps ~0.01-0.05) and gated on pts/deal (handoff M6.3)."""
    evs = policy.decl_evs(hand_masks, us, them)
    z = evs / max(temp, 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    obs = 4 if int(observed_decl) in (-1, 4) else int(observed_decl)
    return z[:, obs] - lse


class EquityChoiceModel:
    """The M6.3 sampler tilt as a spec-loadable object: a policy file from
    `m3_fixed_point.py --policy-out`, answering log-likelihood queries over
    reconstructed declarer hands at the representative match state."""

    def __init__(self, policy_path, device="cuda", temp=0.01, us=0, them=0):
        import torch

        from .dmc_margin_train import load_margin_net

        spec = torch.load(policy_path, map_location="cpu",
                          weights_only=False)
        cond = bool(torch.load(spec["net"], map_location="cpu",
                               weights_only=False)
                    .get("config", {}).get("cond", False))
        self.policy = EquityPolicy(
            MarginOracle(load_margin_net(spec["net"], device), device,
                         cond=cond, symmetrize=True),
            spec["V"], spec["p_meegaan"], spec["p_terug"],
            spec["enable_meegaan"], spec["enable_terug"],
            spec.get("price_terug", spec["enable_terug"]),
            symmetrize_rates=True)
        self.temp = temp
        # Fallback score when a caller does not supply one, from the
        # DECLARING team's perspective (D1 queries rel=0).
        self.us, self.them = int(us), int(them)

    def log_likelihood(self, hand_masks, observed_decl, us=None, them=None):
        """log P(observed_decl | hand) at the CURRENT match score.

        The score is not decoration: D1 argmaxes match-win probability, so a
        trailing team declares for variance and a leading one refuses the
        same mull. Evaluating the likelihood at 0-0 while the real chooser
        sat at 90-50 models a declaration nobody made, and the tilt then
        pushes the particle weights toward the wrong hands. Defaults to the
        representative state only because that is what the fixed-point dump
        was built at."""
        if us is None or them is None:
            us, them = self.us, self.them
        return equity_choice_log_likelihood(self.policy, hand_masks,
                                            observed_decl, us=us, them=them,
                                            temp=self.temp)


def label_meegaan(seat_rows, policy, rollouts, state=(0, 0)):
    """D3's conditioning label, computed post-hoc -- no new rollouts.

    For each rollout, ask the D2 policy whether either defender's hand would
    have called meegaan, and stamp that bit on all four of the rollout's rows.
    Retraining the margin net with it as an extra input is what lets D3 react
    to the announcement instead of ignoring it.

    The label is evaluated at one representative match state, so it captures
    "this hand justifies a meegaan" rather than "at 94-70 it does". That is
    the modal case and keeps one labelled dataset usable at every score."""
    from .dmc_margin_expand import ROLE_DEFENDER

    us, them = state
    out = np.zeros(len(seat_rows), dtype=np.uint8)
    order = np.argsort(seat_rows["rid"], kind="stable")
    rid = seat_rows["rid"][order]
    bounds = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        grp = seat_rows[idx]
        called = False
        for r in grp[grp["role"] == ROLE_DEFENDER]:
            if policy.meegaan(int(r["hand"]), int(r["declaration"]),
                              int(r["rel"]), them, us):
                called = True
                break
        out[idx] = called
    return out

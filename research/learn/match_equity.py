"""Match equity for the race to 101 (manille_match_plan.md §4b).

V(us, them, we_deal) = P(our team reaches the target first).

This is the object every announcement decision is scored against. Points
are not the objective — a deal worth +8 expected points implies opposite
declarations at 20-20 and at 95-88, because score accumulates toward an
absorbing target and the multipliers (mull x2, meegaan x2, terug x2) are
variance amplifiers. Trailing teams want variance; leading teams do not.

Two structural facts make this exactly solvable, no value iteration:

1. **Only the winner of a deal scores**, so at most one coordinate of
   (us, them) increases per deal. The state graph is a DAG on the total
   score, apart from ties.
2. **A 30-30 tie scores nothing** and returns to the same state — a self
   loop, which is removed algebraically rather than iterated:
       V = p_tie * V + rest   =>   V = rest / (1 - p_tie)

So sweeping states in order of decreasing total score yields the exact
fixed point in one pass.

The dealing team alternates every hand: the dealer rotates by one seat and
teams are (0,2) / (1,3), so seat parity flips each deal. `we_deal` is
therefore a bit that simply toggles.

The per-deal outcome distribution is an input, not an assumption. Estimate
it by rolling deals out with the frozen play models under the declaration
policy being evaluated (plan §4a), then re-solve: the policy depends on V
and V depends on the policy, so alternate until the policy stops changing
(plan §4b, "the fixed point that matters").
"""
import numpy as np

TARGET = 101


class OutcomeDist:
    """Per-deal score distribution, from the perspective of a fixed team.

    `p_we_score[m]` / `p_they_score[m]` are the probabilities that the deal
    is won by us / by them with a final margin of exactly `m` points AFTER
    the multiplier is applied (so mull and doubling live in here, which is
    what lets one table serve every declaration).

    Index 0 of both arrays is unused: a winner's margin is at least 1, and
    a margin of 0 is the 30-30 tie, carried separately in `p_tie`.
    """

    def __init__(self, p_we_score, p_they_score, p_tie=0.0):
        self.p_we = np.asarray(p_we_score, dtype=np.float64)
        self.p_they = np.asarray(p_they_score, dtype=np.float64)
        self.p_tie = float(p_tie)
        total = self.p_we.sum() + self.p_they.sum() + self.p_tie
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"outcome distribution sums to {total}, not 1")
        if self.p_we[0] or self.p_they[0]:
            raise ValueError("margin 0 is the tie; use p_tie, not index 0")
        if self.p_tie >= 1.0:
            raise ValueError("p_tie = 1 makes the match unwinnable")

    def swapped(self):
        """The same deal seen from the other team's side."""
        return OutcomeDist(self.p_they, self.p_we, self.p_tie)


def solve_match_equity(dist_we_deal, dist_they_deal, target=TARGET):
    """Exact V[us, them, we_deal] = P(we reach `target` first).

    dist_we_deal:   outcome distribution for a deal WE declare
    dist_they_deal: outcome distribution for a deal THEY declare
    Both are from our perspective. Scores are clipped at `target`; reaching
    it is absorbing and decides the match.

    Returns float64 array of shape (target + 1, target + 1, 2).
    """
    dists = (dist_they_deal, dist_we_deal)  # index by we_deal bit
    V = np.zeros((target + 1, target + 1, 2), dtype=np.float64)

    # Absorbing: someone already reached the target. Only one team can,
    # since only one score moves per deal.
    V[target, :target, :] = 1.0
    V[:target, target, :] = 0.0

    # A tie leaves the score alone and passes the deal to the other side,
    # so (us, them, 0) and (us, them, 1) depend on each other at the SAME
    # total. That 2x2 system must be solved as each state is finalised --
    # not in a pass at the end -- because states at higher totals are read
    # by lower ones and must already be correct when they are.
    t = (dists[0].p_tie, dists[1].p_tie)
    denom = 1.0 - t[0] * t[1]
    if denom <= 0.0:
        raise ValueError("p_tie = 1 on both sides: the match never ends")

    # Sweep by decreasing total score. Every non-tie outcome strictly
    # increases the total, so all remaining dependencies point upward.
    for total in range(2 * (target - 1), -1, -1):
        lo = max(0, total - (target - 1))
        for us in range(lo, min(total, target - 1) + 1):
            them = total - us
            a = [0.0, 0.0]
            for deal_bit in (0, 1):
                d = dists[deal_bit]
                nxt = 1 - deal_bit  # the dealing team alternates every hand
                acc = 0.0
                # We win the deal: our score climbs, possibly past target.
                for m in range(1, len(d.p_we)):
                    p = d.p_we[m]
                    if p:
                        acc += p * V[min(us + m, target), them, nxt]
                # They win the deal.
                for m in range(1, len(d.p_they)):
                    p = d.p_they[m]
                    if p:
                        acc += p * V[us, min(them + m, target), nxt]
                a[deal_bit] = acc
            #   V0 = a0 + t0 * V1 ;  V1 = a1 + t1 * V0
            V[us, them, 0] = (a[0] + t[0] * a[1]) / denom
            V[us, them, 1] = (a[1] + t[1] * a[0]) / denom
    return V


def uniform_margin_dist(p_we_win, max_margin=30, p_tie=0.0):
    """Simple test/bootstrap distribution: a win goes to us with
    probability `p_we_win`, margin uniform on 1..max_margin."""
    n = max_margin + 1
    we = np.zeros(n)
    they = np.zeros(n)
    share = (1.0 - p_tie) / max_margin
    we[1:] = share * p_we_win
    they[1:] = share * (1.0 - p_we_win)
    return OutcomeDist(we, they, p_tie)


def simulate_match(dist_we_deal, dist_they_deal, rng, we_deal=True,
                   target=TARGET, max_deals=10000):
    """Play one match by sampling deals. Returns True if we reach `target`
    first. Used to validate the solver, which is the check that matters --
    a miscalibrated equity table silently corrupts every decision above it."""
    us = them = 0
    bit = bool(we_deal)
    for _ in range(max_deals):
        d = dist_we_deal if bit else dist_they_deal
        r = rng.random()
        c = d.p_tie
        if r < c:
            bit = not bit
            continue
        for m in range(1, len(d.p_we)):
            c += d.p_we[m]
            if r < c:
                us += m
                if us >= target:
                    return True
                break
        else:
            for m in range(1, len(d.p_they)):
                c += d.p_they[m]
                if r < c:
                    them += m
                    if them >= target:
                        return False
                    break
        bit = not bit
    raise RuntimeError("match did not terminate")

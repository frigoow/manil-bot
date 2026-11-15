"""Play full matches to 101 (manille_match_plan.md §6).

The gate for every decision-layer step. Deals are played until a team crosses
`WINNING_SCORE`; the dealer rotates one seat clockwise per deal, which is
also what makes `we_deal` in the equity solver a bit that simply toggles.

Duplicate discipline, at match level: the same deal sequence is played twice
with the two declaration policies swapped between teams, so a match pair is
decided by the policies rather than by who was dealt the good hands. A pair
contributes +1 / 0 / -1 (both / split / neither), and the CI is taken over
PAIRS, never over deals.

Card play is deliberately held identical on both sides: the announcement
policy is what is under test, and SPARTA at ~1 s/decision would put a
2000-match gate out of reach (plan §6, "the cost trap"). The play agent is
selected by declaration -- the mull blueprint at no trump, the trump
blueprint otherwise -- because those are different frozen models.
"""
import numpy as np

from manille_env.card import Card, Suit
from manille_env.game_logic import GamePhase, ManilleGame

TARGET = ManilleGame.WINNING_SCORE
MULL = 4  # declaration index: 0-3 are the suits


class DeclarationPolicy:
    """The object under test. All scores are from the deciding team's own
    perspective, and `rel` is (my seat - declarer) % 4, matching the margin
    net's conditioning."""

    def declare(self, hand_mask, us, them):
        raise NotImplementedError

    def meegaan(self, hand_mask, declaration, rel, us, them):
        return False

    def terug(self, hand_mask, declaration, rel, us, them):
        return False


class Exploiter(DeclarationPolicy):
    """An opponent that DECLARES exactly like the policy under test and
    differs only in how it responds to announcements.

    Every gate in this project has been self-play, which is blind by
    construction to anything that only bites against an opponent who plays
    differently -- §9's terug scored ~0 because a copy of us never redoubles,
    and §11's score-conditional pricing scored ~0 for the same reason. This
    is the missing axis: a human does not have to play our strategy.

    Sharing the declaration policy is what makes the number readable. Two
    identical policies split every duplicate pair, so the symmetric baseline
    is exactly 0 and any NEGATIVE edge for the policy under test is a
    response leak, measured in the same pair-edge units as every other gate.
    A positive edge just means the deviation cost the exploiter.

    Kinds, each a read a human could plausibly form:
      mull     always double a mull -- punishes the tell that mull means a
               top-of-distribution hand
      never    never double -- denies the doubling equity D1 prices in
      leading  double whenever ahead -- the read §11 showed we mispriced by
               a factor of 30 at 90-50
    """

    KINDS = ("mull", "never", "leading")

    def __init__(self, base, kind):
        if kind not in self.KINDS:
            raise ValueError(f"unknown exploiter {kind!r}")
        self.base, self.kind = base, kind

    def declare(self, hand_mask, us, them):
        return self.base.declare(hand_mask, us, them)

    def meegaan(self, hand_mask, declaration, rel, us, them):
        if self.kind == "mull":
            return declaration == MULL
        if self.kind == "never":
            return False
        return us > them

    def terug(self, hand_mask, declaration, rel, us, them):
        # "never" refuses every escalation, including its own side's answer;
        # the others answer as the base policy would, so only the meegaan
        # rule is under test.
        if self.kind == "never":
            return False
        return self.base.terug(hand_mask, declaration, rel, us, them)


class RuleTrumpNeverMull(DeclarationPolicy):
    """The M3 control: the standing C++ trump rule, no mull, no doubling."""

    def declare(self, hand_mask, us, them):
        import manille_ismcts as mi
        return int(mi.apply_trump_rule(hand_mask))


def hand_mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c.to_index()
    return m


def play_deal(hands, dealer, declaration, doubles, agent):
    """Run one scored deal. `doubles` is (meegaan_seat_or_None,
    terug_seat_or_None). Returns (winning_team, scored_points); the engine
    applies the multiplier, so mull x2 and the announcements are already in
    the number. A 30-30 tie returns (None, 0)."""
    game = ManilleGame(dealer=dealer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(declaration) if declaration < MULL else None)
    meegaan_seat, terug_seat = doubles
    if meegaan_seat is not None:
        assert game.announce_double(meegaan_seat), "engine refused meegaan"
        if terug_seat is not None:
            assert game.announce_redouble(terug_seat), "engine refused terug"
    game.start_playing()
    agent.new_game(game, dealer, game.current_player)

    pc, pp = [], []
    while game.phase == GamePhase.PLAYING:
        pid = game.current_player
        card = agent.choose_card(game, pid, pc, pp)
        pre = list(game.current_trick)
        assert game.play_card(pid, card), "agent played an illegal card"
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                pc.append(tc_.to_index())
                pp.append(tp_)

    p0, p1 = game.hand_points
    if p0 == p1:
        return None, 0
    win = 0 if p0 > p1 else 1
    return win, (max(p0, p1) - 30) * game.multiplier


def run_match(deals, policies, agents, start_dealer=0, target=TARGET,
              max_deals=200):
    """One match. `policies[t]` governs team t (seats t, t+2); `agents` maps
    a declaration to the play agent for it. Returns (winner, n_deals)."""
    score = [0, 0]
    dealer = start_dealer
    for i in range(max_deals):
        hands = deals[i % len(deals)]
        d_team = dealer % 2
        o_team = 1 - d_team

        declaration = policies[d_team].declare(
            hand_mask(hands[dealer]), score[d_team], score[o_team])

        # Both defenders get the call in seat order from the dealer's left;
        # the first yes takes it, since the engine allows exactly one per
        # side. Same for the answer.
        meegaan_seat = terug_seat = None
        for off in (1, 3):
            seat = (dealer + off) % 4
            if policies[o_team].meegaan(hand_mask(hands[seat]), declaration,
                                        off, score[o_team], score[d_team]):
                meegaan_seat = seat
                break
        if meegaan_seat is not None:
            for off in (0, 2):
                seat = (dealer + off) % 4
                if policies[d_team].terug(hand_mask(hands[seat]), declaration,
                                          off, score[d_team], score[o_team]):
                    terug_seat = seat
                    break

        win, pts = play_deal(hands, dealer, declaration,
                             (meegaan_seat, terug_seat),
                             agents[declaration])
        if win is not None:
            score[win] += pts
            if score[win] >= target:
                return win, i + 1
        dealer = (dealer + 1) % 4
    raise RuntimeError(f"match did not finish in {max_deals} deals")


def run_duplicate_matches(n_matches, policy_a, policy_b, agents, seed=1000,
                          deals_per_match=40, target=TARGET, progress=None):
    """Match-level duplicate: each deal sequence is played twice, with A
    seated as team 0 then as team 1. Returns an array of per-pair results in
    {-1, 0, +1} from A's side -- the unit the CI must be taken over."""
    from .evaluate import make_deals

    out = np.empty(n_matches, dtype=np.int8)
    for m in range(n_matches):
        raw = make_deals(deals_per_match, seed + m)
        deals = [h for h, _ in raw]
        w0, _ = run_match(deals, [policy_a, policy_b], agents,
                          start_dealer=0, target=target)
        w1, _ = run_match(deals, [policy_b, policy_a], agents,
                          start_dealer=0, target=target)
        # A is team 0 in the first replay, team 1 in the second.
        out[m] = (1 if w0 == 0 else -1) + (1 if w1 == 1 else -1)
        out[m] = np.sign(out[m])
        if progress and (m + 1) % progress == 0:
            print(f"  [{m + 1}/{n_matches}] running A-edge "
                  f"{out[:m + 1].mean():+.3f}", flush=True)
    return out


def blueprint_agents(trump_model, mull_model, device="cuda"):
    """Play agents keyed by declaration: the mull blueprint at no trump, the
    trump blueprint for the four suits."""
    from .dmc_agent import QNetAgent
    from .dmc_model import load_q_model

    trump = QNetAgent(load_q_model(trump_model, device), device)
    mull = QNetAgent(load_q_model(mull_model, device), device)
    return {d: (mull if d == MULL else trump) for d in range(5)}

"""Multi-trick mental play-out for the rule agent.

For each candidate card, the hand is played to the end in every sampled
world (worlds come from the trump-constrained sampler, reweighted by the
rule agent's inferences), all four seats following one fast human-style
policy. Legality always comes from the C++ engine (`ManilleGame.valid_moves`);
the card-beats-card table is built once from the engine's own `Card.beats`.
This is a rule-policy roll-out, not a minimax search: nobody optimises
against the others, everyone just plays the human rules with the cards the
world gives them.
"""
import numpy as np

import manille_ismcts as mi
from manille_env.card import Card, Suit

PTS = [0, 0, 0, 1, 2, 3, 4, 5]
_CARDS = [Card.from_index(i) for i in range(32)]


def _build_beats():
    """BEATS[t][a][b]: card a beats card b already on the table, trump t
    (index 4 = no trump). From the engine's Card.beats, never re-derived."""
    tab = np.zeros((5, 32, 32), dtype=bool)
    for t in range(5):
        ts = None if t == 4 else Suit(t)
        for a in range(32):
            for b in range(32):
                tab[t, a, b] = _CARDS[a].beats(_CARDS[b], ts)
    return tab


BEATS = _build_beats()
SUIT_MASK = [0xFF << (8 * s) for s in range(4)]


def _bits(m):
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


class Playout:
    def __init__(self, trump, declarer, honest=True, agent_like=False, nt_rules=0,
                 tr_rules=0):
        self.agent_like = agent_like
        # No-trump card play (bitmask, only read when there is no trump;
        # from the manual review of rule-agent mull games, 2026-11-12):
        #   1  a seat that wins the trick for certain wins it with its
        #      highest honour an opponent could still capture, not with its
        #      cheapest winner (the ace under an outstanding manille is made
        #      while it can be) -- unless enough small cards guard it
        #   2  when partner's card holds and the opponents are out of the
        #      suit, the hand with fewer cards left in it plays high
        #      (unblocks, overtaking if need be), the longer hand plays low
        #      and overtakes partner's last card of the suit
        #   4  a suit both opponents have shown out of is cashed like a
        #      master suit
        #   8  discards keep the cards no opponent can beat (masters and
        #      cards of a suit the opponents are out of: they are tricks and
        #      the link to partner's winners); a loser goes instead, unless
        #      every loser carries 3+ points
        #  16  a discard on an opponents' trick also keeps a guarded honour
        #      and its guards (ace and a small card under the outstanding
        #      manille: the small card ducks, the ace then wins). With only
        #      winners and guarded honours left, a cheap low card of a
        #      three-winner suit goes. On partner's trick the honour is
        #      still vetted: banking it is the percentage play
        #  32  a master is not cashed while its suit is still long outside
        #      (5+ cards, or 4+ when I hold at most two): the others follow
        #      small under it; kept, it captures one of their honours later
        #      (from a SPARTA-judged review of rule-agent mull games,
        #      2026-11-13)
        #  64  back to the suit partner opened the deal with (lowest card),
        #      after cashing masters, before establishing
        # 128  (clairvoyant) a team boss -- the opponents cannot beat it,
        #      partner may hold the master -- is cashed like a master
        # 256  (clairvoyant) nothing to cash: lead the cheapest card of the
        #      suit where partner holds the most bosses
        # 512  establishing from a touching run: lead its lowest card
        #      (Fable review of 50 mull games, 2026-11-14)
        self.nt = nt_rules if trump is None else 0
        # Trump card play (bitmask, only read with a trump suit; from a
        # review of the play-out policy against the blueprint on 4000
        # self-play suit deals, 2026-11-13):
        #   1  trump through to the declaring partner needs 2+ trumps (a
        #      lone small trump is my ruff)
        #   2  following a side suit on partner's safe trick with an
        #      opponent still to play, vet the biggest card even if it is a
        #      master (bank its points now: a master kept for later can be
        #      ruffed)
        #   4  no plan lead: holding trumps, lead a low (<= 1 pt) non-master
        #      side singleton first -- the void makes a later ruff
        #   8  a discard on an opponent's trick that partner (still to play)
        #      can take: put the biggest non-master side card on it
        #  16  a sure winner is taken with the honour that would otherwise
        #      be captured (no-trump rule 1 at trump, ruffs included)
        #  32  ruff convention: partner opened the deal with a side suit and
        #      has not played it since -- return it while partner can ruff
        self.tr = tr_rules if trump is not None else 0
        self.trump = trump
        self.t = 4 if trump is None else trump
        self.beats = BEATS[self.t]
        self.declarer = declarer
        # honest: every seat decides from its own hand + the public record
        # only. The clairvoyant version let seats check the real hands behind
        # them, so they never misjudged a vet and risky lines looked safe.
        self.honest = honest
        self.ruff_short = 2
        self.voids = [0, 0, 0, 0]   # public voids per seat, suit bitmask
        self.first_seat, self.first_card, self.first_followed = -1, -1, False

    def _note(self, seat, card, trick):
        """Record a public void: not following the led suit; and the deal's
        opening lead, and whether its leader played that suit again."""
        if trick and (card >> 3) != (trick[0][1] >> 3):
            self.voids[seat] |= 1 << (trick[0][1] >> 3)
        if self.first_seat < 0:
            self.first_seat, self.first_card = seat, card
        elif seat == self.first_seat and (card >> 3) == (self.first_card >> 3):
            self.first_followed = True

    def _void(self, seat, suit):
        return (self.voids[seat] >> suit) & 1

    # -- the roll-out policy ---------------------------------------------
    def _winner(self, trick):
        ws, wc = trick[0]
        for s, c in trick[1:]:
            if self.beats[c, wc]:
                ws, wc = s, c
        return ws, wc

    def _can_beat(self, hand, x, led):
        """Could a player holding `hand` beat card x on a trick led in
        `led`? (policy estimate: follow if able, else anything)"""
        follow = hand & SUIT_MASK[led]
        pool = follow if follow else hand
        for y in _bits(pool):
            if self.beats[y, x]:
                return True
        return False

    def _is_master(self, c, hand, remaining):
        """No card I cannot see outranks c in its suit."""
        higher = remaining & SUIT_MASK[c >> 3] & ~((1 << (c + 1)) - 1)
        return not (higher & ~hand)

    def _is_boss(self, c, seat, hands):
        """No opponent can beat c in its suit: clairvoyant seats look at
        the opponents' real hands, honest ones need my own master or both
        opponents publicly void."""
        s = c >> 3
        o1, o2 = (seat + 1) % 4, (seat + 3) % 4
        if self.honest:
            remaining = hands[0] | hands[1] | hands[2] | hands[3]
            return self._is_master(c, hands[seat], remaining) or bool(
                self._void(o1, s) and self._void(o2, s))
        higher = SUIT_MASK[s] & ~((1 << (c + 1)) - 1)
        return not ((hands[o1] | hands[o2]) & higher)

    def _discard_classes(self, seat, legal, hands):
        """For a discard (legal = my whole hand): (free, bosses). Bosses are
        the cards no opponent can beat. Per suit, my top honour below them
        is protected together with its guards when I hold at least as many
        smaller cards as there are higher cards out against it. Free cards
        are the rest."""
        hand = hands[seat]
        if self.honest:
            against = (hands[0] | hands[1] | hands[2] | hands[3]) & ~hand
        else:
            against = hands[(seat + 1) % 4] | hands[(seat + 3) % 4]
        boss = [c for c in legal if self._is_boss(c, seat, hands)]
        keep = set(boss)
        for s in range(4):
            mine = [c for c in legal if (c >> 3) == s and c not in keep]
            hon = [c for c in mine if PTS[c & 7] > 0]
            if not hon:
                continue
            h = hon[-1]
            n_hi = bin(against & SUIT_MASK[s] & ~((1 << (h + 1)) - 1)).count("1")
            low = [c for c in mine if c < h]
            if 1 <= n_hi <= len(low):
                keep.add(h)
                keep.update(low[:n_hi])
        return [c for c in legal if c not in keep], boss

    @staticmethod
    def _hold_master(s, hand, outside):
        n_out = bin(outside & SUIT_MASK[s]).count("1")
        return n_out >= 5 or (n_out >= 4 and bin(hand & SUIT_MASK[s]).count("1") <= 2)

    @staticmethod
    def _lowest_of_longest(boss, min_len=1):
        """Lowest card of the suit holding the most bosses (first suit on
        ties), None when no suit has min_len of them."""
        n = [sum(1 for c in boss if (c >> 3) == s) for s in range(4)]
        s = max(range(4), key=lambda x: n[x])
        if n[s] < min_len:
            return None
        return min(c for c in boss if (c >> 3) == s)

    def _doomed_honour(self, secure, seat, hands, led):
        """My highest honour among the sure winners, if winning cheaply
        instead would leave it to be captured: fewer small cards beside it
        than higher cards the opponents (clairvoyant) or anyone else
        (honest) still hold. A guarded honour waits and becomes a master."""
        hon = [c for c in secure if PTS[c & 7] > 0 and not self._is_boss(c, seat, hands)]
        if not hon:
            return None
        h = max(hon)
        higher = SUIT_MASK[led] & ~((1 << (h + 1)) - 1)
        if self.honest:
            rem = hands[0] | hands[1] | hands[2] | hands[3]
            n_hi = bin(rem & ~hands[seat] & higher).count("1")
        else:
            n_hi = bin((hands[(seat + 1) % 4] | hands[(seat + 3) % 4]) & higher).count("1")
        guards = bin(hands[seat] & SUIT_MASK[led]).count("1") - 2
        return h if guards < n_hi else None

    def _lead_agent(self, seat, legal, hands):
        """The rule agent's lead planner, in its fast public-information
        form: what this seat can see is its own hand, the played cards and
        the voids shown so far."""
        trump = self.trump
        hand = hands[seat]
        remaining = hands[0] | hands[1] | hands[2] | hands[3]
        outside = remaining & ~hand
        opps = ((seat + 1) % 4, (seat + 3) % 4)
        partner = (seat + 2) % 4
        my_tr = sorted(c for c in legal if trump is not None and (c >> 3) == trump)
        out_tr = bin(outside & SUIT_MASK[trump]).count("1") if trump is not None else 0
        opp_may_trump = out_tr > 0 and not all(self._void(o, trump) for o in opps)
        side = [c for c in legal if (c >> 3) != trump]
        n_cards = bin(hand).count("1")

        # 0. ruff convention: partner opened the deal with a side suit and has
        # not played it since -- return it while partner can still ruff
        if (self.tr & 32 and self.first_seat == partner and not self.first_followed
                and (self.first_card >> 3) != trump):
            s0 = self.first_card >> 3
            if self.honest:
                can_ruff = not self._void(partner, trump)
            else:
                can_ruff = (not hands[partner] & SUIT_MASK[s0]
                            and bool(hands[partner] & SUIT_MASK[trump]))
            mine0 = [c for c in side if (c >> 3) == s0]
            if can_ruff and mine0:
                return min(mine0)
        # 1. endgame master trump
        if my_tr and opp_may_trump and n_cards <= 3 and self._is_master(my_tr[-1], hand, remaining):
            return my_tr[-1]
        # 2. draw trumps with a likely majority and side winners to protect
        if my_tr and opp_may_trump:
            # partner is assumed to hold a share of the trumps I cannot see
            partner_share = 0 if self._void(partner, trump) else out_tr / 3.0
            majority = len(my_tr) + partner_share > out_tr - partner_share
            winners = any(self._is_master(c, hand, remaining) for c in side) or any(
                bin(hand & SUIT_MASK[s]).count("1") >= 3 for s in range(4) if s != trump)
            we_declare = self.declarer % 2 == seat % 2
            if majority and winners and (we_declare or len(my_tr) >= 3):
                if self._is_master(my_tr[-1], hand, remaining):
                    return my_tr[-1]
                if self.declarer == seat and len(my_tr) >= 3:
                    return my_tr[0]
            # 3. trump through to the declaring partner
            if (self.declarer == partner and n_cards > 2
                    and (len(my_tr) >= 2 or not self.tr & 1)
                    and not self._is_master(my_tr[0], hand, remaining)
                    and PTS[my_tr[0] & 7] <= 2):
                return my_tr[0]
        # 4. cash a side master unless a known-void opponent can ruff it
        best = None
        for c in side:
            s = c >> 3
            if not self._is_master(c, hand, remaining) and not (
                    self.nt & 4 and all(self._void(o, s) for o in opps)) and not (
                    self.nt & 128 and not self.honest and self._is_boss(c, seat, hands)):
                continue
            if self.nt & 32 and self._hold_master(s, hand, outside):
                continue
            if out_tr and any(self._void(o, s) and not self._void(o, trump) for o in opps):
                continue
            if best is None or PTS[c & 7] > PTS[best & 7]:
                best = c
        if best is not None:
            return best
        # 4b. no trump: back to the suit partner opened the deal with
        if self.nt & 64 and self.first_seat == partner:
            back = [c for c in side if (c >> 3) == (self.first_card >> 3)]
            if back:
                return min(back)
        # 4c. no trump, clairvoyant: cross to partner's bosses
        if self.nt & 256 and not self.honest:
            cross = None
            for c in side:
                n_boss = sum(1 for pc in range((c >> 3) * 8, (c >> 3) * 8 + 8)
                             if hands[partner] >> pc & 1 and self._is_boss(pc, partner, hands))
                if n_boss == 0:
                    continue
                key = (-n_boss, PTS[c & 7], c & 7)
                if cross is None or key < cross[0]:
                    cross = (key, c)
            if cross is not None:
                return cross[1]
        # 5. establish: guarded honour, nobody known void to ruff
        est = None
        for s in range(4):
            if s == trump:
                continue
            mine = sorted(c for c in side if (c >> 3) == s)
            if len(mine) < 2:
                continue
            if out_tr and any(self._void(o, s) for o in opps):
                continue
            hon = [c for c in mine if PTS[c & 7] >= 2 and not self._is_master(c, hand, remaining)]
            if not hon:
                continue
            h = hon[-1]
            n_higher = bin(outside & SUIT_MASK[s] & ~((1 << (h + 1)) - 1)).count("1")
            if len(mine) - 1 < n_higher:
                continue
            lead = h if (h - 1) in mine and (h & 7) > 0 else mine[0]
            if self.nt & 512 and lead == h:
                while (lead & 7) > 0 and (lead - 1) in mine:
                    lead -= 1
            score = PTS[h & 7] + len(mine) - 2 * n_higher
            if est is None or score > est[0]:
                est = (score, lead)
        if est is not None:
            return est[1]
        # 6. low from the longest side suit
        if self.tr & 4 and my_tr:
            c = self._short_lead(side, hand, remaining)
            if c is not None:
                return c
        pool = side or legal
        return min(pool, key=lambda c: (-bin(hand & SUIT_MASK[c >> 3]).count("1"),
                                        PTS[c & 7], c & 7))

    def _short_lead(self, side, hand, remaining):
        """The low (<= 1 pt), non-master side singleton to lead (lowest
        points, then rank, then suit), or None: the void makes a later ruff."""
        best = None
        for c in side:
            if (bin(hand & SUIT_MASK[c >> 3]).count("1") == 1 and PTS[c & 7] <= 1
                    and not self._is_master(c, hand, remaining)):
                if best is None or (PTS[c & 7], c & 7) < (PTS[best & 7], best & 7):
                    best = c
        return best

    def _lead(self, seat, legal, hands):
        if self.agent_like:
            return self._lead_agent(seat, legal, hands)
        trump = self.trump
        remaining = hands[0] | hands[1] | hands[2] | hands[3]
        # cash a side master (highest points first)
        best = None
        for c in legal:
            s = c >> 3
            if s == trump:
                continue
            higher = remaining & SUIT_MASK[s] & ~((1 << (c + 1)) - 1)
            if higher & ~hands[seat]:
                continue
            if best is None or PTS[c & 7] > PTS[best & 7]:
                best = c
        if best is not None:
            return best
        my_tr = [c for c in legal if trump is not None and (c >> 3) == trump]
        if my_tr and self.declarer % 2 == seat % 2:
            if self.honest:
                # all trumps I cannot see (partner's included)
                opp_tr = remaining & ~hands[seat] & SUIT_MASK[trump]
            else:
                opp_tr = (hands[(seat + 1) % 4] | hands[(seat + 3) % 4]) & SUIT_MASK[trump]
            if opp_tr:
                return max(my_tr) if all(not self.beats[o, max(my_tr)] for o in _bits(opp_tr)) \
                    else min(my_tr)
        side = [c for c in legal if (c >> 3) != trump] or legal
        # lowest card of the longest side suit
        def key(c):
            return (-bin(hands[seat] & SUIT_MASK[c >> 3]).count("1"), PTS[c & 7], c & 7)
        return min(side, key=key)

    def _follow(self, seat, legal, hands, trick):
        trump = self.trump
        led = trick[0][1] >> 3
        ws, wc = self._winner(trick)
        n = len(trick)
        behind = [(trick[0][0] + i) % 4 for i in range(n + 1, 4)]
        opp_behind = [o for o in behind if o % 2 != seat % 2]

        if self.honest == 2:
            remaining = hands[0] | hands[1] | hands[2] | hands[3]
            outside = remaining & ~hands[seat]
            for s_, c_ in trick:
                outside &= ~(1 << c_)
            led_short = bin(outside & SUIT_MASK[led]).count("1") <= self.ruff_short
            out_tr = trump is not None and (outside & SUIT_MASK[trump]) != 0

            def can_ruff(o):
                return (out_tr and led != trump and not self._void(o, trump)
                        and (self._void(o, led) or led_short))

            def safe(x):
                xs = x >> 3
                for o in opp_behind:
                    if xs == led:
                        if not self._void(o, xs):
                            for y in _bits(outside & SUIT_MASK[xs]):
                                if self.beats[y, x]:
                                    return False
                        if xs != trump and can_ruff(o):
                            return False
                    elif trump is not None and xs == trump and can_ruff(o):
                        for y in _bits(outside & SUIT_MASK[trump]):
                            if self.beats[y, x]:
                                return False
                return True
        elif self.honest:
            # what I cannot see is what the seats behind me might hold
            remaining = hands[0] | hands[1] | hands[2] | hands[3]
            outside = remaining & ~hands[seat]
            for s_, c_ in trick:
                outside &= ~(1 << c_)

            # a ruff behind is only a real threat once the led suit is short
            # (public count of led-suit cards I cannot see)
            ruff_threat = (trump is not None and led != trump
                           and bin(outside & SUIT_MASK[led]).count("1") <= self.ruff_short
                           and (outside & SUIT_MASK[trump]) != 0)

            def safe(card_on_top):
                if not opp_behind:
                    return True
                if ruff_threat and (card_on_top >> 3) != trump:
                    return False
                for y in _bits(outside):
                    if self.beats[y, card_on_top] and (
                            (y >> 3) == (card_on_top >> 3)):
                        return False
                return True
        else:
            def safe(card_on_top):
                return not any(self._can_beat(hands[o], card_on_top, led) for o in opp_behind)

        cheap = lambda c: (PTS[c & 7], (c >> 3) == trump, c & 7)
        if ws % 2 == seat % 2 and safe(wc):
            if self.nt & 2 and (legal[0] >> 3) == led:
                o1, o2 = (seat + 1) % 4, (seat + 3) % 4
                rem = hands[0] | hands[1] | hands[2] | hands[3]
                if self.honest:
                    ours = self._void(o1, led) and self._void(o2, led)
                    p_n = bin(rem & ~hands[seat] & SUIT_MASK[led]).count("1")
                else:
                    ours = not ((hands[o1] | hands[o2]) & SUIT_MASK[led])
                    p_n = bin(hands[(seat + 2) % 4] & SUIT_MASK[led]).count("1")
                if ours:
                    if p_n == 0:
                        # partner led his last card of our suit: take the
                        # lead over (cheaply) if I have more of it to cash
                        over = [c for c in legal if self.beats[c, wc]]
                        return min(over) if over and len(legal) >= 2 else min(legal)
                    return max(legal) if len(legal) <= p_n else min(legal)
            side = [c for c in legal if (c >> 3) != trump] or legal
            if self.agent_like and not (self.tr & 2 and led != trump
                                        and (legal[0] >> 3) == led and opp_behind):
                # the agent's keep-value holds masters back for their own trick
                rem = hands[0] | hands[1] | hands[2] | hands[3]
                if self.nt & 8 and (legal[0] >> 3) != led:
                    keepers = [c for c in side if not self._is_boss(c, seat, hands)]
                else:
                    keepers = [c for c in side if not self._is_master(c, hands[seat], rem)]
                side = keepers or side
            return max(side, key=lambda c: (PTS[c & 7], -(c & 7)))
        secure = []
        winners = []
        for c in legal:
            if self.beats[c, wc]:
                winners.append(c)
                if safe(c):
                    secure.append(c)
        if secure:
            if self.nt & 1 or self.tr & 16:
                h = self._doomed_honour(secure, seat, hands, secure[0] >> 3)
                if h is not None:
                    return h
            return min(secure, key=lambda c: ((c >> 3) == trump, c & 7))
        if ws % 2 != seat % 2 and winners:
            return min(winners, key=lambda c: ((c >> 3) == trump, c & 7))
        p_seat = (seat + 2) % 4
        if (self.tr & 8 and not self.honest and (legal[0] >> 3) != led
                and p_seat in behind and self._can_beat(hands[p_seat], wc, led)):
            rem = hands[0] | hands[1] | hands[2] | hands[3]
            cand = [c for c in legal if (c >> 3) != trump
                    and not self._is_master(c, hands[seat], rem)]
            if cand:
                return max(cand, key=lambda c: (PTS[c & 7], -(c & 7)))
        if self.nt & 16 and (legal[0] >> 3) != led:
            free, boss = self._discard_classes(seat, legal, hands)
            losers = [c for c in free if PTS[c & 7] <= 2]
            if losers:
                return min(losers, key=cheap)
            low = self._lowest_of_longest(boss, 3)
            if low is not None and PTS[low & 7] <= 2:
                return low
            if free:
                return min(free, key=cheap)
            return min(legal, key=cheap)
        if self.nt & 8 and (legal[0] >> 3) != led:
            losers = [c for c in legal if PTS[c & 7] <= 2 and not self._is_boss(c, seat, hands)]
            if losers:
                return min(losers, key=cheap)
        return min(legal, key=cheap)

    # -- one roll-out -----------------------------------------------------
    def run(self, start_hands, history, first, me):
        """start_hands: 4 deal masks (abs seats). history: [(seat, card)]
        public plays so far. first: candidate card for `me`. Returns my
        team's final points."""
        g = mi.ManilleGame(np.asarray(start_hands, dtype=np.int64),
                           -1 if self.trump is None else self.trump, self.declarer)
        g.start_playing()
        hands = list(start_hands)
        trick = []
        self.voids = [0, 0, 0, 0]
        self.first_seat, self.first_card, self.first_followed = -1, -1, False
        for s, c in history:
            self._note(s, c, trick)
            g.do_move(c)
            hands[s] &= ~(1 << c)
            trick.append((s, c))
            if len(trick) == 4:
                trick = []
        self._note(me, first, trick)
        g.do_move(first)
        hands[me] &= ~(1 << first)
        trick.append((me, first))
        if len(trick) == 4:
            trick = []
        while not g.finished():
            seat = g.current_player()
            legal = g.valid_moves()
            if len(legal) == 1:
                c = legal[0]
            elif not trick:
                c = self._lead(seat, legal, hands)
            else:
                c = self._follow(seat, legal, hands, trick)
            self._note(seat, c, trick)
            g.do_move(c)
            hands[seat] &= ~(1 << c)
            trick.append((seat, c))
            if len(trick) == 4:
                trick = []
        return g.team_points(me % 2)


    def choice(self, seat, legal, hands, trick):
        if len(legal) == 1:
            return legal[0]
        if not trick:
            return self._lead(seat, legal, hands)
        return self._follow(seat, legal, hands, trick)

    def likelihood(self, start_hands, history, me, eps):
        """How well this world explains the record: every play by another
        seat that the (honest) roll-out policy would not have made in this
        world multiplies the weight by eps. The policy itself is the
        likelihood -- no hand-written update rules."""
        g = mi.ManilleGame(np.asarray(start_hands, dtype=np.int64),
                           -1 if self.trump is None else self.trump, self.declarer)
        g.start_playing()
        hands = list(start_hands)
        trick = []
        w = 1.0
        self.voids = [0, 0, 0, 0]
        self.first_seat, self.first_card, self.first_followed = -1, -1, False
        for s, c in history:
            if s != me:
                legal = g.valid_moves()
                if len(legal) > 1 and self.choice(s, legal, hands, trick) != c:
                    w *= eps
            self._note(s, c, trick)
            g.do_move(c)
            hands[s] &= ~(1 << c)
            trick.append((s, c))
            if len(trick) == 4:
                trick = []
        return w


def _starts(worlds_abs, history):
    played_by = [0, 0, 0, 0]
    for s, c in history:
        played_by[s] |= 1 << c
    return [[int(w[s]) | played_by[s] for s in range(4)] for w in worlds_abs]


def world_weights(trump, declarer, me, worlds_abs, history, eps, honest=True,
                  agent_like=False, nt_rules=0, tr_rules=0):
    po = Playout(trump, declarer, honest, agent_like, nt_rules, tr_rules)
    return np.array([po.likelihood(st, history, me, eps)
                     for st in _starts(worlds_abs, history)])


def evaluate_candidates(trump, declarer, me, legal, worlds_abs, history,
                        honest=True, agent_like=False, nt_rules=0, tr_rules=0):
    """Final team points per candidate per world, [n_legal][n_worlds]
    (same worlds for every candidate, so comparisons are paired)."""
    po = Playout(trump, declarer, honest, agent_like, nt_rules, tr_rules)
    starts = _starts(worlds_abs, history)
    return {c: np.array([po.run(st, history, c, me) for st in starts], dtype=float)
            for c in legal}

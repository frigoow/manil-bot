"""Observation layout constants and the Python mirror encoder.

The constants must match cpp/encoding.h exactly; the golden parity tests
(tests/test_encoding_parity.py) assert tensor-identical output between the
C++ encoder (authoritative) and the mirror below.
"""

OWN_HAND_OFF = 0        # 32
PLAYED_OFF = 32         # 4x32
TRICK_OFF = 160         # 4x32
TRICK_LEADER_OFF = 288  # 4
TRICK_POS_OFF = 292     # 4
TRUMP_OFF = 296         # 4
DECLARER_OFF = 300      # 4
TRICK_NUM_OFF = 304     # 8
TEAM_PTS_OFF = 312      # 2
TRICK_PTS_OFF = 314     # 1
VOID_OFF = 315          # 4x4
LEGAL_MASK_OFF = 331    # 32
LEGAL_COUNT_OFF = 363   # 1
STATIC_DIM = 364

HISTORY_OFF = 364
PLY_DIM = 58
PLY_FLAG_OFF = 48
HISTORY_DIM = 32 * PLY_DIM

BELIEF_PRIOR_OFF = HISTORY_OFF + HISTORY_DIM          # 96
BELIEF_MASK_OFF = BELIEF_PRIOR_OFF + 96               # 96

# Tactical block (plan 1.3) — must match cpp/encoding.h.
MASTER_OFF = BELIEF_MASK_OFF + 96                     # 32, own cards only
HIGHER_UNSEEN_OFF = MASTER_OFF + 32                   # 32, count/7
UNSEEN_SUIT_OFF = HIGHER_UNSEEN_OFF + 32              # 4, count/8
SEAT_CAN_HOLD_OFF = UNSEEN_SUIT_OFF + 4               # 3 hidden seats x 4 suits
TACTICAL_DIM = 32 + 32 + 4 + 12

OBS_DIM = SEAT_CAN_HOLD_OFF + 12

# Pre-1.3 widths. Every checkpoint trained before the tactical block has an
# input layer this wide; dmc_model.expand_checkpoint_inputs grows one to the
# current layout so an A/B can warm-start from it.
LEGACY_OBS_DIM = 2412
LEGACY_ACT_DIM = 37

# Candidate-action feature block (DMC Q(s,a); must match cpp/encoding.h).
ACT_IS_TRUMP_OFF = 32
ACT_IS_LED_OFF = 33
ACT_RANK_OFF = 34
ACT_POINTS_OFF = 35
ACT_WINS_OFF = 36
ACT_SURE_WIN_OFF = 37
ACT_OPP_CAN_BEAT_OFF = 38
ACT_PARTNER_WINNING_OFF = 39
ACT_DIM = 40
MAX_LEGAL = 8

# Public per-play context flags (must match encoding.h's Flag enum).
F_FOLLOWED = 0
F_VOID_ESTABLISHED = 1
F_OBLIGATION = 2
F_FOLLOWED_NO_BEAT = 3
F_FOLLOWED_BEAT = 4
F_TRUMPED = 5
F_OVERTRUMPED = 6
F_DUCKED_TRUMP = 7
F_PARTNER_WINNING = 8
F_TRICK_COMPLETED = 9
N_FLAGS = 10

CARD_POINTS = [0, 0, 0, 1, 2, 3, 4, 5] * 4  # by card index


def _trick_winner(trick_cards, n, trump):
    """Mirror of ManilleGame::trickWinner (kept in lockstep by parity tests)."""
    winning = trick_cards[0]
    winning_idx = 0
    for i in range(1, n):
        tc = trick_cards[i]
        if tc // 8 == winning // 8:
            if tc % 8 > winning % 8:
                winning, winning_idx = tc, i
        elif tc // 8 == trump:
            winning, winning_idx = tc, i
    return winning_idx


def encode_mirror(
    hands,
    trick_cards,
    trick_players,
    played_cards,
    played_players,
    n_played,
    n_trick,
    current_player,
    trump,
    pts0,
    pts1,
    declarer,
    observer,
    legal_moves,
):
    """Python mirror of encoding::encode (cpp/encoding.cpp).

    Takes the same primitive state as the C++ binding plus the legal-move
    list from the Python engine. Exists for golden parity tests — training
    and inference always use the C++ encoder.
    """
    import numpy as np

    out = np.zeros(OBS_DIM, dtype=np.float32)
    rel = lambda seat: (seat - observer) % 4

    for c in range(32):
        if int(hands[observer]) & (1 << c):
            out[OWN_HAND_OFF + c] = 1.0

    plays = [(int(played_cards[i]), int(played_players[i])) for i in range(n_played)]
    plays += [(int(trick_cards[i]), int(trick_players[i])) for i in range(n_trick)]
    n_plays = len(plays)

    for i in range(n_played):
        card, player = plays[i]
        out[PLAYED_OFF + rel(player) * 32 + card] = 1.0
    for i in range(n_trick):
        card, player = plays[n_played + i]
        out[TRICK_OFF + rel(player) * 32 + card] = 1.0

    voids = [[False] * 4 for _ in range(4)]
    for i, (card, actor) in enumerate(plays):
        trick_start = (i // 4) * 4
        pos = i % 4
        led_card = plays[trick_start][0]
        led_suit = led_card // 8
        card_suit = card // 8

        if card_suit != led_suit:
            voids[actor][led_suit] = True

        base = HISTORY_OFF + i * PLY_DIM
        out[base + rel(actor)] = 1.0
        out[base + 4 + card] = 1.0
        out[base + 36 + led_suit] = 1.0
        out[base + 40 + pos] = 1.0

        winner_card = winner_seat = -1
        beats = False
        if pos > 0:
            table = [plays[trick_start + k][0] for k in range(pos)]
            widx = _trick_winner(table, pos, trump)
            winner_card, winner_seat = table[widx], plays[trick_start + widx][1]
            out[base + 44 + rel(winner_seat)] = 1.0
            beats = (
                card_suit == trump and winner_card // 8 != trump
            ) or (
                card_suit == winner_card // 8 and card % 8 > winner_card % 8
            )

        partner = (actor + 2) % 4
        followed = card_suit == led_suit
        obligation = pos > 0 and winner_seat != partner

        f = base + PLY_FLAG_OFF
        out[f + F_FOLLOWED] = float(followed)
        out[f + F_VOID_ESTABLISHED] = float(not followed)
        out[f + F_OBLIGATION] = float(obligation)
        out[f + F_FOLLOWED_NO_BEAT] = float(followed and obligation and not beats)
        out[f + F_FOLLOWED_BEAT] = float(followed and beats)
        out[f + F_TRUMPED] = float(not followed and card_suit == trump)
        out[f + F_OVERTRUMPED] = float(
            not followed and card_suit == trump and pos > 0
            and winner_card // 8 == trump and beats
        )
        out[f + F_DUCKED_TRUMP] = float(
            obligation and not beats and not followed and winner_card // 8 == trump
        )
        out[f + F_PARTNER_WINNING] = float(pos > 0 and winner_seat == partner)
        out[f + F_TRICK_COMPLETED] = float(pos == 3)

    trick_leader = plays[n_played][1] if n_trick > 0 else current_player
    out[TRICK_LEADER_OFF + rel(trick_leader)] = 1.0
    out[TRICK_POS_OFF + n_trick] = 1.0
    if trump >= 0:
        out[TRUMP_OFF + trump] = 1.0
    if declarer >= 0:
        out[DECLARER_OFF + rel(declarer)] = 1.0
    out[TRICK_NUM_OFF + n_played // 4] = 1.0

    own_team = observer % 2
    own_pts, opp_pts = (pts0, pts1) if own_team == 0 else (pts1, pts0)
    out[TEAM_PTS_OFF + 0] = own_pts / 60.0
    out[TEAM_PTS_OFF + 1] = opp_pts / 60.0

    trick_pts = sum(CARD_POINTS[plays[n_played + i][0]] for i in range(n_trick))
    out[TRICK_PTS_OFF] = trick_pts / 20.0

    for seat in range(4):
        for suit in range(4):
            if voids[seat][suit]:
                out[VOID_OFF + rel(seat) * 4 + suit] = 1.0

    if observer == current_player:
        for c in legal_moves:
            out[LEGAL_MASK_OFF + c] = 1.0
        out[LEGAL_COUNT_OFF] = len(legal_moves) / 8.0

    seen = int(hands[observer])
    for card, _ in plays:
        seen |= 1 << card

    for c in range(32):
        if seen & (1 << c):
            continue
        suit = c // 8
        allowed = [not voids[(observer + r + 1) % 4][suit] for r in range(3)]
        n_allowed = sum(allowed)
        if n_allowed == 0:
            continue
        for r in range(3):
            out[BELIEF_MASK_OFF + r * 32 + c] = float(allowed[r])
            out[BELIEF_PRIOR_OFF + r * 32 + c] = (
                1.0 / n_allowed if allowed[r] else 0.0
            )

    unseen = [c for c in range(32) if not seen & (1 << c)]
    unseen_suit = [0] * 4
    higher = [0] * 32
    for c in unseen:
        unseen_suit[c // 8] += 1
        for k in range(c // 8 * 8, c // 8 * 8 + 8):
            if c % 8 > k % 8:
                higher[k] += 1
    for c in range(32):
        if int(hands[observer]) & (1 << c):
            out[MASTER_OFF + c] = float(higher[c] == 0)
            out[HIGHER_UNSEEN_OFF + c] = higher[c] / 7.0
    for suit in range(4):
        out[UNSEEN_SUIT_OFF + suit] = unseen_suit[suit] / 8.0
    for r in range(3):
        seat = (observer + r + 1) % 4
        for suit in range(4):
            out[SEAT_CAN_HOLD_OFF + r * 4 + suit] = float(
                not voids[seat][suit] and unseen_suit[suit] > 0
            )

    return out


def _beats(card, winner, trump):
    """Mirror of Card::beats (the engine's trick comparison)."""
    return (card // 8 == trump and winner // 8 != trump) or (
        card // 8 == winner // 8 and card % 8 > winner % 8
    )


def _can_beat(seat, hand_mask, trick_pairs, trump, leading, dealer):
    """Could `seat`, holding `hand_mask`, legally beat `leading`?

    Legality comes from the PYTHON engine, deliberately: the C++ side asks
    ManilleGame::validMoves for the same hypothetical hand, so the parity
    test compares two independent engines rather than one engine against a
    hand-written rule.
    """
    from manille_env.card import Card, Suit
    from manille_env.game_logic import ManilleGame

    game = ManilleGame(dealer=dealer)
    game.hands = [[] for _ in range(4)]
    game.hands[seat] = [
        Card.from_index(c) for c in range(32) if hand_mask >> c & 1
    ]
    game.select_trump(Suit(trump) if trump >= 0 else None)
    game.start_playing()
    game.current_trick = list(trick_pairs)
    game.current_player = seat
    return any(
        _beats(c.to_index(), leading, trump)
        for c in game.get_legal_moves(seat)
    )


def encode_action_mirror(
    hands,
    trick_cards,
    trick_players,
    played_cards,
    played_players,
    n_played,
    n_trick,
    current_player,
    trump,
    declarer,
    card,
):
    """Python mirror of encoding::encode_action (cpp/encoding.cpp).

    Candidate features are always encoded for the seat to move, matching
    every C++ call site (all of them enumerate validMoves()).
    """
    import numpy as np

    from manille_env.card import Card

    out = np.zeros(ACT_DIM, dtype=np.float32)
    me = current_player
    suit = card // 8
    out[card] = 1.0
    out[ACT_IS_TRUMP_OFF] = float(trump >= 0 and suit == trump)
    out[ACT_RANK_OFF] = (card % 8) / 7.0
    out[ACT_POINTS_OFF] = CARD_POINTS[card] / 5.0

    table = [int(trick_cards[i]) for i in range(n_trick)]
    seats = [int(trick_players[i]) for i in range(n_trick)]
    if n_trick == 0:
        out[ACT_WINS_OFF] = 1.0
    else:
        out[ACT_IS_LED_OFF] = float(suit == table[0] // 8)
        widx = _trick_winner(table, n_trick, trump)
        out[ACT_WINS_OFF] = float(_beats(card, table[widx], trump))
        out[ACT_PARTNER_WINNING_OFF] = float(seats[widx] == (me + 2) % 4)

    n_after = n_trick + 1
    if n_after >= 4:
        return out

    plays = [(int(played_cards[i]), int(played_players[i]))
             for i in range(n_played)]
    plays += list(zip(table, seats))
    voids = [[False] * 4 for _ in range(4)]
    for i, (c, actor) in enumerate(plays):
        led = plays[(i // 4) * 4][0] // 8
        if c // 8 != led:
            voids[actor][led] = True
    seen = int(hands[me])
    for c, _ in plays:
        seen |= 1 << c
    unseen = [c for c in range(32) if not seen & (1 << c)]

    after = table + [card]
    after_seats = seats + [me]
    lead_idx = _trick_winner(after, n_after, trump)
    leading = after[lead_idx]
    trick_pairs = [(s, Card.from_index(c))
                   for s, c in zip(after_seats, after)]
    sure_win = after_seats[lead_idx] == me
    opp_can_beat = False
    led_suit = after[0] // 8
    for k in range(n_after, 4):
        seat = (me + 1 + (k - n_after)) % 4
        base = 0
        for c in unseen:
            if not voids[seat][c // 8]:
                base |= 1 << c
        if not base:
            continue
        # Two hypothetical hands, not one: legality is not monotonic in the
        # hand (holding led-suit cards REMOVES the option to trump), so the
        # question is whether any hand the seat could hold makes a beating
        # reply legal. Mirror of the C++ comment in encode_action_responses.
        no_led = base & ~(0xFF << (led_suit * 8))
        if not any(
            _can_beat(seat, hand, trick_pairs, trump, leading,
                      declarer if declarer >= 0 else 0)
            for hand in (base, no_led) if hand
        ):
            continue
        sure_win = False
        if seat % 2 != me % 2:
            opp_can_beat = True
    out[ACT_SURE_WIN_OFF] = float(sure_win)
    out[ACT_OPP_CAN_BEAT_OFF] = float(opp_can_beat)
    return out

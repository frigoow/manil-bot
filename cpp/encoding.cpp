#include "encoding.h"

#include <algorithm>

#include <array>
#include <cstring>

namespace encoding {
namespace {

int suitOf(int c) { return c / 8; }
int rankOf(int c) { return c % 8; }
int cardPoints(int c) { return Card::from_index(c).points(); }

struct Play {
    int card;
    int player;
};

// Public play record: completed tricks followed by the in-progress trick.
int publicPlays(const ManilleGame& game, std::array<Play, 32>& plays) {
    const int n_completed = game.playedCardCount();
    for (int i = 0; i < n_completed; ++i)
        plays[i] = {game.playedCard(i), game.playedPlayer(i)};
    // A finished deal still reports its last trick as the table, though those
    // cards are already among the 32 completed plays: never count them twice.
    const int n_trick = std::min(game.trickSize(), 32 - n_completed);
    for (int i = 0; i < n_trick; ++i)
        plays[n_completed + i] = {game.trickCard(i), game.trickPlayer(i)};
    return n_completed + n_trick;
}

// Established voids. Following suit is mandatory, so any play off the led
// suit is a hard public fact about the actor.
void publicVoids(const std::array<Play, 32>& plays, int n_plays,
                 bool voids[4][4]) {
    for (int seat = 0; seat < 4; ++seat)
        for (int suit = 0; suit < 4; ++suit)
            voids[seat][suit] = false;
    for (int i = 0; i < n_plays; ++i) {
        const int led_suit = suitOf(plays[(i / 4) * 4].card);
        if (suitOf(plays[i].card) != led_suit)
            voids[plays[i].player][led_suit] = true;
    }
}

// Cards the observer cannot see: not in its hand, not played, not on table.
int64_t unseenMask(const ManilleGame& game, int observer,
                   const std::array<Play, 32>& plays, int n_plays) {
    int64_t seen = game.handMask(observer);
    for (int i = 0; i < n_plays; ++i)
        seen |= int64_t(1) << plays[i].card;
    return ~seen & 0xFFFFFFFFLL;
}

// Every unseen card in the suits `seat` is not publicly void in: the largest
// hand `seat` could hold. Asking the engine what is legal from this hand
// gives an upper bound on what that seat can do.
int64_t possibleHand(int64_t unseen, const bool voids[4][4], int seat) {
    int64_t mask = 0;
    for (int suit = 0; suit < 4; ++suit)
        if (!voids[seat][suit])
            mask |= unseen & (int64_t(0xFF) << (suit * 8));
    return mask;
}

// Shared tail: what the seats still to play can do about this candidate.
// Split out of encode_action so the lead case (no cards on the table yet)
// runs it too -- "nobody can beat my lead" is exactly the master-card
// question, and returning early there would have silently dropped it.
void encode_action_responses(const ManilleGame& game, int card, float* out) {
    const int trump = game.trumpSuit();
    const int me = static_cast<int>(game.currentPlayer());
    const int n_trick = game.trickSize();
    const int n_after = n_trick + 1;
    if (n_after >= 4)
        return;  // trick closes on this card; nobody is left to respond

    std::array<Play, 32> plays{};
    const int n_plays = publicPlays(game, plays);
    bool voids[4][4];
    publicVoids(plays, n_plays, voids);
    const int64_t unseen = unseenMask(game, me, plays, n_plays);

    // The trick as it stands after the candidate is played.
    std::array<int32_t, 4> tc{};
    std::array<int32_t, 4> tp{};
    for (int i = 0; i < n_trick; ++i) {
        tc[i] = game.trickCard(i);
        tp[i] = game.trickPlayer(i);
    }
    tc[n_trick] = card;
    tp[n_trick] = me;
    const int lead_widx = ManilleGame::trickWinner(tc, n_after, trump);
    const Card leading = Card::from_index(tc[lead_widx]);
    // A candidate that is already losing cannot be a sure win.
    bool sure_win = tp[lead_widx] == me;
    bool opp_can_beat = false;

    const int led_suit = suitOf(tc[0]);
    const int64_t led_mask = int64_t(0xFF) << (led_suit * 8);
    for (int k = n_after; k < 4; ++k) {
        const int seat = (me + 1 + (k - n_after)) % 4;
        const int64_t base = possibleHand(unseen, voids, seat);
        if (base == 0)
            continue;
        // Legality is NOT monotonic in the hand: handing a seat every unseen
        // card makes it able to FOLLOW the led suit, which removes its option
        // to trump. So one "largest possible hand" understates what the seat
        // can do, and a card that any trump beats would be reported as a sure
        // win. Ask instead whether ANY hand it could hold makes a beating
        // reply legal -- two cases span that: it can follow, or it is void in
        // the led suit. A void is never publicly excluded (only established),
        // so the second case is always consistent.
        const int64_t hypotheticals[2] = {base, base & ~led_mask};
        bool beats = false;
        for (const int64_t hand : hypotheticals) {
            if (hand == 0 || beats)
                continue;
            std::array<int64_t, 4> hands{};
            hands[seat] = hand;
            // Legality comes from the engine, on the trick as it stands. A
            // later seat only ever sees plays that did NOT beat this card
            // (one that did already clears sure_win via its own seat), so the
            // winner it faces is this card and the question is unchanged.
            const ManilleGame probe(hands, tc, tp, n_after, seat, trump,
                                    game.teamPoints(0), game.teamPoints(1),
                                    game.tricksWon(0), game.tricksWon(1),
                                    false, game.declarer());
            for (const Card& reply : probe.validMoves()) {
                if (reply.beats(leading, trump)) {
                    beats = true;
                    break;
                }
            }
        }
        if (!beats)
            continue;
        sure_win = false;
        if (seat % 2 != me % 2)
            opp_can_beat = true;
    }
    out[ACT_SURE_WIN_OFF] = sure_win ? 1.0f : 0.0f;
    out[ACT_OPP_CAN_BEAT_OFF] = opp_can_beat ? 1.0f : 0.0f;
}

} // namespace

void encode_action(const ManilleGame& game, int card, float* out) {
    std::memset(out, 0, ACT_DIM * sizeof(float));
    const int trump = game.trumpSuit();
    const int suit = suitOf(card);
    const int n_trick = game.trickSize();

    out[card] = 1.0f;
    out[ACT_IS_TRUMP_OFF] = (trump >= 0 && suit == trump) ? 1.0f : 0.0f;
    out[ACT_RANK_OFF] = rankOf(card) / 7.0f;
    out[ACT_POINTS_OFF] = cardPoints(card) / 5.0f;

    if (n_trick == 0) {
        // A lead card is trivially winning-so-far.
        out[ACT_WINS_OFF] = 1.0f;
        encode_action_responses(game, card, out);
        return;
    }
    const int led_suit = suitOf(game.trickCard(0));
    out[ACT_IS_LED_OFF] = (suit == led_suit) ? 1.0f : 0.0f;

    // Compare against the current winner with the engine's own trick logic.
    std::array<int32_t, 4> tc{};
    for (int i = 0; i < n_trick; ++i)
        tc[i] = game.trickCard(i);
    const int widx = ManilleGame::trickWinner(tc, n_trick, trump);
    const Card winner = Card::from_index(game.trickCard(widx));
    out[ACT_WINS_OFF] = Card::from_index(card).beats(winner, trump) ? 1.0f
                                                                    : 0.0f;
    const int me = static_cast<int>(game.currentPlayer());
    out[ACT_PARTNER_WINNING_OFF] =
        (game.trickPlayer(widx) == (me + 2) % 4) ? 1.0f : 0.0f;
    encode_action_responses(game, card, out);
}

void encode(const ManilleGame& game, int observer, float* out) {
    std::memset(out, 0, OBS_DIM * sizeof(float));

    const int trump = game.trumpSuit();
    const auto rel = [observer](int seat) { return (seat - observer + 4) % 4; };

    // Own hand — the only private information in the whole vector.
    const int64_t own = game.handMask(observer);
    for (int c = 0; c < 32; ++c)
        if (own & (int64_t(1) << c))
            out[OWN_HAND_OFF + c] = 1.0f;

    // Reconstruct the full public play sequence: completed tricks followed by
    // the in-progress trick.
    std::array<Play, 32> plays{};
    const int n_completed = game.playedCardCount();
    for (int i = 0; i < n_completed; ++i)
        plays[i] = {game.playedCard(i), game.playedPlayer(i)};
    // finished deal: the last trick is already among the completed plays
    const int n_trick = std::min(game.trickSize(), 32 - n_completed);
    for (int i = 0; i < n_trick; ++i)
        plays[n_completed + i] = {game.trickCard(i), game.trickPlayer(i)};
    const int n_plays = n_completed + n_trick;

    // Played cards per relative seat (completed tricks) and cards currently
    // on the table per relative seat.
    for (int i = 0; i < n_completed; ++i)
        out[PLAYED_OFF + rel(plays[i].player) * 32 + plays[i].card] = 1.0f;
    for (int i = 0; i < n_trick; ++i)
        out[TRICK_OFF + rel(plays[n_completed + i].player) * 32
            + plays[n_completed + i].card] = 1.0f;

    // Replay the sequence to derive per-play public context and the void
    // flags. Not following suit always means the player *couldn't* follow
    // (following is mandatory), so every violation is a hard public fact.
    bool voids[4][4] = {};
    publicVoids(plays, n_plays, voids);
    for (int i = 0; i < n_plays; ++i) {
        const int trick_start = (i / 4) * 4;
        const int pos = i % 4;
        const int led_card = plays[trick_start].card;
        const int led_suit = suitOf(led_card);
        const int actor = plays[i].player;
        const int card = plays[i].card;
        const int card_suit = suitOf(card);

        float* p = out + HISTORY_OFF + i * PLY_DIM;
        p[rel(actor)] = 1.0f;
        p[4 + card] = 1.0f;
        p[36 + led_suit] = 1.0f;
        p[40 + pos] = 1.0f;

        int winner_card = -1;
        int winner_seat = -1;
        bool beats = false;
        if (pos > 0) {
            std::array<int32_t, 4> tc{};
            for (int k = 0; k < pos; ++k)
                tc[k] = plays[trick_start + k].card;
            const int widx = ManilleGame::trickWinner(tc, pos, trump);
            winner_card = plays[trick_start + widx].card;
            winner_seat = plays[trick_start + widx].player;
            p[44 + rel(winner_seat)] = 1.0f;
            // Same comparison the engine's trickWinner applies.
            beats = (card_suit == trump && suitOf(winner_card) != trump)
                || (card_suit == suitOf(winner_card)
                    && rankOf(card) > rankOf(winner_card));
        }

        const int partner = (actor + 2) % 4;
        const bool followed = card_suit == led_suit;
        const bool obligation = pos > 0 && winner_seat != partner;

        float* f = p + PLY_FLAG_OFF;
        f[F_FOLLOWED] = followed;
        f[F_VOID_ESTABLISHED] = !followed;
        f[F_OBLIGATION] = obligation;
        f[F_FOLLOWED_NO_BEAT] = followed && obligation && !beats;
        f[F_FOLLOWED_BEAT] = followed && beats;
        f[F_TRUMPED] = !followed && card_suit == trump;
        f[F_OVERTRUMPED] = !followed && card_suit == trump && pos > 0
                           && suitOf(winner_card) == trump && beats;
        f[F_DUCKED_TRUMP] = obligation && !beats && !followed
                            && suitOf(winner_card) == trump;
        f[F_PARTNER_WINNING] = pos > 0 && winner_seat == partner;
        f[F_TRICK_COMPLETED] = pos == 3;
    }

    // Static blocks.
    const int trick_leader = n_trick > 0 ? plays[n_completed].player
                                         : static_cast<int>(game.currentPlayer());
    out[TRICK_LEADER_OFF + rel(trick_leader)] = 1.0f;
    out[TRICK_POS_OFF + n_trick] = 1.0f;
    if (trump >= 0)
        out[TRUMP_OFF + trump] = 1.0f;
    if (game.declarer() >= 0)
        out[DECLARER_OFF + rel(game.declarer())] = 1.0f;
    out[TRICK_NUM_OFF + n_completed / 4] = 1.0f;

    const int own_team = observer % 2;
    out[TEAM_PTS_OFF + 0] = game.teamPoints(own_team) / 60.0f;
    out[TEAM_PTS_OFF + 1] = game.teamPoints(1 - own_team) / 60.0f;

    int trick_pts = 0;
    for (int i = 0; i < n_trick; ++i)
        trick_pts += cardPoints(game.trickCard(i));
    out[TRICK_PTS_OFF] = trick_pts / 20.0f;

    for (int seat = 0; seat < 4; ++seat)
        for (int suit = 0; suit < 4; ++suit)
            if (voids[seat][suit])
                out[VOID_OFF + rel(seat) * 4 + suit] = 1.0f;

    // The legal mask only exists for the player to move; it is all zeros
    // when encoding someone else's perspective.
    if (observer == static_cast<int>(game.currentPlayer())) {
        int n_legal = 0;
        for (const Card& move : game.validMoves()) {
            out[LEGAL_MASK_OFF + move.index()] = 1.0f;
            ++n_legal;
        }
        out[LEGAL_COUNT_OFF] = n_legal / 8.0f;
    }

    // Belief prior and mask over the 3 hidden seats. Seen cards (own hand,
    // played, on the table) are excluded entirely; seats with an established
    // void in the card's suit are masked out. The prior is uniform over the
    // remaining seats — this is the constrained-uniform baseline of plan §5,
    // and zero network output must reproduce it exactly.
    int64_t seen = own;
    for (int i = 0; i < n_plays; ++i)
        seen |= int64_t(1) << plays[i].card;

    for (int c = 0; c < 32; ++c) {
        if (seen & (int64_t(1) << c))
            continue;
        const int suit = suitOf(c);
        bool allowed[3];
        int n_allowed = 0;
        for (int r = 0; r < 3; ++r) {
            const int abs_seat = (observer + r + 1) % 4;
            allowed[r] = !voids[abs_seat][suit];
            n_allowed += allowed[r] ? 1 : 0;
        }
        if (n_allowed == 0)
            continue; // inconsistent state; leave zeros rather than NaN
        for (int r = 0; r < 3; ++r) {
            out[BELIEF_MASK_OFF + r * 32 + c] = allowed[r] ? 1.0f : 0.0f;
            out[BELIEF_PRIOR_OFF + r * 32 + c] = allowed[r] ? 1.0f / n_allowed
                                                            : 0.0f;
        }
    }

    // Tactical block. `seen` above is exactly the complement of unseen, so
    // the two stay consistent by construction.
    const int64_t unseen = ~seen & 0xFFFFFFFFLL;
    int unseen_suit[4] = {};
    int higher[32] = {};
    for (int c = 0; c < 32; ++c) {
        if (!(unseen & (int64_t(1) << c)))
            continue;
        ++unseen_suit[suitOf(c)];
        // Every own card of this suit that this unseen card outranks.
        for (int k = suitOf(c) * 8; k < suitOf(c) * 8 + 8; ++k)
            if (rankOf(c) > rankOf(k))
                ++higher[k];
    }
    for (int c = 0; c < 32; ++c) {
        if (!(own & (int64_t(1) << c)))
            continue;
        out[MASTER_OFF + c] = higher[c] == 0 ? 1.0f : 0.0f;
        out[HIGHER_UNSEEN_OFF + c] = higher[c] / 7.0f;
    }
    for (int suit = 0; suit < 4; ++suit)
        out[UNSEEN_SUIT_OFF + suit] = unseen_suit[suit] / 8.0f;
    for (int r = 0; r < 3; ++r) {
        const int seat = (observer + r + 1) % 4;
        for (int suit = 0; suit < 4; ++suit)
            out[SEAT_CAN_HOLD_OFF + r * 4 + suit] =
                (!voids[seat][suit] && unseen_suit[suit] > 0) ? 1.0f : 0.0f;
    }
}

} // namespace encoding

#include "manille_game.h"
#include "nn_policy.h"
#include "solver_wrapper.h"
#include "encoding.h"
#include "batched_ismcts.h"
#include "rule_playout.h"
#include "rule_agent.h"
#include "fast_policy.h"
#include "prior/trump_rule.h"
#include "sampler/trump_constrained.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// --- Exact endgame solver (manille_dmc_plan §5 endgame hybrid) ---
// Full-information minimax over the remaining cards of one belief-filtered
// world: both teams play optimally for their own points. Memoized on the
// plain public state (hands, trick, mover, pts0); legal play never depends
// on history, so the key is exact. Only sane at <= ~8 cards per hand.

namespace {

struct ExactKey {
    std::array<int64_t, 4> hands;
    std::array<int32_t, 4> trick_cards;
    std::array<int32_t, 4> trick_players;
    int n_trick;
    int current_player;
    int pts0;
    bool operator==(const ExactKey& o) const {
        return hands == o.hands && trick_cards == o.trick_cards
               && trick_players == o.trick_players && n_trick == o.n_trick
               && current_player == o.current_player && pts0 == o.pts0;
    }
};

struct ExactKeyHash {
    size_t operator()(const ExactKey& k) const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        for (int64_t v : k.hands) mix(static_cast<uint64_t>(v));
        for (int i = 0; i < 4; ++i) {
            mix(static_cast<uint32_t>(k.trick_cards[i]));
            mix(static_cast<uint32_t>(k.trick_players[i]));
        }
        mix(static_cast<uint64_t>(k.n_trick));
        mix(static_cast<uint64_t>(k.current_player));
        mix(static_cast<uint64_t>(k.pts0));
        return static_cast<size_t>(h);
    }
};

using ExactMemo = std::unordered_map<ExactKey, int, ExactKeyHash>;

int exact_minimax(const ManilleGame& game, ExactMemo& memo) {
    if (game.handMask(0) == 0 && game.handMask(1) == 0
        && game.handMask(2) == 0 && game.handMask(3) == 0)
        return game.teamPoints(0);

    ExactKey key{
        {game.handMask(0), game.handMask(1), game.handMask(2),
         game.handMask(3)},
        {game.trickCard(0), game.trickCard(1), game.trickCard(2),
         game.trickCard(3)},
        {game.trickPlayer(0), game.trickPlayer(1), game.trickPlayer(2),
         game.trickPlayer(3)},
        game.trickSize(),
        static_cast<int>(game.currentPlayer()),
        game.teamPoints(0),
    };
    auto it = memo.find(key);
    if (it != memo.end())
        return it->second;

    const bool team0_to_move = game.currentPlayer() % 2 == 0;
    int best = team0_to_move ? -1 : 61;
    for (const Card& move : game.validMoves()) {
        ManilleGame next = game;
        next.doMove(move);
        const int v = exact_minimax(next, memo);
        best = team0_to_move ? std::max(best, v) : std::min(best, v);
    }
    memo.emplace(key, best);
    return best;
}

} // namespace


static ManilleGame game_from_args(
    py::array_t<int64_t> hands,
    py::array_t<int32_t> trick_cards,
    py::array_t<int32_t> trick_players,
    py::array_t<int32_t> played_cards,
    py::array_t<int32_t> played_players,
    int n_played_cards,
    int n_trick,
    int current_player,
    int trump_suit,
    int pts0, int pts1,
    int tricks_won0, int tricks_won1,
    bool perfect_info,
    int declarer = -1)
{
    auto h = hands.unchecked<1>();
    auto tc = trick_cards.unchecked<1>();
    auto tp = trick_players.unchecked<1>();
    auto pc = played_cards.unchecked<1>();
    auto pp = played_players.unchecked<1>();

    std::array<int64_t, 4> h_arr{h(0), h(1), h(2), h(3)};
    std::array<int32_t, 4> tc_arr{tc(0), tc(1), tc(2), tc(3)};
    std::array<int32_t, 4> tp_arr{tp(0), tp(1), tp(2), tp(3)};
    std::array<int32_t, 32> pc_arr{};
    std::array<int32_t, 32> pp_arr{};
    for (int i = 0; i < 32; ++i) {
        pc_arr[i] = pc(i);
        pp_arr[i] = pp(i);
    }

    return ManilleGame(h_arr, tc_arr, tp_arr, n_trick,
                       current_player, trump_suit, pts0, pts1,
                       tricks_won0, tricks_won1,
                       pc_arr, pp_arr, n_played_cards, perfect_info,
                       declarer);
}

static py::dict result_to_dict(const SolverResult& result) {
    py::dict d;
    d["best_move"] = result.best_move;
    d["visit_counts"] = result.visit_counts;
    d["total_visits"] = result.total_visits;
    d["root_value"] = result.root_value;
    return d;
}

static void apply_particle_pool(
    ManilleGame& game,
    py::object particles_obj,
    py::object particle_weights_obj
) {
    if (particles_obj.is_none()) return;

    auto particles = particles_obj.cast<py::array_t<int64_t>>();
    auto buf = particles.unchecked<2>();
    std::vector<std::array<int64_t, 4>> pool;
    pool.reserve(buf.shape(0));
    for (ssize_t i = 0; i < buf.shape(0); ++i)
        pool.push_back({buf(i, 0), buf(i, 1), buf(i, 2), buf(i, 3)});

    std::vector<double> weights;
    if (!particle_weights_obj.is_none()) {
        auto particle_weights = particle_weights_obj.cast<py::array_t<double>>();
        auto wbuf = particle_weights.unchecked<1>();
        if (wbuf.shape(0) != buf.shape(0)) {
            throw std::invalid_argument(
                "particle_weights must have the same length as particles."
            );
        }

        weights.reserve(wbuf.shape(0));
        double total_weight = 0.0;
        for (ssize_t i = 0; i < wbuf.shape(0); ++i) {
            const double weight = wbuf(i);
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument(
                    "particle_weights must be finite and non-negative."
                );
            }
            weights.push_back(weight);
            total_weight += weight;
        }
        if (total_weight <= 0.0) {
            throw std::invalid_argument("particle_weights must sum to a positive value.");
        }
    }

    game.setParticlePool(std::move(pool), std::move(weights));
}

PYBIND11_MODULE(manille_ismcts, m) {
    m.doc() = "ISMCTS solver for Manille card game";

    // Direct engine access (match plan M0): lets Python drive a full deal
    // on the C++ engine — trump selection (constructor; trump_suit=-1 is
    // mull and sets the x2 multiplier), the announcement window
    // (announce_double / announce_redouble / start_playing), card play, and
    // terminal scoring (hand_score / get_result). The Python engine in
    // manille_env/game_logic.py is the reference implementation; the two
    // must agree exactly (tests/test_stakes_parity.py).
    py::class_<ManilleGame>(m, "ManilleGame")
        .def(py::init([](py::array_t<int64_t> hands, int trump_suit,
                         int declarer) {
                 auto h = hands.unchecked<1>();
                 std::array<int64_t, 4> h_arr{h(0), h(1), h(2), h(3)};
                 return ManilleGame(h_arr,
                                    std::array<int32_t, 4>{},
                                    std::array<int32_t, 4>{},
                                    0,
                                    (declarer + 1) % 4,
                                    trump_suit,
                                    0, 0, 0, 0,
                                    std::array<int32_t, 32>{},
                                    std::array<int32_t, 32>{},
                                    0,
                                    false,
                                    declarer);
             }),
             py::arg("hands"), py::arg("trump_suit"), py::arg("declarer"),
             "Fresh deal: hands are 4 card bitmasks, trump_suit is 0-3 or "
             "-1 for mull (which sets the x2 multiplier), declarer is the "
             "dealer. The player left of the dealer moves first.")
        .def("valid_moves", [](const ManilleGame& self) {
            std::vector<int32_t> out;
            for (const Card& move : self.validMoves())
                out.push_back(move.index());
            return out;
        })
        .def("do_move", [](ManilleGame& self, int card) {
            self.doMove(Card::from_index(card));
        }, py::arg("card"),
            "Play a card (must be in valid_moves). Closes the announcement "
            "window on the first card played.")
        .def("current_player", [](const ManilleGame& self) {
            return static_cast<int>(self.currentPlayer());
        })
        .def("team_points", &ManilleGame::teamPoints)
        .def("tricks_won", &ManilleGame::tricksWon)
        .def("hand_mask", &ManilleGame::handMask)
        .def("trump_suit", &ManilleGame::trumpSuit)
        .def("declarer", &ManilleGame::declarer)
        .def("finished", &ManilleGame::finished)
        .def("multiplier", &ManilleGame::multiplier)
        .def("doubled", &ManilleGame::doubled)
        .def("redoubled", &ManilleGame::redoubled)
        .def("doubling_phase", &ManilleGame::doublingPhase)
        .def("announce_double", &ManilleGame::announceDouble, py::arg("player"),
             "\"Ik ga mee\": defending team doubles. Only during the "
             "announcement window, only the non-trump team, only once.")
        .def("announce_redouble", &ManilleGame::announceRedouble,
             py::arg("player"),
             "\"Ik ga tegen\": trump team answers a meegaan. Flag-gated, "
             "never multiplier-gated (mull alone already makes the "
             "multiplier 2).")
        .def("start_playing", &ManilleGame::startPlaying,
             "Close the announcement window without playing a card "
             "(Python's start_playing).")
        .def("hand_score", &ManilleGame::handScore, py::arg("team"),
             "Terminal deal score in match points: (winner_points - 30) * "
             "multiplier for the winning team, 0 for the loser and for a "
             "30-30 tie.")
        .def("get_result", &ManilleGame::getResult, py::arg("player"),
             "hand_score of the player's team / 30 (the MCTS utility).");

    py::class_<ManilleSolverSO>(m, "ManilleSolverSO")
        .def(py::init<unsigned, bool>(),
             py::arg("iterations") = 1000,
             py::arg("perfect_info") = false)
        .def("solve", [](ManilleSolverSO& self,
                         py::array_t<int64_t> hands,
                         py::array_t<int32_t> trick_cards,
                         py::array_t<int32_t> trick_players,
                         py::array_t<int32_t> played_cards,
                         py::array_t<int32_t> played_players,
                         int n_played_cards,
                         int n_trick,
                         int current_player,
                         int trump_suit,
                         int pts0, int pts1,
                         int tricks_won0, int tricks_won1,
                         py::object particles,
                         py::object particle_weights,
                         bool partner_transparent) {
            auto game = game_from_args(hands, trick_cards, trick_players,
                                       played_cards, played_players, n_played_cards,
                                       n_trick, current_player, trump_suit,
                                       pts0, pts1, tricks_won0, tricks_won1,
                                       self.perfectInfo());
            if (partner_transparent)
                game.setPartnerTransparent(true);
            apply_particle_pool(game, particles, particle_weights);
            auto result = self.solve(game);
            return result_to_dict(result);
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"),
        py::arg("particles") = py::none(),
        py::arg("particle_weights") = py::none(),
        py::arg("partner_transparent") = false);

    py::class_<ManilleSolverMO>(m, "ManilleSolverMO")
        .def(py::init<unsigned, bool>(),
             py::arg("iterations") = 1000,
             py::arg("perfect_info") = false)
        .def("solve", [](ManilleSolverMO& self,
                         py::array_t<int64_t> hands,
                         py::array_t<int32_t> trick_cards,
                         py::array_t<int32_t> trick_players,
                         py::array_t<int32_t> played_cards,
                         py::array_t<int32_t> played_players,
                         int n_played_cards,
                         int n_trick,
                         int current_player,
                         int trump_suit,
                         int pts0, int pts1,
                         int tricks_won0, int tricks_won1,
                         py::object particles,
                         py::object particle_weights) {
            auto game = game_from_args(hands, trick_cards, trick_players,
                                       played_cards, played_players, n_played_cards,
                                       n_trick, current_player, trump_suit,
                                       pts0, pts1, tricks_won0, tricks_won1,
                                       self.perfectInfo());
            apply_particle_pool(game, particles, particle_weights);
            auto result = self.solve(game);
            return result_to_dict(result);
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"),
        py::arg("particles") = py::none(),
        py::arg("particle_weights") = py::none());

    py::class_<ManilleSolverTime>(m, "ManilleSolverTime")
        .def(py::init<int, int, bool>(),
             py::arg("threads") = 15,
             py::arg("budget_ms") = 500,
             py::arg("perfect_info") = false)
        .def("solve", [](ManilleSolverTime& self,
                         py::array_t<int64_t> hands,
                         py::array_t<int32_t> trick_cards,
                         py::array_t<int32_t> trick_players,
                         py::array_t<int32_t> played_cards,
                         py::array_t<int32_t> played_players,
                         int n_played_cards,
                         int n_trick,
                         int current_player,
                         int trump_suit,
                         int pts0, int pts1,
                         int tricks_won0, int tricks_won1) {
            auto game = game_from_args(hands, trick_cards, trick_players,
                                       played_cards, played_players, n_played_cards,
                                       n_trick, current_player, trump_suit,
                                       pts0, pts1, tricks_won0, tricks_won1,
                                       self.perfectInfo());
            auto result = self.solve(game);
            return result_to_dict(result);
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"));

    m.def("sample_determinization",
          [](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players,
             int n_played_cards,
             int n_trick,
             int current_player,
             int trump_suit,
             int pts0, int pts1,
             int tricks_won0, int tricks_won1,
             int observer,
             bool perfect_info,
             py::object particles,
             py::object particle_weights,
             int declarer) {
              auto game = game_from_args(
                  hands, trick_cards, trick_players,
                  played_cards, played_players, n_played_cards,
                  n_trick, current_player, trump_suit,
                  pts0, pts1, tricks_won0, tricks_won1, perfect_info,
                  declarer);
              apply_particle_pool(game, particles, particle_weights);
              auto clone = game.cloneAndRandomise(static_cast<unsigned int>(observer));
              auto* det = dynamic_cast<ManilleGame*>(clone.get());
              if (det == nullptr)
                  throw std::runtime_error("Determinization did not return ManilleGame.");

              py::array_t<int64_t> out(4);
              auto out_mut = out.mutable_unchecked<1>();
              for (int p = 0; p < 4; ++p)
                  out_mut(p) = det->handMask(p);
              return out;
          },
          py::arg("hands"),
          py::arg("trick_cards"),
          py::arg("trick_players"),
          py::arg("played_cards"),
          py::arg("played_players"),
          py::arg("n_played_cards"),
          py::arg("n_trick"),
          py::arg("current_player"),
          py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"),
          py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("observer"),
          py::arg("perfect_info") = false,
          py::arg("particles") = py::none(),
          py::arg("particle_weights") = py::none(),
          // declarer (== dealer in the learning stack) activates the
          // TrumpConstrainedDealSampler: without it the sampler silently
          // falls back to plain uniform and the trump-chooser signal is lost.
          py::arg("declarer") = -1);

    m.def("sample_determinizations",
          [](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players,
             int n_played_cards,
             int n_trick,
             int current_player,
             int trump_suit,
             int pts0, int pts1,
             int tricks_won0, int tricks_won1,
             int observer,
             int n,
             int declarer,
             bool allow_short) {
              // Batched version of sample_determinization: builds the game
              // once and draws n worlds from the hard-constraint sampler.
              // Used by belief-weighted determinization (many candidates per
              // move) — the per-call pybind path rebuilt the game every time.
              // declarer (== dealer) must be threaded through or the
              // TrumpConstrainedDealSampler no-ops to plain uniform.
              auto game = game_from_args(
                  hands, trick_cards, trick_players,
                  played_cards, played_players, n_played_cards,
                  n_trick, current_player, trump_suit,
                  pts0, pts1, tricks_won0, tricks_won1, false, declarer);
              // A single draw can exhaust the rejection budget in the rare
              // states where the trump rule pins the declarer's hand tightly.
              // Those states are legal and their world set is non-empty (the
              // real deal is in it), so all-or-nothing turns a few-percent
              // per-draw miss into a near-certain failure of the whole call
              // at n=32. With allow_short, retry misses within a bounded
              // budget and return the worlds that did land: a draw accepted
              // by rejection sampling is exactly on-distribution however many
              // attempts it cost, so a short pool costs variance, not bias.
              static const sampler::TrumpConstrainedDealSampler s_sampler;
              const auto public_state = sampler::PublicState::fromGame(game, observer);
              const auto deals = s_sampler.sampleDeals(public_state, n, allow_short);
              if (deals.empty())
                  throw sampler::RejectionBudgetExhausted(
                      "TrumpConstrainedDealSampler: exhausted rejection "
                      "budget without a trump-consistent deal.");
              py::array_t<int64_t> out(
                  std::vector<ssize_t>{static_cast<ssize_t>(deals.size()), 4});
              auto out_mut = out.mutable_unchecked<2>();
              for (ssize_t k = 0; k < static_cast<ssize_t>(deals.size()); ++k)
                  for (int p = 0; p < 4; ++p)
                      out_mut(k, p) = deals[static_cast<size_t>(k)].hands[p];
              return out;
          },
          py::arg("hands"),
          py::arg("trick_cards"),
          py::arg("trick_players"),
          py::arg("played_cards"),
          py::arg("played_players"),
          py::arg("n_played_cards"),
          py::arg("n_trick"),
          py::arg("current_player"),
          py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"),
          py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("observer"),
          py::arg("n"),
          py::arg("declarer") = -1,
          // Opt-in: return fewer than n worlds rather than failing the whole
          // call when individual draws exhaust the rejection budget. Callers
          // that size arrays off n instead of len(result) must leave it off.
          py::arg("allow_short") = false);

    // --- Learning-stack bindings ---

    m.attr("OBS_DIM") = encoding::OBS_DIM;
    m.attr("ACT_DIM") = encoding::ACT_DIM;

    m.def("apply_trump_rule", &prior::apply_trump_rule,
          py::arg("hand_mask"),
          "The rule-based trump selector. Used for all learning-stack data "
          "generation and evaluation so the trump distribution is consistent.");

    m.def("encode_observation",
          [](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players,
             int n_played_cards,
             int n_trick,
             int current_player,
             int trump_suit,
             int pts0, int pts1,
             int tricks_won0, int tricks_won1,
             int declarer,
             int observer) {
              auto game = game_from_args(
                  hands, trick_cards, trick_players,
                  played_cards, played_players, n_played_cards,
                  n_trick, current_player, trump_suit,
                  pts0, pts1, tricks_won0, tricks_won1,
                  false, declarer);
              py::array_t<float> out(encoding::OBS_DIM);
              encoding::encode(game, observer, out.mutable_data());
              return out;
          },
          py::arg("hands"),
          py::arg("trick_cards"),
          py::arg("trick_players"),
          py::arg("played_cards"),
          py::arg("played_players"),
          py::arg("n_played_cards"),
          py::arg("n_trick"),
          py::arg("current_player"),
          py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"),
          py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("declarer"),
          py::arg("observer"));

    m.def("encode_trajectory",
          [](py::array_t<int64_t> deal_hands,
             int trump_suit,
             int declarer,
             py::array_t<int32_t> actions) {
              auto h = deal_hands.unchecked<1>();
              auto a = actions.unchecked<1>();
              if (a.shape(0) != 32)
                  throw std::invalid_argument("actions must have length 32.");

              std::array<int64_t, 4> hands{h(0), h(1), h(2), h(3)};
              py::array_t<float> obs(
                  std::vector<ssize_t>{32, encoding::OBS_DIM});
              py::array_t<int32_t> players(32);
              int pts0 = 0, pts1 = 0;
              float* obs_ptr = obs.mutable_data();
              int32_t* players_ptr = players.mutable_data();

              {
                  py::gil_scoped_release release;
                  ManilleGame game(
                      hands,
                      std::array<int32_t, 4>{},
                      std::array<int32_t, 4>{},
                      0,
                      (declarer + 1) % 4,
                      trump_suit,
                      0, 0, 0, 0,
                      std::array<int32_t, 32>{},
                      std::array<int32_t, 32>{},
                      0,
                      false,
                      declarer);

                  for (int t = 0; t < 32; ++t) {
                      const int card = a(t);
                      bool legal = false;
                      for (const Card& move : game.validMoves())
                          if (move.index() == card) {
                              legal = true;
                              break;
                          }
                      if (!legal)
                          throw std::invalid_argument(
                              "Trajectory action " + std::to_string(t)
                              + " (card " + std::to_string(card)
                              + ") is not legal in the replayed state.");

                      const int observer = static_cast<int>(game.currentPlayer());
                      players_ptr[t] = observer;
                      encoding::encode(
                          game, observer,
                          obs_ptr + t * encoding::OBS_DIM);
                      game.doMove(Card::from_index(card));
                  }
                  pts0 = game.teamPoints(0);
                  pts1 = game.teamPoints(1);
              }

              py::dict out;
              out["obs"] = obs;
              out["players"] = players;
              out["pts0"] = pts0;
              out["pts1"] = pts1;
              return out;
          },
          py::arg("deal_hands"),
          py::arg("trump_suit"),
          py::arg("declarer"),
          py::arg("actions"),
          "Replay a full deal and return the per-ply observations "
          "(observer = player to move) plus per-ply current player and the "
          "final team points. Validates every action against the engine.");

    m.def("encode_q_observation",
          [](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players,
             int n_played_cards,
             int n_trick,
             int current_player,
             int trump_suit,
             int pts0, int pts1,
             int tricks_won0, int tricks_won1,
             int declarer,
             int observer) {
              auto game = game_from_args(
                  hands, trick_cards, trick_players,
                  played_cards, played_players, n_played_cards,
                  n_trick, current_player, trump_suit,
                  pts0, pts1, tricks_won0, tricks_won1,
                  false, declarer);
              py::array_t<float> obs(encoding::OBS_DIM);
              encoding::encode(game, observer, obs.mutable_data());

              py::array_t<float> act(
                  std::vector<ssize_t>{encoding::MAX_LEGAL, encoding::ACT_DIM});
              py::array_t<int32_t> legal(encoding::MAX_LEGAL);
              float* act_ptr = act.mutable_data();
              std::memset(act_ptr, 0,
                          encoding::MAX_LEGAL * encoding::ACT_DIM
                              * sizeof(float));
              int32_t* legal_ptr = legal.mutable_data();
              std::fill(legal_ptr, legal_ptr + encoding::MAX_LEGAL, -1);

              int n_legal = 0;
              if (observer == static_cast<int>(game.currentPlayer())) {
                  for (const Card& move : game.validMoves()) {
                      if (n_legal >= encoding::MAX_LEGAL)
                          break;  // unreachable: a hand holds at most 8 cards
                      encoding::encode_action(
                          game, move.index(),
                          act_ptr + n_legal * encoding::ACT_DIM);
                      legal_ptr[n_legal] = move.index();
                      ++n_legal;
                  }
              }

              py::dict out;
              out["obs"] = obs;
              out["act"] = act;
              out["legal"] = legal;
              out["n_legal"] = n_legal;
              return out;
          },
          py::arg("hands"),
          py::arg("trick_cards"),
          py::arg("trick_players"),
          py::arg("played_cards"),
          py::arg("played_players"),
          py::arg("n_played_cards"),
          py::arg("n_trick"),
          py::arg("current_player"),
          py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"),
          py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("declarer"),
          py::arg("observer"),
          "Observation plus per-legal-candidate action features for DMC "
          "Q(s,a). 'act' row j is the candidate feature vector of card "
          "'legal'[j]; rows past n_legal are zero / -1 padded.");

    m.def("encode_q_trajectory",
          [](py::array_t<int64_t> deal_hands,
             int trump_suit,
             int declarer,
             py::array_t<int32_t> actions) {
              auto h = deal_hands.unchecked<1>();
              auto a = actions.unchecked<1>();
              if (a.shape(0) != 32)
                  throw std::invalid_argument("actions must have length 32.");

              std::array<int64_t, 4> hands{h(0), h(1), h(2), h(3)};
              py::array_t<float> obs(
                  std::vector<ssize_t>{32, encoding::OBS_DIM});
              py::array_t<float> act(std::vector<ssize_t>{
                  32, encoding::MAX_LEGAL, encoding::ACT_DIM});
              py::array_t<int32_t> legal(
                  std::vector<ssize_t>{32, encoding::MAX_LEGAL});
              py::array_t<int32_t> n_legal(32);
              py::array_t<int32_t> players(32);
              int pts0 = 0, pts1 = 0;
              float* obs_ptr = obs.mutable_data();
              float* act_ptr = act.mutable_data();
              int32_t* legal_ptr = legal.mutable_data();
              int32_t* n_legal_ptr = n_legal.mutable_data();
              int32_t* players_ptr = players.mutable_data();

              {
                  py::gil_scoped_release release;
                  ManilleGame game(
                      hands,
                      std::array<int32_t, 4>{},
                      std::array<int32_t, 4>{},
                      0,
                      (declarer + 1) % 4,
                      trump_suit,
                      0, 0, 0, 0,
                      std::array<int32_t, 32>{},
                      std::array<int32_t, 32>{},
                      0,
                      false,
                      declarer);

                  for (int t = 0; t < 32; ++t) {
                      const int card = a(t);
                      const int observer =
                          static_cast<int>(game.currentPlayer());
                      players_ptr[t] = observer;

                      encoding::encode(
                          game, observer,
                          obs_ptr + t * encoding::OBS_DIM);

                      float* act_t = act_ptr
                          + t * encoding::MAX_LEGAL * encoding::ACT_DIM;
                      std::memset(act_t, 0,
                                  encoding::MAX_LEGAL * encoding::ACT_DIM
                                      * sizeof(float));
                      int32_t* legal_t = legal_ptr + t * encoding::MAX_LEGAL;
                      std::fill(legal_t, legal_t + encoding::MAX_LEGAL, -1);

                      int n = 0;
                      bool legal_found = false;
                      for (const Card& move : game.validMoves()) {
                          if (move.index() == card)
                              legal_found = true;
                          encoding::encode_action(
                              game, move.index(),
                              act_t + n * encoding::ACT_DIM);
                          legal_t[n] = move.index();
                          ++n;
                      }
                      n_legal_ptr[t] = n;
                      if (!legal_found)
                          throw std::invalid_argument(
                              "Trajectory action " + std::to_string(t)
                              + " (card " + std::to_string(card)
                              + ") is not legal in the replayed state.");

                      game.doMove(Card::from_index(card));
                  }
                  pts0 = game.teamPoints(0);
                  pts1 = game.teamPoints(1);
              }

              py::dict out;
              out["obs"] = obs;
              out["act"] = act;
              out["legal"] = legal;
              out["n_legal"] = n_legal;
              out["players"] = players;
              out["pts0"] = pts0;
              out["pts1"] = pts1;
              return out;
          },
          py::arg("deal_hands"),
          py::arg("trump_suit"),
          py::arg("declarer"),
          py::arg("actions"),
          "encode_trajectory plus per-ply candidate action features "
          "(DMC training-side re-encoding of self-play shards).");

    m.def("exact_solve",
          [](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players,
             int n_played_cards,
             int n_trick,
             int current_player,
             int trump_suit,
             int pts0, int pts1,
             int tricks_won0, int tricks_won1,
             int declarer,
             py::array_t<int64_t> world) {
              // Build the mid-deal state from the public arrays but with
              // the world's full-information holdings substituted in.
              auto tc = trick_cards.unchecked<1>();
              auto tp = trick_players.unchecked<1>();
              auto pc = played_cards.unchecked<1>();
              auto pp = played_players.unchecked<1>();
              auto w = world.unchecked<1>();
              std::array<int64_t, 4> wh{w(0), w(1), w(2), w(3)};
              std::array<int32_t, 4> tc_arr{tc(0), tc(1), tc(2), tc(3)};
              std::array<int32_t, 4> tp_arr{tp(0), tp(1), tp(2), tp(3)};
              std::array<int32_t, 32> pc_arr{};
              std::array<int32_t, 32> pp_arr{};
              for (int i = 0; i < 32; ++i) {
                  pc_arr[i] = pc(i);
                  pp_arr[i] = pp(i);
              }
              ManilleGame game(wh, tc_arr, tp_arr, n_trick, current_player,
                               trump_suit, pts0, pts1, tricks_won0,
                               tricks_won1, pc_arr, pp_arr, n_played_cards,
                               true, declarer);

              std::vector<int32_t> moves;
              std::vector<double> v0;
              ExactMemo memo;
              for (const Card& move : game.validMoves()) {
                  ManilleGame next = game;
                  next.doMove(move);
                  moves.push_back(move.index());
                  const int p0 = exact_minimax(next, memo);
                  v0.push_back(static_cast<double>(p0));
              }
              py::dict out;
              out["moves"] = moves;
              out["pts0"] = v0;
              out["memo_size"] = memo.size();
              return out;
          },
          py::arg("hands"),
          py::arg("trick_cards"),
          py::arg("trick_players"),
          py::arg("played_cards"),
          py::arg("played_players"),
          py::arg("n_played_cards"),
          py::arg("n_trick"),
          py::arg("current_player"),
          py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"),
          py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("declarer"),
          py::arg("world"),
          "Full-information exact minimax for one belief-filtered world: "
          "per legal move, the optimal-play final team-0 points (team 1 "
          "gets 60 minus that). Used by the endgame hybrid agent.");

    py::class_<mcts::BatchedISMCTS>(m, "BatchedISMCTS")
        .def(py::init([](unsigned sims, unsigned wave, const std::string& mode,
                         bool puct, double exploration, double c_puct,
                         double dirichlet_epsilon, uint64_t seed,
                         bool partner_belief) {
                 const mcts::LeafMode leaf_mode = mode == "value"
                     ? mcts::LeafMode::Value
                     : mcts::LeafMode::Rollout;
                 return mcts::BatchedISMCTS(sims, wave, leaf_mode, puct,
                                            exploration, c_puct,
                                            dirichlet_epsilon, seed,
                                            partner_belief);
             }),
             py::arg("sims"), py::arg("wave"), py::arg("mode") = "rollout",
             py::arg("puct") = false, py::arg("exploration") = 0.7,
             py::arg("c_puct") = 2.0, py::arg("dirichlet_epsilon") = 0.0,
             py::arg("seed") = 0, py::arg("partner_belief") = false)
        .def("solve", [](mcts::BatchedISMCTS& self,
                         py::array_t<int64_t> hands,
                         py::array_t<int32_t> trick_cards,
                         py::array_t<int32_t> trick_players,
                         py::array_t<int32_t> played_cards,
                         py::array_t<int32_t> played_players,
                         int n_played_cards,
                         int n_trick,
                         int current_player,
                         int trump_suit,
                         int pts0, int pts1,
                         int tricks_won0, int tricks_won1,
                         int declarer,
                         py::object particles,
                         py::object particle_weights,
                         py::function policy_value_fn) {
            auto game = game_from_args(hands, trick_cards, trick_players,
                                       played_cards, played_players, n_played_cards,
                                       n_trick, current_player, trump_suit,
                                       pts0, pts1, tricks_won0, tricks_won1,
                                       false, declarer);
            apply_particle_pool(game, particles, particle_weights);
            mcts::PolicyValueFn fn =
                [&policy_value_fn](const float* obs, int n, float* probs,
                                   float* values) {
                    py::array_t<float> obs_arr(
                        std::vector<ssize_t>{n, encoding::OBS_DIM});
                    std::memcpy(obs_arr.mutable_data(), obs,
                                static_cast<size_t>(n) * encoding::OBS_DIM
                                    * sizeof(float));
                    py::tuple out = policy_value_fn(obs_arr);
                    auto p = out[0].cast<py::array_t<float>>();
                    auto v = out[1].cast<py::array_t<float>>();
                    std::memcpy(probs, p.data(),
                                static_cast<size_t>(n) * 32 * sizeof(float));
                    std::memcpy(values, v.data(),
                                static_cast<size_t>(n) * sizeof(float));
                };
            auto result = self.solve(game, fn);
            return result_to_dict(result);
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"),
        py::arg("declarer"),
        py::arg("particles") = py::none(),
        py::arg("particle_weights") = py::none(),
        py::arg("policy_value_fn") = py::none())
        // Resumable API for cross-game batching (self-play generation).
        .def("begin", [](mcts::BatchedISMCTS& self,
                         py::array_t<int64_t> hands,
                         py::array_t<int32_t> trick_cards,
                         py::array_t<int32_t> trick_players,
                         py::array_t<int32_t> played_cards,
                         py::array_t<int32_t> played_players,
                         int n_played_cards,
                         int n_trick,
                         int current_player,
                         int trump_suit,
                         int pts0, int pts1,
                         int tricks_won0, int tricks_won1,
                         int declarer,
                         py::object particles,
                         py::object particle_weights) {
            auto game = game_from_args(
                hands, trick_cards, trick_players,
                played_cards, played_players, n_played_cards,
                n_trick, current_player, trump_suit,
                pts0, pts1, tricks_won0, tricks_won1, false, declarer);
            apply_particle_pool(game, particles, particle_weights);
            self.begin(std::move(game));
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"),
        py::arg("declarer"),
        py::arg("particles") = py::none(),
        py::arg("particle_weights") = py::none())
        .def("done", &mcts::BatchedISMCTS::done)
        .def("pending_count", &mcts::BatchedISMCTS::pendingCount)
        .def("pending_obs", [](mcts::BatchedISMCTS& self) {
            const int n = self.pendingCount();
            py::array_t<float> obs(std::vector<ssize_t>{n, encoding::OBS_DIM});
            if (n > 0)
                self.pendingObs(obs.mutable_data());
            return obs;
        })
        .def("provide", [](mcts::BatchedISMCTS& self,
                           py::array_t<float> probs,
                           py::array_t<float> values) {
            auto p = probs.unchecked<1>();
            auto v = values.unchecked<1>();
            if (p.shape(0) != self.pendingCount() * 32
                || v.shape(0) != self.pendingCount())
                throw std::invalid_argument(
                    "provide: probs/values size does not match pending_count.");
            self.provide(probs.data(), values.data());
        })
        .def("result", [](mcts::BatchedISMCTS& self) {
            return result_to_dict(self.result());
        })
        // C1: same one-shot solve but with in-process libtorch inference —
        // no Python callback per wave, GIL released for the whole search.
        .def("solve_nn", [](mcts::BatchedISMCTS& self,
                            py::array_t<int64_t> hands,
                            py::array_t<int32_t> trick_cards,
                            py::array_t<int32_t> trick_players,
                            py::array_t<int32_t> played_cards,
                            py::array_t<int32_t> played_players,
                            int n_played_cards,
                            int n_trick,
                            int current_player,
                            int trump_suit,
                            int pts0, int pts1,
                            int tricks_won0, int tricks_won1,
                            int declarer,
                            py::object particles,
                            py::object particle_weights,
                            NNPolicyValue& nn) {
            auto game = game_from_args(hands, trick_cards, trick_players,
                                       played_cards, played_players, n_played_cards,
                                       n_trick, current_player, trump_suit,
                                       pts0, pts1, tricks_won0, tricks_won1,
                                       false, declarer);
            apply_particle_pool(game, particles, particle_weights);
            SolverResult result;
            {
                py::gil_scoped_release release;
                result = self.solve(game, std::ref(nn));
            }
            return result_to_dict(result);
        },
        py::arg("hands"),
        py::arg("trick_cards"),
        py::arg("trick_players"),
        py::arg("played_cards"),
        py::arg("played_players"),
        py::arg("n_played_cards"),
        py::arg("n_trick"),
        py::arg("current_player"),
        py::arg("trump_suit"),
        py::arg("pts0"), py::arg("pts1"),
        py::arg("tricks_won0"), py::arg("tricks_won1"),
        py::arg("declarer"),
        py::arg("particles") = py::none(),
        py::arg("particle_weights") = py::none(),
        py::arg("nn"));

    py::class_<NNPolicyValue>(m, "NNPolicyValue")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("path"), py::arg("device") = "cuda")
        .def("eval", [](NNPolicyValue& self, py::array_t<float> obs) {
            auto buf = obs.unchecked<2>();
            const auto n = static_cast<int>(buf.shape(0));
            py::array_t<float> probs(std::vector<ssize_t>{n, 32});
            py::array_t<float> values(n);
            self(buf.data(0, 0), n, probs.mutable_data(),
                 values.mutable_data());
            return py::make_tuple(probs, values);
        });

    // Rule-agent play-out (cpp/rule_playout.h). C++ port of
    // learn/rule_playout.py; tests/test_rule_playout_parity.py pins them.
    auto rollout_inputs = [](py::array_t<int64_t> worlds,
                             py::array_t<int32_t> hist_seats,
                             py::array_t<int32_t> hist_cards) {
        auto w = worlds.unchecked<2>();
        if (w.shape(1) != 4)
            throw std::invalid_argument("worlds must be (n, 4)");
        std::vector<rollout::World> ws(static_cast<size_t>(w.shape(0)));
        for (py::ssize_t i = 0; i < w.shape(0); ++i)
            for (int s = 0; s < 4; ++s)
                ws[i][s] = w(i, s);
        auto hs = hist_seats.unchecked<1>();
        auto hc = hist_cards.unchecked<1>();
        if (hs.shape(0) != hc.shape(0))
            throw std::invalid_argument("history seats/cards length mismatch");
        rollout::History h;
        for (py::ssize_t i = 0; i < hs.shape(0); ++i)
            h.emplace_back(hs(i), hc(i));
        return std::make_pair(ws, h);
    };
    m.def("rule_playout_evaluate",
          [rollout_inputs](int trump, int declarer, int me, std::vector<int> candidates,
             py::array_t<int64_t> worlds, py::array_t<int32_t> hist_seats,
             py::array_t<int32_t> hist_cards, int honest, bool agent_like, int nt_rules,
             int tr_rules, py::object net, int net_plies, bool net_trick_end) {
              auto in = rollout_inputs(worlds, hist_seats, hist_cards);
              rollout::Options opt;
              opt.honest = honest;
              opt.agent_like = agent_like;
              opt.nt_rules = nt_rules;
              opt.tr_rules = tr_rules;
              const fastpolicy::Net* np = net.is_none() ? nullptr
                                                        : net.cast<const fastpolicy::Net*>();
              rollout::NetChoose nc_fn = [np](const ManilleGame& g, int seat) {
                  return fastpolicy::choose(*np, g, seat);
              };
              std::vector<std::vector<double>> r;
              {
                  py::gil_scoped_release release;
                  r = rollout::evaluate_candidates(trump, declarer, me, candidates,
                                                   in.first, in.second, opt,
                                                   np ? &nc_fn : nullptr, net_plies, net_trick_end);
              }
              const py::ssize_t nc = static_cast<py::ssize_t>(candidates.size());
              const py::ssize_t nw = static_cast<py::ssize_t>(in.first.size());
              py::array_t<double> out({nc, nw});
              auto o = out.mutable_unchecked<2>();
              for (py::ssize_t i = 0; i < nc; ++i)
                  for (py::ssize_t j = 0; j < nw; ++j)
                      o(i, j) = r[i][j];
              return out;
          },
          py::arg("trump"), py::arg("declarer"), py::arg("me"), py::arg("candidates"),
          py::arg("worlds"), py::arg("hist_seats"), py::arg("hist_cards"),
          py::arg("honest") = 0, py::arg("agent_like") = false,
          py::arg("nt_rules") = 0, py::arg("tr_rules") = 0,
          py::arg("net") = py::none(), py::arg("net_plies") = 0,
          py::arg("net_trick_end") = false,
          "Rule-policy play-out: my team's final points, [candidate][world]. "
          "trump -1 = no trump; worlds are current hands per absolute seat.");
    m.def("rule_playout_weights",
          [rollout_inputs](int trump, int declarer, int me, py::array_t<int64_t> worlds,
             py::array_t<int32_t> hist_seats, py::array_t<int32_t> hist_cards,
             double eps, int honest, bool agent_like, int nt_rules, int tr_rules) {
              auto in = rollout_inputs(worlds, hist_seats, hist_cards);
              rollout::Options opt;
              opt.honest = honest;
              opt.agent_like = agent_like;
              opt.nt_rules = nt_rules;
              opt.tr_rules = tr_rules;
              std::vector<double> r;
              {
                  py::gil_scoped_release release;
                  r = rollout::world_weights(trump, declarer, me, in.first, in.second,
                                             eps, opt);
              }
              return py::array_t<double>(static_cast<py::ssize_t>(r.size()), r.data());
          },
          py::arg("trump"), py::arg("declarer"), py::arg("me"), py::arg("worlds"),
          py::arg("hist_seats"), py::arg("hist_cards"), py::arg("eps"),
          py::arg("honest") = 0, py::arg("agent_like") = false,
          py::arg("nt_rules") = 0, py::arg("tr_rules") = 0,
          "Play-likelihood weight per world (eps per unexplained play).");
    m.def("rule_playout_mismatches",
          [rollout_inputs](int trump, int declarer, int me, py::array_t<int64_t> worlds,
             py::array_t<int32_t> hist_seats, py::array_t<int32_t> hist_cards,
             int honest, bool agent_like, int nt_rules, int tr_rules) {
              auto in = rollout_inputs(worlds, hist_seats, hist_cards);
              rollout::Options opt;
              opt.honest = honest;
              opt.agent_like = agent_like;
              opt.nt_rules = nt_rules;
              opt.tr_rules = tr_rules;
              std::vector<std::array<int, 4>> r;
              {
                  py::gil_scoped_release release;
                  r = rollout::world_mismatches(trump, declarer, me, in.first, in.second, opt);
              }
              py::array_t<int32_t> a({static_cast<py::ssize_t>(r.size()), py::ssize_t(4)});
              auto v = a.mutable_unchecked<2>();
              for (size_t i = 0; i < r.size(); ++i)
                  for (int s = 0; s < 4; ++s) v(i, s) = r[i][s];
              return a;
          },
          py::arg("trump"), py::arg("declarer"), py::arg("me"), py::arg("worlds"),
          py::arg("hist_seats"), py::arg("hist_cards"),
          py::arg("honest") = 0, py::arg("agent_like") = false,
          py::arg("nt_rules") = 0, py::arg("tr_rules") = 0,
          "Unexplained plays per world and absolute seat, (n_worlds, 4) int32.");

    // Fast learned play-out policy (cpp/fast_policy.h).
    m.attr("FAST_POLICY_DIM") = fastpolicy::F_DIM;
    m.attr("FAST_POLICY_BINARY") = fastpolicy::F_BINARY;
    m.def("fast_policy_features",
          [](py::array_t<int64_t> deal_hands, int trump, int declarer,
             py::array_t<int32_t> cards) {
              auto h = deal_hands.unchecked<1>();
              auto a = cards.unchecked<1>();
              std::array<int64_t, 4> hands{h(0), h(1), h(2), h(3)};
              ManilleGame g(hands, std::array<int32_t, 4>{}, std::array<int32_t, 4>{}, 0,
                            (declarer + 1) % 4, trump, 0, 0, 0, 0,
                            std::array<int32_t, 32>{}, std::array<int32_t, 32>{}, 0,
                            false, declarer);
              g.startPlaying();
              const py::ssize_t n = a.shape(0);
              py::array_t<uint8_t> X({n, py::ssize_t(fastpolicy::F_BINARY)});
              py::array_t<float> D({n, py::ssize_t(fastpolicy::F_DENSE)});
              py::array_t<int64_t> L(n);
              py::array_t<int32_t> P(n);
              std::memset(X.mutable_data(), 0, static_cast<size_t>(n) * fastpolicy::F_BINARY);
              auto x = X.mutable_unchecked<2>();
              auto d = D.mutable_unchecked<2>();
              std::vector<int> active;
              for (py::ssize_t i = 0; i < n; ++i) {
                  const int seat = static_cast<int>(g.currentPlayer());
                  float dense[fastpolicy::F_DENSE];
                  const int64_t legal = fastpolicy::features(g, seat, active, dense);
                  for (int k : active)
                      x(i, k) = 1;
                  for (int k = 0; k < fastpolicy::F_DENSE; ++k)
                      d(i, k) = dense[k];
                  L.mutable_at(i) = legal;
                  P.mutable_at(i) = seat;
                  if (!((legal >> a(i)) & 1))
                      throw std::invalid_argument("fast_policy_features: illegal recorded card");
                  g.doMove(Card::from_index(a(i)));
              }
              return py::make_tuple(X, D, L, P);
          },
          py::arg("deal_hands"), py::arg("trump"), py::arg("declarer"), py::arg("cards"),
          "Replay a deal; per ply the fast-policy features of the seat to move: "
          "(binary uint8 [n, FAST_POLICY_BINARY], dense [n, 2], legal masks, seats).");
    py::class_<fastpolicy::Net>(m, "FastPolicyNet")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> w1,
                         py::array_t<float, py::array::c_style | py::array::forcecast> b1,
                         py::array_t<float, py::array::c_style | py::array::forcecast> w2,
                         py::array_t<float, py::array::c_style | py::array::forcecast> b2,
                         py::array_t<float, py::array::c_style | py::array::forcecast> w3,
                         py::array_t<float, py::array::c_style | py::array::forcecast> b3) {
            fastpolicy::Net n;
            n.h1 = static_cast<int>(b1.size());
            n.h2 = static_cast<int>(b2.size());
            n.w1.assign(w1.data(), w1.data() + w1.size());
            n.b1.assign(b1.data(), b1.data() + b1.size());
            n.w2.assign(w2.data(), w2.data() + w2.size());
            n.b2.assign(b2.data(), b2.data() + b2.size());
            n.w3.assign(w3.data(), w3.data() + w3.size());
            n.b3.assign(b3.data(), b3.data() + b3.size());
            n.finalize();
            return n;
        }));
    m.def("fast_policy_load_binary", &fastpolicy::load_binary, py::arg("path"),
          "The net from a learn/fast_policy.py export_binary file (the phone's format).");
    m.def("fast_policy_evaluate",
          [rollout_inputs](const fastpolicy::Net& net, int trump, int declarer, int me,
             std::vector<int> candidates, py::array_t<int64_t> worlds,
             py::array_t<int32_t> hist_seats, py::array_t<int32_t> hist_cards) {
              auto in = rollout_inputs(worlds, hist_seats, hist_cards);
              std::vector<std::vector<double>> r;
              {
                  py::gil_scoped_release release;
                  r = fastpolicy::evaluate_candidates(net, trump, declarer, me, candidates,
                                                      in.first, in.second);
              }
              const py::ssize_t nc = static_cast<py::ssize_t>(candidates.size());
              const py::ssize_t nw = static_cast<py::ssize_t>(in.first.size());
              py::array_t<double> out({nc, nw});
              auto o = out.mutable_unchecked<2>();
              for (py::ssize_t i = 0; i < nc; ++i)
                  for (py::ssize_t j = 0; j < nw; ++j)
                      o(i, j) = r[i][j];
              return out;
          },
          py::arg("net"), py::arg("trump"), py::arg("declarer"), py::arg("me"),
          py::arg("candidates"), py::arg("worlds"), py::arg("hist_seats"),
          py::arg("hist_cards"),
          "Play-out with every seat following the fast policy net: my team's "
          "final points, [candidate][world].");
    m.def("fast_policy_choose",
          [](const fastpolicy::Net& net, py::array_t<int64_t> deal_hands, int trump,
             int declarer, py::array_t<int32_t> cards) {
              auto h = deal_hands.unchecked<1>();
              auto a = cards.unchecked<1>();
              std::array<int64_t, 4> hands{h(0), h(1), h(2), h(3)};
              ManilleGame g(hands, std::array<int32_t, 4>{}, std::array<int32_t, 4>{}, 0,
                            (declarer + 1) % 4, trump, 0, 0, 0, 0,
                            std::array<int32_t, 32>{}, std::array<int32_t, 32>{}, 0,
                            false, declarer);
              g.startPlaying();
              py::array_t<int32_t> out(a.shape(0));
              for (py::ssize_t i = 0; i < a.shape(0); ++i) {
                  out.mutable_at(i) = fastpolicy::choose(net, g, static_cast<int>(g.currentPlayer()));
                  g.doMove(Card::from_index(a(i)));
              }
              return out;
          },
          py::arg("net"), py::arg("deal_hands"), py::arg("trump"), py::arg("declarer"),
          py::arg("cards"),
          "The net's card at every ply of a recorded deal (parity with the trainer).");

    // Native rule agent (cpp/rule_agent.h): the deploy path of
    // learn/rule_agent.py, pinned by tests/test_rule_agent_native.py.
    py::class_<rule::Params>(m, "RuleParams")
        .def(py::init<>())
        .def_readwrite("last_ruff_low", &rule::Params::last_ruff_low)
        .def_readwrite("through_fix", &rule::Params::through_fix)
        .def_readwrite("ruff_convention", &rule::Params::ruff_convention)
        .def_readwrite("trick_rest", &rule::Params::trick_rest)
        .def_readwrite("keep_win_value", &rule::Params::keep_win_value)
        .def_readwrite("keep_pts_weight", &rule::Params::keep_pts_weight)
        .def_readwrite("win_own_pts_w", &rule::Params::win_own_pts_w)
        .def_readwrite("partner_after_f", &rule::Params::partner_after_f)
        .def_readwrite("sim_worlds", &rule::Params::sim_worlds)
        .def_readwrite("sim_vet", &rule::Params::sim_vet)
        .def_readwrite("guard_power", &rule::Params::guard_power)
        .def_readwrite("trump_power", &rule::Params::trump_power)
        .def_readwrite("trump_power_slope", &rule::Params::trump_power_slope)
        .def_readwrite("ruff_future", &rule::Params::ruff_future)
        .def_readwrite("trump_keep_bonus", &rule::Params::trump_keep_bonus)
        .def_readwrite("min_worlds", &rule::Params::min_worlds)
        .def_readwrite("plan_skip_mask", &rule::Params::plan_skip_mask)
        .def_readwrite("plan_pull_min_opp", &rule::Params::plan_pull_min_opp)
        .def_readwrite("plan_majority_margin", &rule::Params::plan_majority_margin)
        .def_readwrite("plan_to_declarer", &rule::Params::plan_to_declarer)
        .def_readwrite("plan_ruff", &rule::Params::plan_ruff)
        .def_readwrite("plan_force_trump", &rule::Params::plan_force_trump)
        .def_readwrite("plan_cash", &rule::Params::plan_cash)
        .def_readwrite("plan_cash_risk", &rule::Params::plan_cash_risk)
        .def_readwrite("plan_est_ruff", &rule::Params::plan_est_ruff)
        .def_readwrite("plan_est_v2", &rule::Params::plan_est_v2)
        .def_readwrite("nt_est_low_run", &rule::Params::nt_est_low_run)
        .def_readwrite("nt_lock_cash_from_trick", &rule::Params::nt_lock_cash_from_trick)
        .def_readwrite("def_trump_lock", &rule::Params::def_trump_lock)
        .def_readwrite("def_trump_lock_weak", &rule::Params::def_trump_lock_weak)
        .def_readwrite("def_trump_margin", &rule::Params::def_trump_margin)
        .def_readwrite("plan_est_v2_ruff", &rule::Params::plan_est_v2_ruff)
        .def_readwrite("plan_partner_top", &rule::Params::plan_partner_top)
        .def_readwrite("plan_exit_sim", &rule::Params::plan_exit_sim)
        .def_readwrite("nt_tiers", &rule::Params::nt_tiers)
        .def_readwrite("trump_lead_pen", &rule::Params::trump_lead_pen)
        .def_readwrite("to_declarer_bonus", &rule::Params::to_declarer_bonus)
        .def_readwrite("bare_honour_lead_pen", &rule::Params::bare_honour_lead_pen)
        .def_readwrite("pull_trumps", &rule::Params::pull_trumps)
        .def_readwrite("pull_min_trumps", &rule::Params::pull_min_trumps)
        .def_readwrite("pull_min_trumps_def", &rule::Params::pull_min_trumps_def)
        .def_readwrite("declarer_pull", &rule::Params::declarer_pull)
        .def_readwrite("declarer_pull_min", &rule::Params::declarer_pull_min)
        .def_readwrite("endgame_master_trump", &rule::Params::endgame_master_trump)
        .def_readwrite("cash_threshold", &rule::Params::cash_threshold)
        .def_readwrite("lead_pts_w", &rule::Params::lead_pts_w)
        .def_readwrite("low_pull_min", &rule::Params::low_pull_min)
        .def_readwrite("low_pull_bonus", &rule::Params::low_pull_bonus)
        .def_readwrite("guard_lead_bonus", &rule::Params::guard_lead_bonus)
        .def_readwrite("bare_honour_pen", &rule::Params::bare_honour_pen)
        .def_readwrite("partner_top_w", &rule::Params::partner_top_w)
        .def_readwrite("opp_ruff_pen", &rule::Params::opp_ruff_pen)
        .def_readwrite("partner_ruff_bonus", &rule::Params::partner_ruff_bonus)
        .def_readwrite("singleton_bonus", &rule::Params::singleton_bonus)
        .def_readwrite("length_w", &rule::Params::length_w)
        .def_readwrite("playout", &rule::Params::playout)
        .def_readwrite("playout_worlds", &rule::Params::playout_worlds)
        .def_readwrite("playout_max_cards", &rule::Params::playout_max_cards)
        .def_readwrite("playout_margin", &rule::Params::playout_margin)
        .def_readwrite("playout_z", &rule::Params::playout_z)
        .def_readwrite("playout_nt_rules", &rule::Params::playout_nt_rules)
        .def_readwrite("playout_tr_rules", &rule::Params::playout_tr_rules)
        .def_readwrite("lik_belief_w", &rule::Params::lik_belief_w)
        .def_readwrite("lik_partner_eps", &rule::Params::lik_partner_eps)
        .def_readwrite("lik_opp_eps", &rule::Params::lik_opp_eps)
        .def_readwrite("root_q_w", &rule::Params::root_q_w)
        .def_readwrite("mull_root_q_w", &rule::Params::mull_root_q_w)
        .def_readwrite("playout_net_mix", &rule::Params::playout_net_mix)
        .def_readwrite("playout_net_plies", &rule::Params::playout_net_plies)
        .def_readwrite("playout_net_trick_end", &rule::Params::playout_net_trick_end)
        .def_readwrite("playout_sir", &rule::Params::playout_sir)
        .def_readwrite("lik_net_partner_w", &rule::Params::lik_net_partner_w)
        .def_readwrite("lik_net_opp_w", &rule::Params::lik_net_opp_w)
        .def_readwrite("lik_net_tau", &rule::Params::lik_net_tau)
        .def_readwrite("mull_endgame_ab_cards", &rule::Params::mull_endgame_ab_cards)
        .def_readwrite("mull_endgame_q_w", &rule::Params::mull_endgame_q_w)
        .def_readwrite("mull_endgame_z", &rule::Params::mull_endgame_z)
        .def_readwrite("suit_net_z", &rule::Params::suit_net_z);
    m.def("rule_agent_choose",
          [rollout_inputs](py::array_t<int64_t> hands,
             py::array_t<int32_t> trick_cards, py::array_t<int32_t> trick_players,
             py::array_t<int32_t> played_cards, py::array_t<int32_t> played_players,
             int n_played_cards, int n_trick, int current_player, int trump_suit,
             int pts0, int pts1, int tricks_won0, int tricks_won1, int declarer,
             int me, py::array_t<int64_t> worlds, py::array_t<int32_t> hist_seats,
             py::array_t<int32_t> hist_cards, py::object belief, py::object q,
             const rule::Params& params, py::object extra, py::object net) {
              auto game = game_from_args(hands, trick_cards, trick_players,
                                         played_cards, played_players, n_played_cards,
                                         n_trick, current_player, trump_suit,
                                         pts0, pts1, tricks_won0, tricks_won1,
                                         false, declarer);
              auto in = rollout_inputs(worlds, hist_seats, hist_cards);
              std::vector<float> b, qv;
              if (!belief.is_none()) {
                  auto a = py::array_t<float, py::array::c_style | py::array::forcecast>(belief);
                  if (a.size() != 96)
                      throw std::invalid_argument("belief must hold 3x32 floats");
                  b.assign(a.data(), a.data() + 96);
              }
              if (!q.is_none()) {
                  auto a = py::array_t<float, py::array::c_style | py::array::forcecast>(q);
                  if (a.size() != 32)
                      throw std::invalid_argument("q must hold 32 floats (by card)");
                  qv.assign(a.data(), a.data() + 32);
              }
              std::vector<double> ex;
              if (!extra.is_none()) {
                  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>(extra);
                  if (static_cast<size_t>(a.size()) != in.first.size())
                      throw std::invalid_argument("extra must hold one log-weight per world");
                  ex.assign(a.data(), a.data() + a.size());
              }
              const fastpolicy::Net* np = net.is_none() ? nullptr
                                                        : net.cast<const fastpolicy::Net*>();
              rule::Decision d;
              {
                  py::gil_scoped_release release;
                  d = rule::choose(game, me, declarer, in.first, in.second,
                                   b.empty() ? nullptr : b.data(),
                                   qv.empty() ? nullptr : qv.data(), params,
                                   ex.empty() ? nullptr : ex.data(), np);
              }
              py::dict vals;
              for (size_t i = 0; i < d.value_cards.size(); ++i)
                  vals[py::int_(d.value_cards[i])] = d.values[i];
              return py::make_tuple(d.card, d.why, vals);
          },
          py::arg("hands"), py::arg("trick_cards"), py::arg("trick_players"),
          py::arg("played_cards"), py::arg("played_players"), py::arg("n_played_cards"),
          py::arg("n_trick"), py::arg("current_player"), py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"), py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("declarer"), py::arg("me"), py::arg("worlds"), py::arg("hist_seats"),
          py::arg("hist_cards"), py::arg("belief"), py::arg("q"), py::arg("params"),
          py::arg("extra") = py::none(), py::arg("net") = py::none(),
          "Native rule agent: (card, why, {card: play-out value}). worlds = the "
          "n_worlds sample (absolute seats); belief 3x32 or None; q by card or None.");

    // Exact late-hand support: every deal sample_determinizations can return,
    // once each. Empty (0, 4) array when more than max_worlds exist (then
    // sample instead). Same arguments as sample_determinizations.
    m.def("enumerate_determinizations",
          [](py::array_t<int64_t> hands, py::array_t<int32_t> trick_cards,
             py::array_t<int32_t> trick_players, py::array_t<int32_t> played_cards,
             py::array_t<int32_t> played_players, int n_played_cards, int n_trick,
             int current_player, int trump_suit, int pts0, int pts1,
             int tricks_won0, int tricks_won1, int observer, int max_worlds,
             int declarer) {
              auto game = game_from_args(
                  hands, trick_cards, trick_players, played_cards, played_players,
                  n_played_cards, n_trick, current_player, trump_suit,
                  pts0, pts1, tricks_won0, tricks_won1, false, declarer);
              static const sampler::TrumpConstrainedDealSampler s_enum_sampler;
              auto ps = sampler::PublicState::fromGame(game, observer);
              std::vector<sampler::Deal> deals;
              {
                  py::gil_scoped_release release;
                  deals = s_enum_sampler.enumerateDeals(ps, static_cast<size_t>(max_worlds));
              }
              py::array_t<int64_t> out({static_cast<py::ssize_t>(deals.size()),
                                        static_cast<py::ssize_t>(4)});
              auto o = out.mutable_unchecked<2>();
              for (size_t i = 0; i < deals.size(); ++i)
                  for (int p = 0; p < 4; ++p)
                      o(static_cast<py::ssize_t>(i), p) = deals[i].hands[p];
              return out;
          },
          py::arg("hands"), py::arg("trick_cards"), py::arg("trick_players"),
          py::arg("played_cards"), py::arg("played_players"), py::arg("n_played_cards"),
          py::arg("n_trick"), py::arg("current_player"), py::arg("trump_suit"),
          py::arg("pts0"), py::arg("pts1"), py::arg("tricks_won0"), py::arg("tricks_won1"),
          py::arg("observer"), py::arg("max_worlds"), py::arg("declarer"),
          "Every consistent determinization once (trump-rule filtered), or an "
          "empty (0, 4) array when more than max_worlds exist.");
}

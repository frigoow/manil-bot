#include "batched_ismcts.h"

#include "encoding.h"
#include "sampler/uniform.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mcts {

namespace {

// C2 (strength_program.md): exact value of the remaining game from `team`'s
// perspective on the nodeValue scale ((pts-30)/30), via alpha-beta over the
// last plies of a determinization. Only called with <= 2 tricks left, so the
// tree is tiny (<= 8 plies). Perfect information *within the determinization*
// — same legal footing as rollouts (plan §3 permits perfect-info leaf eval).
double exactEndgame(ManilleGame& game, int team, double alpha, double beta) {
    const auto moves = game.validMoves();
    if (moves.empty())
        return (game.teamPoints(team) - 30.0) / 30.0;

    const bool maximizing = static_cast<int>(game.currentPlayer()) % 2 == team;
    double best = maximizing ? -2.0 : 2.0;
    for (const Card& move : moves) {
        ManilleGame branch(game); // small state; no undo needed at this depth
        branch.doMove(move);
        const double v = exactEndgame(branch, team, alpha, beta);
        if (maximizing) {
            best = std::max(best, v);
            alpha = std::max(alpha, v);
        } else {
            best = std::min(best, v);
            beta = std::min(beta, v);
        }
        if (beta <= alpha)
            break;
    }
    return best;
}

int popcount64(int64_t mask) {
    return __builtin_popcountll(static_cast<unsigned long long>(mask));
}

} // namespace

BatchedISMCTS::BatchedISMCTS(
    unsigned sims,
    unsigned wave,
    LeafMode mode,
    bool puct,
    double exploration,
    double c_puct,
    double dirichlet_epsilon,
    uint64_t seed,
    bool partner_belief,
    bool opponent_belief)
    : m_sims(sims)
    , m_wave(std::max(1u, wave))
    , m_mode(mode)
    , m_puct(puct)
    , m_exploration(exploration)
    , m_c_puct(c_puct)
    , m_dirichlet_epsilon(dirichlet_epsilon)
    , m_partner_belief(partner_belief)
    , m_opponent_belief(opponent_belief)
    , m_seed(seed)
{
    if (seed != 0)
        m_rng.seed(seed);
    else
        m_rng.seed(std::random_device{}());
}

double BatchedISMCTS::nodeValue(const ManilleGame& terminal, int player) const {
    return (terminal.teamPoints(player % 2) - 30.0) / 30.0;
}

void BatchedISMCTS::applyVirtualLoss(Sim& sim) {
    // Applied incrementally to every node added to the path during selection,
    // so the whole path always carries exactly one virtual loss; finalize()
    // undoes it exactly. (This helper only documents the invariant now.)
    (void)sim;
}

void BatchedISMCTS::finalize(Sim& sim) {
    if (sim.finalized)
        return;
    sim.finalized = true;
    for (Node* n : sim.path) {
        if (n->parent == nullptr)
            continue;
        // Undo virtual loss, then back up the terminal result.
        --n->visits;
        n->score += 1.0;
        ++n->visits;
        n->score += nodeValue(*sim.det, n->player_just_moved);
    }
    ++m_root.visits;
}

// Value-mode backup: v is from the team of the player to move at the
// evaluated state; flip the sign for nodes whose mover is on the other team.
void BatchedISMCTS::finalizeValue(Sim& sim, double v, int team) {
    if (sim.finalized)
        return;
    sim.finalized = true;
    for (Node* n : sim.path) {
        if (n->parent == nullptr)
            continue;
        --n->visits;
        n->score += 1.0;
        ++n->visits;
        n->score += (n->player_just_moved % 2 == team) ? v : -v;
    }
    ++m_root.visits;
}

// Sample one move index from the callback's distribution (defensively
// re-masked to legal). Shared by rollout plies and Phase-S partner nodes.
int BatchedISMCTS::sampleIndex(Sim& sim, const float* p) {
    const auto moves = sim.det->validMoves();
    double weights[8];
    double total = 0.0;
    for (size_t mi = 0; mi < moves.size(); ++mi) {
        const double w =
            std::max(0.0, static_cast<double>(p[moves[mi].index()]));
        weights[mi] = w;
        total += w;
    }
    if (total <= 1e-12) {
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        return moves[pick(m_rng)].index();
    }
    std::uniform_real_distribution<double> u(0.0, total);
    const double r = u(m_rng);
    double acc = 0.0;
    int chosen = moves.back().index();
    for (size_t mi = 0; mi < moves.size(); ++mi) {
        acc += weights[mi];
        if (r <= acc) {
            chosen = moves[mi].index();
            break;
        }
    }
    return chosen;
}

// Sample one rollout move from the callback's distribution, apply it, and
// update the sim's stage.
void BatchedISMCTS::rolloutMove(Sim& sim, const float* p) {
    const auto moves = sim.det->validMoves();
    if (moves.empty()) {
        finalize(sim);
        sim.stage = Sim::Stage::Done;
        return;
    }
    sim.det->doMove(Card::from_index(sampleIndex(sim, p)));
    if (sim.det->validMoves().empty()) {
        finalize(sim);
        sim.stage = Sim::Stage::Done;
    } else {
        sim.stage = Sim::Stage::AwaitEval;
        m_pending.push_back(&sim);
        m_pending_kind.push_back(1);
    }
}

// Phase S: at a partner node, sample the partner's move from their
// information-conditioned policy (pi_theta over the partner's own view) and
// descend — the partner is never tree-searched against the det truth.
void BatchedISMCTS::partnerMove(Sim& sim, const float* p) {
    const int chosen = sampleIndex(sim, p);
    Node* node = sim.path.back();
    Node* child = nullptr;
    for (auto& c : node->children)
        if (c->move == chosen) {
            child = c.get();
            break;
        }
    if (!child) {
        auto nc = std::make_unique<Node>();
        nc->move = chosen;
        nc->player_just_moved = static_cast<int>(sim.det->currentPlayer());
        nc->parent = node;
        node->children.push_back(std::move(nc));
        child = node->children.back().get();
    }
    sim.det->doMove(Card::from_index(chosen));
    sim.path.push_back(child);
    ++child->visits;
    child->score -= 1.0; // virtual loss
    if (sim.det->validMoves().empty()) {
        finalize(sim);
        sim.stage = Sim::Stage::Done;
    } else {
        sim.stage = Sim::Stage::Selecting;
    }
}

// C2: if the sim's determinization has <= 2 tricks left, back up the exact
// minimax value and mark the sim done. Returns true when it handled the sim.
// Not applied at the root — there the exactness belongs to the children's
// evals.
bool BatchedISMCTS::finishExact(Sim& sim) {
    if (m_mode != LeafMode::Value || sim.path.size() <= 1)
        return false;
    const int current = static_cast<int>(sim.det->currentPlayer());
    if (popcount64(sim.det->handMask(current)) > 2)
        return false;
    const int team = current % 2;
    const double v = exactEndgame(*sim.det, team, -2.0, 2.0);
    finalizeValue(sim, v, team);
    sim.stage = Sim::Stage::Done;
    return true;
}

void BatchedISMCTS::startSim(Sim& sim) {
    auto clone = m_root_game->cloneAndRandomise(m_observer);
    sim.det.reset(dynamic_cast<ManilleGame*>(clone.release()));
    if (!sim.det)
        throw std::runtime_error("BatchedISMCTS: determinization failed.");
    sim.path.clear();
    sim.path.push_back(&m_root);
    sim.stage = Sim::Stage::Selecting;
    sim.frontier = nullptr;
    sim.finalized = false;
}

void BatchedISMCTS::advance(Sim& sim) {
    while (true) {
        if (sim.stage == Sim::Stage::Parked) {
            if (sim.frontier->expansion_state != 2)
                return; // still waiting on another sim's eval
            sim.stage = Sim::Stage::Selecting;
        }
        if (sim.stage != Sim::Stage::Selecting)
            return;

        Node* node = sim.path.back();
        const auto moves = sim.det->validMoves();
        if (moves.empty()) {
            finalize(sim);
            sim.stage = Sim::Stage::Done;
            return;
        }

        // C2: exact minimax replaces the NN leaf eval at <= 2 tricks left.
        if (finishExact(sim))
            return;

        // Phase S: the root observer's partner plays from their own
        // information state — sample pi_theta(partner's view), never search.
        if (m_partner_belief
            && static_cast<int>(sim.det->currentPlayer())
                   == (static_cast<int>(m_observer) + 2) % 4) {
            sim.stage = Sim::Stage::AwaitEval;
            sim.frontier = node;
            m_pending.push_back(&sim);
            m_pending_kind.push_back(2);
            return;
        }

        if (!m_puct) {
            // Reference ismcsolver semantics: stop at untried moves, expand
            // one at random, else descend by UCB1.
            std::vector<Node*> tried;
            std::vector<Card> untried;
            for (const Card& move : moves) {
                Node* child = nullptr;
                for (auto& c : node->children)
                    if (c->move == move.index()) {
                        child = c.get();
                        break;
                    }
                if (child)
                    tried.push_back(child);
                else
                    untried.push_back(move);
            }
            if (!untried.empty()) {
                std::uniform_int_distribution<size_t> pick(0, untried.size() - 1);
                const Card move = untried[pick(m_rng)];
                auto child = std::make_unique<Node>();
                child->move = move.index();
                child->player_just_moved =
                    static_cast<int>(sim.det->currentPlayer());
                child->parent = node;
                node->children.push_back(std::move(child));
                Node* next = node->children.back().get();
                sim.det->doMove(move);
                sim.path.push_back(next);
                ++next->visits;
                next->score -= 1.0; // virtual loss
                if (sim.det->validMoves().empty()) {
                    finalize(sim);
                    sim.stage = Sim::Stage::Done;
                    return;
                }
                if (finishExact(sim))
                    return;
                sim.frontier = next;
                sim.stage = Sim::Stage::AwaitEval;
                m_pending.push_back(&sim);
                m_pending_kind.push_back(m_mode == LeafMode::Value ? 0 : 1);
                return;
            }
            for (Node* c : tried)
                ++c->available;
            Node* best = nullptr;
            double best_score = -1e18;
            for (Node* c : tried) {
                const double mean = c->visits > 0
                    ? c->score / static_cast<double>(c->visits)
                    : 0.0;
                const double u = c->visits > 0
                    ? mean + m_exploration * std::sqrt(
                          std::log(static_cast<double>(c->available))
                          / static_cast<double>(c->visits))
                    : 1e18;
                if (u > best_score) {
                    best_score = u;
                    best = c;
                }
            }
            sim.det->doMove(Card::from_index(best->move));
            sim.path.push_back(best);
            ++best->visits;
            best->score -= 1.0; // virtual loss
            continue;
        }

        // PUCT path.
        if (node->expansion_state == 1) {
            sim.stage = Sim::Stage::Parked;
            sim.frontier = node;
            return;
        }
        if (node->expansion_state == 0) {
            // C2: exact endgame replaces the expansion eval entirely — and
            // must run before children/expansion_state exist, so sims parked
            // on a peer's expansion never deadlock.
            if (finishExact(sim))
                return;
            for (const Card& move : moves) {
                auto child = std::make_unique<Node>();
                child->move = move.index();
                child->player_just_moved =
                    static_cast<int>(sim.det->currentPlayer());
                child->parent = node;
                node->children.push_back(std::move(child));
            }
            node->expansion_state = 1;
            sim.stage = Sim::Stage::AwaitEval;
            sim.frontier = node;
            m_pending.push_back(&sim);
            m_pending_kind.push_back(0);
            return;
        }
        // Evaluated: PUCT selection among children legal in this
        // determinization. Two R2 fixes vs the naive version:
        //  - late-legal children (illegal in the expanding determinization,
        //  so policy prob 0) get a uniform-floor prior instead of starving;
        //  - priors are renormalized over the currently-legal subset, since
        //  legal sets vary across determinizations in an information-set
        //  tree.
        const double sqrt_n = std::sqrt(static_cast<double>(node->visits));
        std::vector<Node*> legal_children;
        legal_children.reserve(moves.size());
        double prior_sum = 0.0;
        for (const Card& move : moves) {
            Node* child = nullptr;
            for (auto& c : node->children)
                if (c->move == move.index()) {
                    child = c.get();
                    break;
                }
            if (!child) {
                auto nc = std::make_unique<Node>();
                nc->move = move.index();
                nc->player_just_moved =
                    static_cast<int>(sim.det->currentPlayer());
                nc->parent = node;
                nc->prior = 0.9f * node->policy[move.index()]
                    + 0.1f / 32.0f; // uniform floor (R2)
                nc->expansion_state = 2; // policy already known at parent
                node->children.push_back(std::move(nc));
                child = node->children.back().get();
            }
            legal_children.push_back(child);
            prior_sum += child->prior;
        }
        Node* best = nullptr;
        double best_score = -1e18;
        for (Node* child : legal_children) {
            const double q = child->visits > 0
                ? child->score / static_cast<double>(child->visits)
                : 0.0;
            const double prior = prior_sum > 1e-9
                ? child->prior / prior_sum
                : 1.0 / static_cast<double>(legal_children.size());
            const double u = m_c_puct * prior * sqrt_n
                / (1.0 + static_cast<double>(child->visits));
            if (q + u > best_score) {
                best_score = q + u;
                best = child;
            }
        }
        sim.det->doMove(Card::from_index(best->move));
        sim.path.push_back(best);
        ++best->visits;
        best->score -= 1.0; // virtual loss
    }
}

void BatchedISMCTS::advanceAll() {
    while (!done() && m_pending.empty()) {
        if (m_active.size() < m_wave
            && m_completed + m_active.size() < m_sims) {
            auto sim = std::make_unique<Sim>();
            startSim(*sim);
            m_active.push_back(sim.get());
            m_pool.push_back(std::move(sim));
            continue;
        }
        if (m_active.empty())
            return; // budget exhausted (should coincide with done())

        // Advance every active sim once, accumulating eval requests across
        // the whole wave before returning (R1: the early break made every
        // NN call batch size 1, silently turning wave into a no-op). Sims
        // that go AwaitEval stay in m_pending until provide(); sims that
        // reach a node another sim is expanding simply park.
        bool erased = false;
        for (size_t i = 0; i < m_active.size();) {
            Sim* sim = m_active[i];
            advance(*sim);
            if (sim->stage == Sim::Stage::Done) {
                ++m_completed;
                m_active.erase(m_active.begin() + i);
                erased = true;
            } else {
                ++i;
            }
        }
        if (!m_pending.empty())
            return;
        if (!erased) {
            // Nothing finished this pass. Parked sims whose frontier got
            // evaluated will resume next pass; if none can, the expanding
            // sim's eval must be pending — but m_pending is empty, so this
            // is a defensive bail against spinning (should be unreachable).
            bool any_resumable = false;
            for (Sim* s : m_active)
                if (s->stage == Sim::Stage::Selecting
                    || (s->stage == Sim::Stage::Parked
                        && s->frontier->expansion_state == 2))
                    any_resumable = true;
            if (!any_resumable)
                return;
        }
    }
}

void BatchedISMCTS::begin(ManilleGame root) {
    m_root_game = std::make_unique<ManilleGame>(std::move(root));
    if (m_seed != 0)
        sampler::UniformDealSampler::seedPrng(m_seed);
    m_observer = m_root_game->currentPlayer();
    m_root = Node{};
    m_root.player_just_moved = static_cast<int>(m_observer);
    m_root.expansion_state = 0;
    m_pool.clear();
    m_active.clear();
    m_pending.clear();
    m_pending_kind.clear();
    m_completed = 0;
    advanceAll();
}

bool BatchedISMCTS::done() const {
    return m_completed >= m_sims && m_active.empty();
}

int BatchedISMCTS::pendingCount() const {
    return static_cast<int>(m_pending.size());
}

void BatchedISMCTS::pendingObs(float* out) {
    for (size_t i = 0; i < m_pending.size(); ++i) {
        Sim* sim = m_pending[i];
        encoding::encode(
            *sim->det,
            static_cast<int>(sim->det->currentPlayer()),
            out + i * encoding::OBS_DIM);
    }
}

void BatchedISMCTS::provide(const float* probs, const float* values) {
    std::vector<Sim*> pending = std::move(m_pending);
    std::vector<int> kinds = std::move(m_pending_kind);
    m_pending.clear();
    m_pending_kind.clear();

    for (size_t i = 0; i < pending.size(); ++i) {
        Sim* sim = pending[i];
        const float* p = probs + i * 32;
        const float v = values[i];

        if (kinds[i] == 0) {
            // Frontier eval: assign priors (PUCT) and/or back up the value.
            Node* node = sim->frontier;
            if (m_puct) {
                std::copy(p, p + 32, node->policy.begin());
                for (auto& c : node->children)
                    c->prior = p[c->move];
                node->expansion_state = 2;
                if (node == &m_root && m_dirichlet_epsilon > 0.0) {
                    const double alpha = 10.0
                        / std::sqrt(static_cast<double>(node->children.size()));
                    double total = 0.0;
                    std::vector<double> noise(node->children.size());
                    for (size_t k = 0; k < noise.size(); ++k) {
                        std::gamma_distribution<double> g(alpha, 1.0);
                        noise[k] = g(m_rng);
                        total += noise[k];
                    }
                    if (total > 0.0)
                        for (size_t k = 0; k < noise.size(); ++k)
                            node->children[k]->prior = static_cast<float>(
                                (1.0 - m_dirichlet_epsilon)
                                    * node->children[k]->prior
                                + m_dirichlet_epsilon * noise[k] / total);
                }
            }
            if (m_mode == LeafMode::Value) {
                const int team = static_cast<int>(sim->det->currentPlayer()) % 2;
                finalizeValue(*sim, std::max(-1.0f, std::min(1.0f, v)), team);
                sim->stage = Sim::Stage::Done;
            } else {
                // Rollout: this eval's policy covers the first rollout ply
                // from the frontier state, so no extra eval is needed.
                rolloutMove(*sim, p);
            }
        } else if (kinds[i] == 2) {
            partnerMove(*sim, p);
        } else {
            rolloutMove(*sim, p);
        }
    }
    advanceAll();
}

SolverResult BatchedISMCTS::result() const {
    SolverResult result;
    result.total_visits = 0;
    result.root_value = 0.0;
    result.best_move = -1;
    unsigned max_visits = 0;
    double weighted = 0.0;
    for (auto& child : m_root.children) {
        result.visit_counts[child->move] = child->visits;
        result.total_visits += child->visits;
        const double mean = child->visits > 0
            ? child->score / static_cast<double>(child->visits)
            : 0.0;
        weighted += child->visits * mean;
        if (child->visits > max_visits) {
            max_visits = child->visits;
            result.best_move = child->move;
        }
    }
    if (result.total_visits > 0)
        result.root_value = weighted / result.total_visits;
    return result;
}

SolverResult BatchedISMCTS::solve(const ManilleGame& root, const PolicyValueFn& fn) {
    begin(root);
    std::vector<float> obs, probs, values;
    while (!done()) {
        const int n = pendingCount();
        if (n <= 0)
            break;
        obs.resize(static_cast<size_t>(n) * encoding::OBS_DIM);
        probs.resize(static_cast<size_t>(n) * 32);
        values.resize(n);
        pendingObs(obs.data());
        fn(obs.data(), n, probs.data(), values.data());
        provide(probs.data(), values.data());
    }
    return result();
}

} // namespace mcts

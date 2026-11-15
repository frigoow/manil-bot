#include "solver_wrapper.h"

#include <ismcts/sosolver.h>
#include <ismcts/mosolver.h>
#include <ismcts/tree/ucb1.h>

#include <chrono>
#include <thread>
#include <vector>

// Helper: extract visit counts and value from a root node's children.
// The default tree policy is UCB1, so children are UCBNode<Card>.
static SolverResult extractResults(
    const ISMCTS::Node<Card>* root,
    unsigned int result_player)
{
    SolverResult result;
    result.total_visits = 0;
    result.root_value = 0.0;
    result.best_move = -1;

    unsigned max_visits = 0;
    double weighted_value = 0.0;

    for (auto& child : root->children()) {
        int card_idx = child->move().index();
        unsigned v = child->visits();
        result.visit_counts[card_idx] = v;
        result.total_visits += v;

        // Get mean value from UCBNode
        auto* ucb_node = dynamic_cast<const ISMCTS::UCBNode<Card>*>(child.get());
        double mean_val = 0.0;
        if (ucb_node && v > 0) {
            // ucbScore with exploration=0 gives mean reward = score/visits
            mean_val = ucb_node->ucbScore(0.0);
        }
        weighted_value += v * mean_val;

        if (v > max_visits) {
            max_visits = v;
            result.best_move = card_idx;
        }
    }

    if (result.total_visits > 0)
        result.root_value = weighted_value / result.total_visits;

    return result;
}

// --- SOSolver wrapper ---

ManilleSolverSO::ManilleSolverSO(unsigned iterations, bool perfect_info)
    : m_iterations(iterations)
    , m_perfect_info(perfect_info)
{}

SolverResult ManilleSolverSO::solve(const ManilleGame& game) {
    ISMCTS::SOSolver<Card> solver(m_iterations);
    solver(game);  // run search
    auto trees = solver.currentTrees();
    // Sequential: single tree, root is trees[0]
    return extractResults(trees[0].get(), game.currentPlayer());
}

// --- MOSolver wrapper ---

ManilleSolverMO::ManilleSolverMO(unsigned iterations, bool perfect_info)
    : m_iterations(iterations)
    , m_perfect_info(perfect_info)
{}

SolverResult ManilleSolverMO::solve(const ManilleGame& game) {
    ISMCTS::MOSolver<Card> solver(m_iterations);
    solver(game);  // run search
    auto trees = solver.currentTrees();
    // Sequential: single TreeMap, get tree for current player
    auto& tree_map = trees[0];
    auto it = tree_map.find(game.currentPlayer());
    return extractResults(it->second.get(), game.currentPlayer());
}

// --- Root-parallel time-boxed solver ---

ManilleSolverTime::ManilleSolverTime(int threads, int budget_ms, bool perfect_info)
    : m_threads(std::max(1, threads))
    , m_budget_ms(std::max(1, budget_ms))
    , m_perfect_info(perfect_info)
{}

SolverResult ManilleSolverTime::solve(const ManilleGame& game) {
    if (m_iters_per_sec <= 0.0) {
        // One-time calibration probe at the real thread count, so memory
        // contention is priced into the rate. Each thread fills the whole
        // budget wall-clock with its own search; the budget is NOT split.
        constexpr unsigned kProbeIters = 1000;
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(m_threads));
            for (int t = 0; t < m_threads; ++t) {
                pool.emplace_back([&] {
                    ManilleSolverSO worker(kProbeIters, m_perfect_info);
                    worker.solve(game);
                });
            }
            for (auto& th : pool)
                th.join();
        }
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        m_iters_per_sec = kProbeIters / std::max(secs, 1e-6);
    }

    const unsigned per_thread = static_cast<unsigned>(std::max(
        1.0, m_iters_per_sec * (m_budget_ms / 1000.0) * 0.95));

    std::vector<SolverResult> partials(static_cast<size_t>(m_threads));
    {
        std::vector<std::thread> pool;
        pool.reserve(partials.size());
        for (int t = 0; t < m_threads; ++t) {
            pool.emplace_back([&, t] {
                ManilleSolverSO worker(per_thread, m_perfect_info);
                partials[static_cast<size_t>(t)] = worker.solve(game);
            });
        }
        for (auto& th : pool)
            th.join();
    }

    SolverResult merged;
    merged.total_visits = 0;
    merged.root_value = 0.0;
    merged.best_move = -1;
    for (const auto& part : partials) {
        for (const auto& kv : part.visit_counts)
            merged.visit_counts[kv.first] += kv.second;
        merged.total_visits += part.total_visits;
        merged.root_value += part.root_value * part.total_visits;
    }
    unsigned max_visits = 0;
    for (const auto& kv : merged.visit_counts) {
        if (kv.second > max_visits) {
            max_visits = kv.second;
            merged.best_move = kv.first;
        }
    }
    if (merged.total_visits > 0)
        merged.root_value /= merged.total_visits;
    return merged;
}

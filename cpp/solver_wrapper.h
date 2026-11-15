#ifndef MANILLE_SOLVER_WRAPPER_H
#define MANILLE_SOLVER_WRAPPER_H

#include "card.h"
#include "manille_game.h"

#include <map>

struct SolverResult {
    int best_move;                        // card index 0-31
    std::map<int, unsigned> visit_counts; // card_index -> visits
    unsigned total_visits;
    double root_value;                    // weighted avg of child values
};

class ManilleSolverSO {
public:
    explicit ManilleSolverSO(unsigned iterations, bool perfect_info = false);
    SolverResult solve(const ManilleGame& game);
    bool perfectInfo() const { return m_perfect_info; }

private:
    unsigned m_iterations;
    bool m_perfect_info;
};

class ManilleSolverMO {
public:
    explicit ManilleSolverMO(unsigned iterations, bool perfect_info = false);
    SolverResult solve(const ManilleGame& game);
    bool perfectInfo() const { return m_perfect_info; }

private:
    unsigned m_iterations;
    bool m_perfect_info;
};

// Root-parallel plain ISMCTS under a wall-clock budget (strength_program.md
// Phase B): K independent SOSolver runs from the same root in threads,
// root visit counts summed. Iteration count is auto-tuned from a one-time
// calibration probe to fill the budget at ~92%.
class ManilleSolverTime {
public:
    ManilleSolverTime(int threads, int budget_ms, bool perfect_info = false);
    SolverResult solve(const ManilleGame& game);
    bool perfectInfo() const { return m_perfect_info; }

private:
    int m_threads;
    int m_budget_ms;
    bool m_perfect_info;
    double m_iters_per_sec = 0.0;  // lazily calibrated per process
};

#endif // MANILLE_SOLVER_WRAPPER_H

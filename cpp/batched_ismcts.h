#ifndef MANILLE_BATCHED_ISMCTS_H
#define MANILLE_BATCHED_ISMCTS_H

#include "manille_game.h"
#include "solver_wrapper.h"

#include <array>
#include <functional>
#include <memory>
#include <random>
#include <vector>

// Batched NN-guided ISMCTS (plan Phase 2): SO-ISMCTS whose leaf evaluation
// comes from the networks instead of random rollouts.
//
// Two tree policies:
//   - UCB1 (puct=false): reference ismcsolver semantics — one-child expansion,
//     mover-perspective scores, 'available' accounting.
//   - PUCT (puct=true):  AlphaZero-style — full expansion with pi_theta
//     priors, Q + c_puct * P * sqrt(N) / (1 + n). Priors come from the same
//     batched policy evaluation as the leaf value, so value mode costs one
//     NN eval per simulation.
//
// Two leaf modes:
//   - Rollout: play out the determinization with moves sampled from pi_theta.
//   - Value:   no rollout; back up v_theta at the frontier.
//
// Simulations run in waves with virtual loss so NN inference is batched:
// advanceAll() accumulates up to `wave` in-flight sims' eval requests into
// one batch. wave=1 is reference-quality sequential search — that is what
// the resumable API (begin/pending_obs/provide) is driven with by the
// self-play generator: many games interleaved at wave=1, with the NN batch
// formed across games (cross-game batching, plan §6).
//
// The policy callback receives a batch of observations (encoded from each
// simulated player's own perspective inside its determinization — never an
// oracle view) and fills legal-masked policy probabilities and value
// estimates in [-1, 1] from the current player's team perspective.
//
// Phase S (belief models for other seats): with partner_belief=true, nodes
// where the root observer's PARTNER is to move are not tree-searched against
// the determinized truth; with opponent_belief=true the same holds for both
// OPPONENT seats. In each case the sim requests a policy eval (pending
// kind 2 — encoded from that player's own perspective, i.e. exactly their
// information state) and samples the move from it. Informative plays earn
// value inside the tree (partner reads only what's public), and deceptive
// structure survives (opponents counter only what they could know).

namespace mcts {

using PolicyValueFn =
    std::function<void(const float* obs, int n, float* probs, float* values)>;

struct Node {
    int move = -1;               // card index that led to this node (-1: root)
    int player_just_moved = -1;  // who played `move`
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    unsigned visits = 0;
    double score = 0.0;
    unsigned available = 1;      // UCB1 'available' count, as in ismcsolver
    // PUCT state:
    float prior = 0.0f;                       // prior prob of `move`
    int expansion_state = 0;                  // 0 fresh, 1 expanding, 2 evaluated
    std::array<float, 32> policy{};           // policy at this node (if evaluated)
};

enum class LeafMode { Rollout, Value };

class BatchedISMCTS {
public:
    BatchedISMCTS(
        unsigned sims,
        unsigned wave,
        LeafMode mode,
        bool puct = false,
        double exploration = 0.7,
        double c_puct = 2.0,
        double dirichlet_epsilon = 0.0,
        uint64_t seed = 0,
        bool partner_belief = false,
        bool opponent_belief = false
    );

    // One-shot driver: runs the whole search, calling fn for NN batches.
    SolverResult solve(const ManilleGame& root, const PolicyValueFn& fn);

    // Resumable driver (cross-game batching). begin() resets; then alternate
    // pending_obs()/provide() until done(); result() at the end.
    void begin(ManilleGame root);
    bool done() const;
    // Encode the pending evaluations into out (capacity pending_count() *
    // encoding::OBS_DIM floats).
    int pendingCount() const;
    void pendingObs(float* out);
    // Provide policy probs [pending_count * 32] and values [pending_count],
    // in the same order as pendingObs, then advance until more evals are
    // needed or the search is done.
    void provide(const float* probs, const float* values);
    SolverResult result() const;

private:
    struct Sim {
        std::unique_ptr<ManilleGame> det;
        std::vector<Node*> path;
        enum class Stage { Selecting, Parked, AwaitEval, Done } stage =
            Stage::Selecting;
        Node* frontier = nullptr;  // node whose eval this sim waits for
        bool finalized = false;
    };

    unsigned m_sims;
    unsigned m_wave;
    LeafMode m_mode;
    bool m_puct;
    double m_exploration;
    double m_c_puct;
    double m_dirichlet_epsilon;
    bool m_partner_belief;  // Phase S: sample partner moves from pi_theta
    bool m_opponent_belief; // Phase S: sample opponent moves from pi_theta
    uint64_t m_seed;
    std::mt19937 m_rng;

    // Resumable state.
    Node m_root;
    std::unique_ptr<ManilleGame> m_root_game;
    std::vector<std::unique_ptr<Sim>> m_pool;
    std::vector<Sim*> m_active;
    std::vector<Sim*> m_pending;   // sims waiting for an NN eval
    std::vector<int> m_pending_kind; // 0 = frontier eval, 1 = rollout ply
    unsigned m_completed = 0;
    unsigned m_observer = 0;

    void startSim(Sim& sim);
    void advance(Sim& sim);        // progress until eval needed / parked / done
    void advanceAll();             // advance sims / start new ones as possible
    void finalize(Sim& sim);
    void finalizeValue(Sim& sim, double v, int team);
    bool finishExact(Sim& sim);    // C2: exact minimax at <= 2 tricks left
    int sampleIndex(Sim& sim, const float* probs);  // legal-masked sample
    void rolloutMove(Sim& sim, const float* probs); // sample + apply one move
    void partnerMove(Sim& sim, const float* probs); // Phase S: sample + descend
    void applyVirtualLoss(Sim& sim);
    double nodeValue(const ManilleGame& terminal, int player_just_moved) const;
};

} // namespace mcts

#endif // MANILLE_BATCHED_ISMCTS_H

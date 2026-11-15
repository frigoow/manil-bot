"""ISMCTS endgame handoff (endgame_leak_plan.md arm C, 2026-07-25).

SPARTA's exact-minimax-per-world leaves are PIMC: within each
determinization partner AND opponents play double-dummy (strategy fusion),
so endgame values misprice uninformed play — the suspected mechanism behind
the human-facing endgame point leak. This agent plays SPARTA until trick
`handoff_trick`, then hands the decision to the C++ ISMCTS solver with
SPARTA's own belief-weighted particle pool injected as the determinization
set (`solver.solve(..., particles=, particle_weights=)` — the CLAUDE.md
belief hook). ISMCTS shares one action per information set across
determinizations, modelling partner/opponents as uninformed.
"""
import numpy as np

from manille_env.card import Card

from .dmc_sparta import SpartaDMCAgent


class SpartaIsmctsHandoffAgent(SpartaDMCAgent):
    """SPARTA pre-`handoff_trick`, ISMCTS-on-SPARTA-worlds from then on.

    `endgame_handoff` marks the agent for the endgame diagnostic, which
    drives endgame plies through choose_card (ISMCTS) instead of analyze.
    """

    endgame_handoff = True

    def __init__(self, *args, handoff_trick=6, ismcts_iterations=5000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        import manille_ismcts as mi
        self.handoff_trick = handoff_trick
        self.ismcts_iterations = ismcts_iterations
        self.solver = mi.ManilleSolverSO(iterations=ismcts_iterations,
                                         perfect_info=False)

    def choose_card(self, game, player_id, played_cards,
                    played_players) -> Card:
        legal = game.get_legal_moves(player_id)
        if len(legal) <= 1:
            return legal[0]
        trick_idx = (len(played_cards) + len(game.current_trick)) // 4
        if trick_idx < self.handoff_trick:
            return super().choose_card(game, player_id, played_cards,
                                       played_players)
        particles, weights, arrays, _, _, _ = self._build_worlds(
            game, player_id, played_cards, played_players)
        result = self.solver.solve(
            *arrays,
            particles=np.ascontiguousarray(particles),
            particle_weights=np.ascontiguousarray(weights))
        best = int(result["best_move"])
        for card in legal:
            if card.to_index() == best:
                return card
        return legal[0]  # solver fallback (should not happen)

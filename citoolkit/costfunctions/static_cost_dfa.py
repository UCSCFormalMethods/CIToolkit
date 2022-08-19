""" Contains the StaticCostDfa class"""

from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

import time
from numbers import Rational

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.costfunctions.cost_func import ExactCostFunc


class StaticCostDfa(ExactCostFunc):
    """Class encoding a Static Cost Deterministic Finite Automata.
    This is represented with a Dfa that has each accepting state
    mapped to a cost. A word has a cost if it is accepted by the
    Dfa and the cost associated with that word is the one that is
    associated with that accepting state.

    :param dfa: A Dfa specification, which accepts words that have a cost.
    :param cost_map: A dictionary mapping each accepting state in dfa to
        a cost.
    """

    def __init__(self, dfa: Dfa, cost_map: dict[State, Rational]):
        # Initialize super class and stores attributes.
        self.dfa = dfa
        self.cost_map = {State(state): cost for (state, cost) in cost_map.items()}
        costs = frozenset(self.cost_map.values())

        super().__init__(self.dfa.alphabet, costs)

        # Perform checks to ensure a well formed Static Cost Dfa.
        for target_state in self.dfa.accepting_states:
            if target_state not in self.cost_map.keys():
                raise ValueError(
                    f"The accepting state '{target_state}' is missing an associated"
                    " cost in cost_map"
                )

        # Initialize cache values to None
        self.decomp_cost_func = None

    ####################################################################################################
    # CostFunc Functions
    ####################################################################################################

    def cost(self, word) -> Rational | None:
        """Returns the appropriate cost for a word. This cost is
        found by first checking if the interior Dfa accepts a word.
        If it does, then the accepting state that the Dfa terminates
        in is mapped through cost_map to determine the cost for
        that word. If the word has no cost, i.e. it is not accepted
        by this Dfa, returns None.

        :param word: A word over this cost function's alphabet.
        :returns: The cost associated with this word.
        """
        if self.dfa.accepts(word):
            return self.cost_map[self.dfa.get_terminal_state(word)]

        return None

    def decompose(self, num_threads: int = 1) -> dict[Rational, Dfa]:
        """Decomposes this cost function into a Dfa indicator
        function for each cost.

        :returns: A dictionary mapping each cost to a Dfa Spec that
            accepts if and only if a word has that cost.
        """
        # Check if value is cached.
        if self.decomp_cost_func is not None:
            logger.debug("Returning cached StaticCostDfa Decomposition.")

            return self.decomp_cost_func

        # Compute decomposed cost function
        start_time = time.time()
        logger.info("Computing StaticCostDfa Decomposition...")

        self.decomp_cost_func = {}

        # Compute a mapping from each cost to its associated accepting states.
        cost_states_mapping = {cost: set() for cost in self.costs}

        for accepting_state in self.dfa.accepting_states:
            cost = self.cost_map[accepting_state]
            cost_states_mapping[cost].add(accepting_state)

        # Compute the indicator function for each cost
        for cost in self.costs:
            indicator_dfa = Dfa(
                self.dfa.alphabet,
                self.dfa.states,
                cost_states_mapping[cost],
                self.dfa.start_state,
                self.dfa.transitions,
            )
            self.decomp_cost_func[cost] = indicator_dfa

        wall_time = time.time() - start_time
        logger.info(
            "StaticCostDfa Decomposition completed. Wallclock Runtime: %.4f", wall_time
        )

        return self.decomp_cost_func

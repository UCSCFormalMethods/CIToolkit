""" Contains the AccumulatedCostDfa class."""

from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

import time
from numbers import Rational
from collections import Counter

from multiprocess import Pool

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.costfunctions.cost_func import ExactCostFunc


class AccumulatedCostDfa(ExactCostFunc):
    """Class encoding an Accumulated Cost Deterministic Finite Automata.
    This is represented with a Dfa that has each state mapped to a cost.
    A word has a cost if and only if it is accepted by the Dfa. The cost
    of any word that has one is the sum of all the costs associated with
    each state in its accepting path. The max length of a word must also
    be specified to ensure that the set of costs is always finite.

    :param dfa: A Dfa specification, which accepts words that have a cost.
    :param cost_map: A dictionary mapping each state in dfa to a cost.
    :param max_word_length: The maximum length of a word that has a cost.
        Any word with length longer than this will not have a cost associated
        with it.
    """

    def __init__(self, dfa: Dfa, cost_map: dict[State, Rational], max_word_length: int):
        # Store attributes
        self.dfa = dfa
        self.cost_map = {State(state): cost for (state, cost) in cost_map.items()}
        self.max_word_length = max_word_length

        # Perform checks to ensure a well formed Static Cost Dfa.
        for target_state in self.dfa.states:
            if target_state not in self.cost_map.keys():
                raise ValueError(
                    f"The state '{target_state}' is missing an associated cost in"
                    " cost_map"
                )

        for cost in self.cost_map.values():
            if not isinstance(cost, Rational):
                raise ValueError(
                    f"'{cost}' is not of the type Rational, and therefore cannot be a"
                    " cost. Consider constructing one using the 'fractions' library."
                )
            if cost < 0:
                raise ValueError(
                    f"'{cost}' is less than zero, and therefore cannot be a cost."
                )

        # Compute the cost_table for this AccumulatedCostDfa, which maps
        # a state and a word length tuple (state, cost) to a multiset
        # of the costs of all words ending in a state with a certain length.
        self.cost_table = self._compute_cost_table(
            self.dfa, self.cost_map, self.max_word_length
        )

        # Compute the set of intermediate costs that this cost function
        # can assign based on the cost table.
        self.intermediate_costs = set()

        for counter in self.cost_table.values():
            self.intermediate_costs = self.intermediate_costs | counter.keys()

        # Compute the set of costs that this cost function can assign
        # based on the cost table and initialize the CostFunc super class.
        costs = set()

        for accepting_state in self.dfa.accepting_states:
            for current_length in range(0, self.max_word_length + 1):
                costs = costs | set(self.cost_table[(accepting_state, current_length)])

        super().__init__(self.dfa.alphabet, costs)

        # Initialize cache values to None
        self.decomp_cost_func = None

    ####################################################################################################
    # CostFunc Functions
    ####################################################################################################

    def cost(self, word) -> Rational | None:
        """Returns the appropriate cost for a word. This cost is
        found by first checking if the interior Dfa accepts a word.
        If it does, then the each state in the accepting path is
        mapped through the cost_map and this list of costs is summed
        and returned.

        :param word: A word over this cost function's alphabet.
        :returns: The cost associated with this word.
        """
        if self.dfa.accepts(word) and len(word) <= self.max_word_length:
            state_path = self.dfa.get_state_path(word)

            cost = sum([self.cost_map[state] for state in state_path])

            return cost
        else:
            return None

    def decompose(self, num_threads: int = 1) -> dict[Rational, Dfa]:
        """Decomposes this cost function into a Dfa indicator
        function for each cost.

        :returns: A dictionary mapping each cost to a Dfa Spec that
        accepts if and only if a word has that cost.
        """
        # Check if value is cached.
        if self.decomp_cost_func is not None:
            logger.debug("Returning cached AccumulatedCostDfa Decomposition.")

            return self.decomp_cost_func

        # Compute decomposed cost function
        start_time = time.time()
        logger.info("Computing AccumulatedCostDfa decomposition...")

        # If there are no costs return an empty dictionary.
        if len(self.costs) == 0:
            return {}

        # Create a new state for every combination of original state
        # and cost in between the 0 and the max cost possible.
        state_rename_map = {}
        state_iter = 0

        # While creating new states record all states that are accepting
        # and have accumulated a particular cost.
        cost_accepting_map = {cost: set() for cost in self.costs}

        for state in self.dfa.states:
            for cost in sorted(self.intermediate_costs, key=lambda x: str(x)):
                new_state = "State_" + str(state_iter) + "_" + str(cost)
                state_rename_map[(state, cost)] = new_state

                if state in self.dfa.accepting_states and cost in self.costs:
                    cost_accepting_map[cost].add(new_state)

            state_iter += 1

        # Add a sink state and a cost state which is the new state associated
        # with the original start state and its cost.
        new_states = set(state_rename_map.values()) | {"Sink"}
        assert (
            self.dfa.start_state,
            self.cost_map[self.dfa.start_state],
        ) in state_rename_map, (
            self.dfa.start_state,
            self.cost_map[self.dfa.start_state],
        )
        new_start_state = state_rename_map[
            (self.dfa.start_state, self.cost_map[self.dfa.start_state])
        ]

        # Create a new transition map over the new states and preserves the
        # original transition relation, but also accounts for accumulating cost.
        new_transitions = {}

        for state in self.dfa.states:
            for cost in sorted(self.intermediate_costs, key=lambda x: str(x)):
                for symbol in self.dfa.alphabet:
                    dest_state = self.dfa.transitions[(state, symbol)]

                    new_origin_state = state_rename_map[(state, cost)]

                    # Calculate what the cost would be when arriving at the
                    # destination state. If it is too high to be feasible,
                    # instead map to the Sink state.
                    new_cost = cost + self.cost_map[dest_state]

                    # If new cost is unreachable or too large, then ignore.
                    if new_cost not in self.intermediate_costs or new_cost > max(
                        self.costs
                    ):
                        new_dest_state = "Sink"
                    else:
                        new_dest_state = state_rename_map[(dest_state, new_cost)]

                    new_transitions[(new_origin_state, symbol)] = new_dest_state

        # Complete transition function making the Sink state a trap state.
        for symbol in self.dfa.alphabet:
            new_transitions[("Sink", symbol)] = "Sink"

        # Create each cost indicator function, which are all identical except
        # that they only accept at states which have the correct accepting cost,
        # and then minimize each indicator function. If num_threads >1, multithread
        # in this computation, as it can be heavy.
        if num_threads <= 1:
            # Use one thread
            cpu_time = -1
            decomp_cost_func = {
                cost: Dfa(
                    self.dfa.alphabet,
                    new_states,
                    cost_accepting_map[cost],
                    new_start_state,
                    new_transitions,
                ).minimize()
                for cost in self.costs
            }
        else:
            # Multithread using a multiprocessing pool
            with Pool(num_threads) as pool:
                # Helper function for pool.map
                def cost_indicator_wrapper(cost):
                    process_start_time = time.process_time()

                    # Create cost indicator spec
                    spec = Dfa(
                        self.dfa.alphabet,
                        new_states,
                        cost_accepting_map[cost],
                        new_start_state,
                        new_transitions,
                    ).minimize()

                    return (cost, spec, time.process_time() - process_start_time)

                pool_output = pool.map(cost_indicator_wrapper, self.costs)

                # Extract relevant info from pool_output
                cpu_time = sum([runtime for _, _, runtime in pool_output])

                decomp_cost_func = {cost: spec for cost, spec, _ in pool_output}

        self.decomp_cost_func = decomp_cost_func

        wall_time = time.time() - start_time
        logger.info(
            "AccumulatedCostDfa deconstruction completed. Wallclock Runtime: %.4f CPU"
            " Runtime: %.4f",
            wall_time,
            cpu_time,
        )

        return decomp_cost_func

    ####################################################################################################
    # AccumulatedCostDfa Property Functions
    ####################################################################################################

    @staticmethod
    def _compute_cost_table(
        dfa: Dfa, cost_map: dict[State, Rational], max_word_length: int
    ):
        """Computes the cost table for this accumulated cost dfa, which
        maps a state and a word length tuple (s, l) to a multiset
        of the costs of all words ending in state s with length l.

        :param dfa: A Dfa specification, which accepts words that have a cost.
        :param cost_map: A dictionary mapping each state in dfa to feasible costs.
        :param max_word_length: The maximum length of a word that has a cost.
            Any word with length longer than this will not have a cost associated
            with it.
        :returns: The computed cost table.
        """
        # Create a parent dictionary mapping each state to a set of states
        # that transition into it.
        parents = {state: set() for state in dfa.states}

        for parent_state in dfa.states:
            for symbol in dfa.alphabet:
                target_state = dfa.transitions[(parent_state, symbol)]
                parents[target_state].add(parent_state)

        # Create and initialize table for 0 length words. The cost_table entry
        # (start_state, 0) is set to a multiset containing cost of the
        # start_state. All other (state, 0) entries are set to an empty multiset.
        cost_table = {}

        for state in dfa.states:
            if state == dfa.start_state:
                cost_table[(dfa.start_state, 0)] = Counter([cost_map[dfa.start_state]])
            else:
                cost_table[(state, 0)] = Counter()

        # Use dynamic programming approach to fill table.
        for current_length in range(1, max_word_length + 1):
            for target_state in dfa.states:
                target_counter = Counter()

                # For each parent state, add to the target state's counter every possible
                # cost that reaches the parent state plus the cost to reach the target state.
                for parent_state in parents[target_state]:
                    for cost, count in cost_table[
                        (parent_state, current_length - 1)
                    ].items():
                        target_counter[cost + cost_map[target_state]] += count

                cost_table[(target_state, current_length)] = target_counter

        # Return completed cost table.
        return cost_table

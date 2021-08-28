""" Contains the AccumulatedCostDfa specification class."""

from __future__ import annotations
from typing import Optional

from numbers import Rational
from collections import Counter

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.costfunctions.cost_func import CostFunc

class AccumulatedCostDfa(CostFunc):
    """ Class encoding an Accumulated Cost Deterministic Finite Automata.
    This is represented with a Dfa that has each state mapped to a cost.
    A word has a cost if and only if it is accepted by the Dfa. The cost
    of any word that has one is the sum of all the costs associated with
    each state in its accepting path. The max length of a word must also
    be specified to ensure that the set of costs is always finite.

    :param dfa: A Dfa specification, which accepts words that have a cost.
    :param cost_map: A dictionary mapping each state in dfa ta cost.
    :param max_word_length: The maximum length of a word that has a cost.
        Any word with length longer than this will not have a cost associated
        with it.
    """
    def __init__(self, dfa: Dfa, cost_map: dict[State, Rational], max_word_length: int):
        # Store attributes
        self.dfa = dfa
        self.cost_map = {State(state):cost for (state,cost) in cost_map.items()}
        self.max_word_length = max_word_length

        # Compute the cost_table for this AccumulatedCostDfa.
        self.cost_table = self._compute_cost_table(self.dfa, self.cost_map, self.max_word_length)

        # Compute the set of costs that this cost function can assign.
        costs = set()

        for accepting_state in self.dfa.accepting_states:
            for current_length in range(0, self.max_word_length + 1):
                costs = costs | set(self.cost_table[(accepting_state, current_length)])

        super().__init__(self.dfa.alphabet, costs)

    def cost(self, word) -> Optional[Rational]:
        if self.dfa.accepts(word):
            state_path = self.dfa.get_state_path(word)

            cost = sum([self.cost_map[state] for state in state_path])

            return cost
        else:
            return None

    def decompose(self) -> dict[Rational, Dfa]:
        if len(self.costs) == 0:
            return dict()

        decomp_cost_func = dict()
        
        state_rename_map = dict()
        state_iter = 0

        cost_accepting_map = {cost:set() for cost in self.costs}

        for state in self.dfa.states:
            for cost in range(max(self.costs)+1):
                new_state = "State" + str(state_iter) + "_" + str(cost)
                state_rename_map[(state, cost)] = new_state
                
                if state in self.dfa.accepting_states and cost in self.costs:
                    cost_accepting_map[cost].add(new_state)

            state_iter += 1

        new_states = set(state_rename_map.values()) | {"Sink"}
        new_start_state = state_rename_map[(self.dfa.start_state, self.cost_map[self.dfa.start_state])]

        new_transitions = dict()

        for state in self.dfa.states:
            for cost in range(max(self.costs)+1):
                for symbol in self.dfa.alphabet:
                    dest_state = self.dfa.transitions[(state, symbol)]

                    new_origin_state = state_rename_map[(state, cost)]

                    new_cost = cost + self.cost_map[dest_state]

                    if new_cost > max(self.costs):
                        new_dest_state = "Sink"
                    else:
                        new_dest_state = state_rename_map[(dest_state, new_cost)]

                    new_transitions[(new_origin_state, symbol)] = new_dest_state 

        for symbol in self.dfa.alphabet:
            new_transitions[("Sink", symbol)] = "Sink"

        for cost in self.costs:
            new_dfa = Dfa(self.dfa.alphabet, new_states, cost_accepting_map[cost], new_start_state, new_transitions)
            decomp_cost_func[cost] = new_dfa.minimize()

        return decomp_cost_func

    @staticmethod
    def _compute_cost_table(dfa, cost_map, max_word_length):
        # Create a parent dictionary mapping each state to a set of states
        # that transition into it.
        parents = dict()
        for target_state in dfa.states:
            parents[target_state] = set()
            for parent_state in dfa.states:
                for symbol in dfa.alphabet:
                    if dfa.transitions[(parent_state, symbol)] == target_state:
                        parents[target_state].add(parent_state)

        # Create a cost table, which maps a state and a word length tuple
        # (state, cost) to a multiset of the costs of all words ending
        # in a state with a certain length.
        cost_table = dict()

        # Initialize table for 0 length words. The cost_table entry
        # (start_state, 0) is set to a multiset containing cost of the
        # start_state. All other (state, 0) entries are set to an empty multiset.
        for state in dfa.states:
            if state == dfa.start_state:
                cost_table[(state, 0)] = Counter([cost_map[dfa.start_state]])
            else:
                cost_table[(state, 0)] = Counter()

        # Augment table for each word length up to max_word_length.
        for current_length in range(1, max_word_length + 1):
            for target_state in dfa.states:
                target_counter = Counter()

                # For each parent state, add to the target state's counter every possible
                # cost that reaches the parent state plus the cost to reach the target state.
                for parent_state in parents[target_state]:
                    for cost in cost_table[(parent_state, current_length-1)].elements():
                        target_counter[cost + cost_map[target_state]] += 1

                cost_table[(target_state, current_length)] = target_counter

        # Return the cost table, which now maps (s, l) to the multiset of accumulated costs for
        # all paths of length l ending in state s.
        return cost_table

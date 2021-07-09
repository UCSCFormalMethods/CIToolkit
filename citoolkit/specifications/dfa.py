""" Contains the Dfa specification class.
    Based off the implementation by Daniel Fremont for Reactive Control Improvisation (https://github.com/dfremont/rci)
"""

from __future__ import annotations
from typing import List, Set, Dict, Tuple, Union

import random

from citoolkit.specifications.spec import Spec

class Dfa(Spec):
    """
    The Dfa class encodes a Deterministic Finite Automata specification.

    :param alphabet: The alphabet this Dfa is defined over.
    :param states: The set of all states in this Dfa.
    :param accepting_states: The set of all states in this Dfa that are accepting.
        Must be a subset of states.
    :param start_state: The start state in this Dfa. Must be a member of states.
    :param transitions: A dictionary mapping every combination of state in states
        and symbol in alphabet to a state in states. (state, symbol) -> state
    """
    def __init__(self, alphabet: Set[str], states: Set[Union[str, State]], accepting_states: Set[Union[str, State]], \
                 start_state: Union[str, State], transitions: Dict[Tuple(State, str), State]) -> None:
        # Intializes super class and stores all parameters. Also ensures
        # all states are of the State class
        super().__init__(alphabet)
        self.states = frozenset(map(State, states))
        self.accepting_states = frozenset(map(State, accepting_states))
        self.start_state = State(start_state)

        self.transitions = {(State(state),symbol):State(dest_state) for \
                            ((state, symbol),dest_state) in transitions.items()}

        # Perform checks to ensure well formed DFA.
        if not self.accepting_states.issubset(self.states):
            raise ValueError("Accepting states are not a subset of the DFA states.")

        if not self.start_state in self.states:
            raise ValueError("The starting state is not included in the DFA states")

        for symbol in self.alphabet:
            for state in self.states:
                if (state, symbol) not in self.transitions:
                    raise ValueError("The transition map is missing a transition for " + str((state, symbol)))

                if self.transitions[(state, symbol)] not in self.states:
                    raise ValueError("The transition from state '" + str(state) + "' with symbol '" + str(symbol) +\
                                     "' leads to '" + str(self.transitions[(state, symbol)]) + "', which is not a state.")

        # Initialize cache values to None
        self._topological_ordering = None
        self._accepting_path_counts = None

    ####################################################################################################
    # DFA Property Functions
    ####################################################################################################

    def accepts(self, word: List[str]) -> bool:
        """ Returns true if this Dfa accepts word, and false otherwise.
        """
        # Checks that word is composed only of symbols in the alphabet.
        for symbol in word:
            if symbol not in self.alphabet:
                raise ValueError("'" + str(word) + "' contains the symbol '" + \
                                 str(symbol) + "' which is not in the Dfa's alphabet.")

        # Checks if word accepts
        current_state = self.start_state

        for symbol in word:
            if (current_state, symbol) in self.transitions:
                current_state = self.transitions[(current_state, symbol)]
            else:
                raise ValueError("There is no transition from '" + str(current_state) + \
                  "' for the symbol '" + str(symbol) + "'.")

        return current_state in self.accepting_states

    def language_size(self) -> int:
        """ Returns the number of words accepted by this Dfa."""
        accepting_path_counts = self.compute_accepting_path_counts()

        return accepting_path_counts[self.start_state]

    def sample(self) -> Tuple[str]:
        """ Samples a word uniformly at random from the language of
        this Dfa and returns the sampled word.
        """
        # Compute or retrieve dictionary containing counts for accepting
        # paths per state.
        accepting_path_counts = self.compute_accepting_path_counts()

        # Initialize sample variables
        current_state = self.start_state
        state_count = accepting_path_counts[current_state]
        generated_word = []

        while True:
            # Select a random number in the range of all possible
            # remaining words in our language from our current stage
            remaining_count = random.randint(0, state_count-1)

            # Check if the current state is an accepting one, and
            # if so, have we finished generating words.
            if current_state in self.accepting_states:
                if remaining_count == 0:
                    # Word generation complete, return the current word
                    return tuple(generated_word)

                # Word generation is not complete, but we must adjust
                # the count to account for the possible word we are skipping.
                remaining_count -= 1

            # As word generation is not complete, we now pick the next transition.
            for symbol in self.alphabet:
                destination_state = self.transitions[(current_state, symbol)]
                destination_count = accepting_path_counts[destination_state]

                # Check our current count to see if we continue to
                # the destination state.
                if remaining_count < destination_count:
                    # Transition to destination state, and update state
                    # variables accordingly.
                    current_state = destination_state
                    state_count = destination_count
                    generated_word.append(symbol)
                    break

                # Do not transition to destination state. Update
                # remaining count and check next symbol.
                remaining_count -= destination_count

    def compute_accepting_path_counts(self) -> Dict[State, int]:
        """ Computes the number of accepting paths from a state
        to an accepting state for all states.
        """
        # Check if we have already computed accepting path counts
        if self._accepting_path_counts is not None:
            return self._accepting_path_counts
        else:
            self._accepting_path_counts = {state:0 for state in self.states}

        # Compute topological ordering to ensure that we traverse
        # Dfa in well defined order, specifically a reverse topological
        # order.
        rev_topological_ordering = self.states_topological()[::-1]

        # Compute accepting paths for each state
        for state in rev_topological_ordering:
            accepting_paths = 0

            # If state is accepting, it has an accepting path to itself.
            if state in self.accepting_states:
                accepting_paths += 1

            # Add accepting paths of states that can be reached with a transition
            for symbol in self.alphabet:
                transition_state = self.transitions[(state, symbol)]
                if transition_state in rev_topological_ordering:
                    accepting_paths += self._accepting_path_counts[transition_state]

            # Add accepting paths count to dictionary
            self._accepting_path_counts[state] = accepting_paths

        return self._accepting_path_counts

    def states_topological(self) -> List[State]:
        """ Returns a topologically sorted list of this DFA's states, excluding those
        that are unreachable or dead.
        """
        # Check if we have already computed a topological ordering
        if self._topological_ordering is not None:
            return list(self._topological_ordering)

        # Remove all unreachable or dead states. Unreachable states are automatically
        # excluded by partitioning algorithm.
        partition_sets = self.states_partition()
        feasible_sets = []

        # Copy over all non dead state sets to feasible_sets.
        for partition_set in partition_sets:
            # If a set contains an accepting state, it must be feasible. Otherwise,
            # if it can leave the set it is also feasible.
            if len(partition_set & self.accepting_states) > 0:
                feasible_sets.append(partition_set)
            else:
                # Peek a representative of partition_set
                state = partition_set.pop()
                partition_set.add(state)

                for symbol in self.alphabet:
                    if self.transitions[(state, symbol)] not in partition_set:
                        feasible_sets.append(partition_set)
                        break

        # Initialize topological ordering variables.
        topological_ordering = []
        working_states = set().union(*feasible_sets)
        states_indegree = {state:0 for state in working_states}

        # Calculate initial indegree for all states
        for state in working_states:
            for symbol in self.alphabet:
                transition_state = self.transitions[(state, symbol)]
                if transition_state in working_states:
                    states_indegree[transition_state] += 1

        # Intialize free_states queue
        free_states = []
        for state in frozenset(working_states):
            if states_indegree[state] == 0:
                free_states.append(state)
                topological_ordering.append(state)
                working_states.remove(state)

        # Continue to update indegree for states and add to free_states
        # queue until no more free states can be found.
        while len(free_states) > 0:
            # Pops next free state and updates states_indegree accordingly
            next_free_state = free_states.pop(0)

            for symbol in self.alphabet:
                transition_state = self.transitions[(next_free_state, symbol)]
                if transition_state in working_states:
                    states_indegree[transition_state] -= 1

            # Removes any states that now have indegree 0 and adds them
            # to free_states and our ordering.
            for state in frozenset(working_states):
                if states_indegree[state] == 0:
                    free_states.append(state)
                    topological_ordering.append(state)
                    working_states.remove(state)

        # Checks that we have emptied working_states. If not, we have a cycle.
        if len(working_states) > 0:
            raise ValueError("Cannot compute a topological_ordering for a cyclical DFA.")

        self._topological_ordering = tuple(topological_ordering)

        return topological_ordering

    def states_partition(self) -> List[Set[State]]:
        """ Uses Hopcroft's algorithm to partition all this DFA's states
        into equivalence classes, excluding unreachable states.
        """
        # We first remove any states that are unreachable via a breadth first search.
        current_state = None
        reachable_states = {self.start_state}
        state_queue = [self.start_state]

        while len(state_queue) > 0:
            current_state = state_queue.pop(0)

            for symbol in self.alphabet:
                next_state = self.transitions[(current_state, symbol)]
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    state_queue.append(next_state)

        reachable_accepting_states = reachable_states & self.accepting_states

        # Use Hopcroft's algorithm to merge nondistingishuable states.
        partition_sets = [reachable_states - reachable_accepting_states, reachable_accepting_states.copy()]
        working_sets = [reachable_states - reachable_accepting_states, reachable_accepting_states.copy()]

        while len(working_sets) > 0:
            working_set = working_sets.pop()

            # Iterate over each symbol so that we check all transitions.
            for symbol in self.alphabet:
                # Find all states that when transitioning with symbol
                # remain lead to the working_set.
                incoming_states = set()

                for origin_state in reachable_states:
                    dest_state = self.transitions[(origin_state, symbol)]
                    if dest_state in working_set:
                        incoming_states.add(origin_state)

                # Iterate over all partition sets to find any sets such
                # that incoming_states is a subset of that set but not
                # equal to that set.
                for partition_set in partition_sets.copy():
                    # Compute the intersection and difference of
                    # incoming_states and partition_set.
                    intersection_set = partition_set & incoming_states
                    difference_set = partition_set - incoming_states

                    if len(intersection_set) > 0 and len(difference_set) > 0:
                        # Replace partition set in partition_sets with intersection_set
                        # and difference_set.
                        partition_sets.remove(partition_set)

                        partition_sets.append(intersection_set)
                        partition_sets.append(difference_set)

                        # If partition_set is in working_sets, replace it with
                        # intersection_set and difference_set. Otherwise, add the
                        # smaller of the two to working_sets.

                        if partition_set in working_sets:
                            working_sets.remove(partition_set)

                            working_sets.append(intersection_set)
                            working_sets.append(difference_set)
                        elif len(intersection_set) <= len(difference_set):
                            working_sets.append(intersection_set)
                        else:
                            working_sets.append(difference_set)

        real_partition_sets = []

        for partition_set in partition_sets:
            if len(partition_set) != 0:
                real_partition_sets.append(partition_set)

        return real_partition_sets

    ####################################################################################################
    # DFA Modification Functions
    ####################################################################################################

    def minimize(self) -> Dfa:
        """ Computes and returns a minimal Dfa that accepts the same
            language as self.
        """
        partition_sets = self.states_partition()

        # Pop one representative from each equivalence class and
        # creates a map from each state to its representative.
        minimal_states = set()
        representative_map = dict()

        for partition_set in partition_sets:
            representative = partition_set.pop()
            representative_map[representative] = representative

            minimal_states.add(representative)

            while len(partition_set) > 0:
                representative_map[partition_set.pop()] = representative

        # Create transition map for minimal DFA.
        minimal_transitions = dict()

        for state in minimal_states:
            for symbol in self.alphabet:
                target_state = self.transitions[(state, symbol)]
                minimal_transitions[(state, symbol)] = representative_map[target_state]

        # Create minimal DFA.
        minimal_accepting_states = minimal_states & self.accepting_states
        start_state_rep = representative_map[self.start_state]

        return Dfa(self.alphabet, minimal_states, minimal_accepting_states, start_state_rep, minimal_transitions)

    def negation(self) -> Dfa:
        """ Computes and returns a Dfa that accepts words
        if and only if they are not accepted by self.
        """
        states = self.states
        new_accepting_states = states - self.accepting_states
        start_state = self.start_state
        transitions = self.transitions.copy()

        return Dfa(self.alphabet, states, new_accepting_states, start_state, transitions)

    ####################################################################################################
    # Constructor Functions
    ####################################################################################################

    @classmethod
    def union_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        """ Computes the union product construction for two DFAs and
        return its minimized form.

        :param dfa_a: The first dfa to use in the product construction.
        :param dfa_b: The second dfa to use in the product construction.
        """
        return cls._product_construction(dfa_a, dfa_b, union=True).minimize()

    @classmethod
    def intersection_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        """ Computes the union product construction for two DFAs and
        return its minimized form.

        :param dfa_a: The first dfa to use in the product construction.
        :param dfa_b: The second dfa to use in the product construction.
        """

        return cls._product_construction(dfa_a, dfa_b, union=False).minimize()

    @classmethod
    def _product_construction(cls, dfa_a: Dfa, dfa_b: Dfa, union: bool) -> Dfa:
        """ Computes the product construction for two DFAs, for either
        the union or intersection depending on the value of the union
        parameter.

        :param dfa_a: The first dfa to use in the product construction.
        :param dfa_b: The second dfa to use in the product construction.
        :param union: If true, use the union rules to decide which states
            are accepting. If false, use intersection rules.
        """
        # Performs checks to make sure that the product construction
        # is being used correctly.
        alphabet = dfa_a.alphabet | dfa_b.alphabet
        if alphabet != dfa_a.alphabet or alphabet != dfa_b.alphabet:
            raise ValueError("Cannot perform operations on specifications with different alphabets.")

        if not isinstance(dfa_a, Dfa) or not isinstance(dfa_b, Dfa):
            raise ValueError("The product construction can only be performed on DFAs.")

        # Initialize parameters for new Dfa
        new_states = set()
        new_accepting_states = set()
        new_start_state = State(dfa_a.start_state, dfa_b.start_state)
        new_transitions = dict()

        # Iterate through every combination of states in the dfa_a
        # and dfa_b, each of which becomes a new state in our product
        # dfa.
        for state_a in dfa_a.states:
            for state_b in dfa_b.states:
                # Create the new product state and add it to our set
                # of all new states.
                new_state = State(state_a, state_b)
                new_states.add(new_state)

                # Check if this new state is an accepting state, depending
                # on whether we are computing a union or intersection construction.
                if (union and (state_a in dfa_a.accepting_states or state_b in dfa_b.accepting_states)) \
                  or (state_a in dfa_a.accepting_states and state_b in dfa_b.accepting_states):
                    new_accepting_states.add(new_state)

                # Determines the transition for this new state for each symbol
                # and adds them to new_transitions.
                for symbol in alphabet:
                    new_transitions[(new_state, symbol)] = State(dfa_a.transitions[(state_a, symbol)], \
                                                       dfa_b.transitions[(state_b, symbol)])

        # Uses the above pieces to create the new product Dfa and returns it.
        return Dfa(alphabet, new_states, new_accepting_states, new_start_state, new_transitions)

    @classmethod
    def max_length_dfa(cls, alphabet, length_requirement) -> Dfa:
        """ Returns a Dfa that accepts all strings over alphabet with length at most
        length_requirement.

        :param length_requirement: The maximum length of strings that the returned DFA should accept.
        """

        if length_requirement < 0:
            raise ValueError("length_requirement must be non-negative.")

        states = {"Seen" + str(state_num) for state_num in range(length_requirement+1)} | {"Sink"}

        accepting_states = {"Seen" + str(state_num) for state_num in range(length_requirement+1)}
        start_state = "Seen0"

        transitions = {}

        for state_num in range(length_requirement):
            for symbol in alphabet:
                transitions[("Seen" + str(state_num), symbol)] = "Seen" + str(state_num+1)

        for symbol in alphabet:
            transitions[("Seen" + str(length_requirement), symbol)] = "Sink"
            transitions[("Sink", symbol)] = "Sink"

        return Dfa(alphabet, states, accepting_states, start_state, transitions)

    @classmethod
    def exact_length_dfa(cls, alphabet, length_requirement) -> Dfa:
        """ Returns a Dfa that accepts all strings over alphabet with length exactly
        length_requirement.

        :param length_requirement: The length of strings that the returned DFA should accept.
        """

        if length_requirement < 0:
            raise ValueError("length_requirement must be non-negative.")

        states = {"Seen" + str(state_num) for state_num in range(length_requirement+1)} | {"Sink"}

        accepting_states = {("Seen" + str(length_requirement))}
        start_state = "Seen0"

        transitions = {}

        for state_num in range(length_requirement):
            for symbol in alphabet:
                transitions[("Seen" + str(state_num), symbol)] = "Seen" + str(state_num+1)

        for symbol in alphabet:
            transitions[("Seen" + str(length_requirement), symbol)] = "Sink"
            transitions[("Sink", symbol)] = "Sink"

        return Dfa(alphabet, states, accepting_states, start_state, transitions)

class State:
    """ Class representing a state in a DFA. Primarily used for merging states
    and pretty printing them.

    :param *args: A sequence of strings or other State objects that will be
        merged into a new state.
    """
    def __init__(self, *args: Union[str, State]):
        # Parses
        labels = []

        for arg in args:
            if isinstance(arg, str):
                labels.append(arg)
            elif isinstance(arg, State):
                labels += list(arg.state_tuple)
            else:
                raise ValueError("Only strings or State objects can be used to create another state.")

        self.state_tuple = tuple(labels)

    def __str__(self):
        if len(self.state_tuple) == 1:
            return "(" + self.state_tuple[0] + ")"
        else:
            return str(self.state_tuple)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (isinstance(other, State) and (self.state_tuple == other.state_tuple)) or \
               (isinstance(other, str) and (len(self.state_tuple) == 1) and (other == self.state_tuple[0]))


    def __hash__(self):
        return hash(self.state_tuple)

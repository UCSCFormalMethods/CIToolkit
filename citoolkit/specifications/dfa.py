""" Contains the Dfa specification class.
    Based off the implementation by Daniel Fremont for Reactive Control Improvisation (https://github.com/dfremont/rci)
"""

from typing import List, Set, Dict, Tuple, Union

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
        # Perform checks to ensure well formed DFA.
        if not accepting_states.issubset(states):
            raise ValueError("Accepting states are not a subset of the DFA states.")

        if not start_state in states:
            raise ValueError("The starting state is not included in the DFA states")

        # Intializes super class and stores all parameters. Also ensures
        # all states are of the State class
        super().__init__(alphabet)
        self.states = set(map(State, states))
        self.accepting_states = set(map(State, accepting_states))
        self.start_state = State(start_state)

        self.transitions = {(State(state),symbol):State(dest_state) for ((state, symbol),dest_state) in transitions.items()}

    def accepts(self, word: List[str]) -> bool:
        current_state = self.start_state

        for symbol in word:
            if (current_state, symbol) in self.transitions:
                current_state = self.transitions[(current_state, symbol)]
            else:
                raise ValueError("There is no transition from '" + str(current_state) + \
                  "' for the symbol '" + str(symbol) + "'.")

        return current_state in self.accepting_states

    def minimize(self) -> "Dfa":
        """ Computes and returns a minimal Dfa that accepts the same
            language as self.
        """
        # We first remove any states that are unreachable via a breadth
        # first search.
        current_state = None
        reachable_states = set(self.start_state)
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
        partition_sets = set(reachable_states, reachable_accepting_states)
        working_sets = set(reachable_states, reachable_accepting_states)

        while len(working_sets) > 0:
            working_set = working_sets.pop()

            # Iterate over each symbol so that we check all transitions.
            for symbol in self.alphabet:
                # Find all states that when transitioning with symbol
                # remain lead to the working_set.
                incoming_states = set()

                for origin_state in reachable_accepting_states:
                    dest_state = self.transitions[(origin_state, symbol)]
                    if dest_state in working_set:
                        incoming_states.add(dest_state)

                # Iterate over all partition sets to find any sets such
                # that incoming_states is a subset of that set but not
                # equal to that set.
                for partition_set in partition_sets:
                    # Compute the intersection and difference of
                    # incoming_states and partition_set.
                    intersection_set = partition_set & incoming_states
                    difference_set = partition_set - incoming_states

                    if len(intersection_set) > 0 and difference_set > 0:
                        # Replace partition set in partition_sets with intersection_set
                        # and difference_set.
                        partition_sets.remove(partition_set)

                        partition_sets.add(intersection_set)
                        partition_sets.add(difference_set)

                        # If partition_set is in working_sets, replace it with
                        # intersection_set and difference_set. Otherwise, add the
                        # smaller of the two to working_sets.

                        if partition_set in working_sets:
                            working_sets.remove(partition_set)

                            working_sets.add(intersection_set)
                            working_sets.add(difference_set)
                        elif len(intersection_set) <= difference_set:
                            working_sets.add(intersection_set)
                        else:
                            working_sets.add(difference_set)

        # Pop one representative from each equivalence class and
        # creates a map from each state to its representative.
        minimal_states = set()
        representative_map = dict()

        for partition_set in partition_sets:
            representative = partition_set.pop()

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
        minimal_accepting_states = minimal_states & reachable_accepting_states
        start_state_rep = representative_map[self.start_state]

        return Dfa(self.alphabet, minimal_states, minimal_accepting_states, start_state_rep, minimal_transitions)


    @classmethod
    def dfa_union_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        """ Computes the union product construction for two DFAs and
        return its minimized form.

        :param dfa_a: The first dfa to use in the product construction.
        :param dfa_b: The second dfa to use in the product construction.
        """

        return cls._dfa_product_construction(dfa_a, dfa_b, union=True).minimize()

    @classmethod
    def dfa_intersection_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        """ Computes the union product construction for two DFAs and
        return its minimized form.

        :param dfa_a: The first dfa to use in the product construction.
        :param dfa_b: The second dfa to use in the product construction.
        """

        return cls._dfa_product_construction(dfa_a, dfa_b, union=False).minimize()

    @classmethod
    def _dfa_product_construction(cls, dfa_a: Dfa, dfa_b: Dfa, union: bool) -> Dfa:
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
        new_starting_state = State(dfa_a.start_state, dfa_b.start_state)
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
                    new_transitions[new_state] = State(dfa_a.transitions[(state_a, symbol)], \
                                                       dfa_b.transitions[(state_b, symbol)])

        # Uses the above pieces to create the new product Dfa and returns it.
        return Dfa(alphabet, new_states, new_accepting_states, new_starting_state, new_transitions)

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
                labels.append(str)
            elif isinstance(arg, State):
                labels += arg.labels
            else:
                raise ValueError("Only strings or State objects can be used to create another state.")

        self.state_tuple = tuple(labels)

    def __str__(self):
        if len(self.state_tuple) == 1:
            return "(" + self.state_tuple[0] + ")"
        else:
            return str(self.state_tuple)

    def __eq__(self, other):
        return isinstance(other, State) and (self.state_tuple == other.state_tuple)


    def __hash__(self):
        return hash(self.state_tuple)

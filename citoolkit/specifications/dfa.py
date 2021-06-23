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
        raise NotImplementedError()

    @classmethod
    def dfa_union_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        cls._dfa_product_construction(dfa_a, dfa_b, union=True)

    @classmethod
    def dfa_intersection_construction(cls, dfa_a: Dfa, dfa_b: Dfa) -> Dfa:
        cls._dfa_product_construction(dfa_a, dfa_b, union=False)

    @classmethod
    def _dfa_product_construction(cls, dfa_a: Dfa, dfa_b: Dfa, union: bool) -> Dfa:
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
        return str(self.state_tuple)

    def __eq__(self, other):
        return isinstance(other, State) and (self.state_tuple == other.state_tuple)

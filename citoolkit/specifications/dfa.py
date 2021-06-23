"""Contains the Dfa specification class."""

from typing import Set, Dict, Tuple

from citoolkit.specifications.spec import Spec

class Dfa(Spec):
    """
    The Dfa class encodes a Deterministic Finite Automata specification.
    """
    def __init__(self, alphabet: Set[str], states: Set[str], accepting_states: Set[str], start_state: str, transitions: Dict[Tuple(str, str), str],) -> None:
        # TODO: Add checks to ensure well formed DFA.

        # Intializes super class and stores all parameters
        super().__init__(alphabet)
        self.states = states
        self.accepting_states = accepting_states
        self.start_state = start_state

        self.transitions = transitions

    def accepts(self, word: str) -> bool:
        current_state = self.start_state

        for symbol in word:
            current_state = self.transitions[(current_state, symbol)]

        return current_state in self.accepting_states

    def language_size(self):
        """ Computes the number of strings accepted by this specification."""
        raise NotImplementedError()

""" Contains the LabellingDfa class"""

from __future__ import annotations
from typing import Optional

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.labellingfunctions.labelling_func import LabellingFunc

class LabellingDfa(LabellingFunc):
    """ Class encoding a Labelling Deterministic Finite Automata.
    This is represented with a Dfa that has each accepting state
    mapped to a label. A word is labelled if it is accepted by the
    Dfa and the label associated with that word is the one that is
    associated with that accepting state.

    :param dfa: A Dfa specification, which accepts words that have a label.
    :param label_map: A dictionary mapping each accepting state in dfa to
        a label.
    :raises ValueError: Raised if an accepting state in dfa is missing an
        associated label in label_map.
    """
    def __init__(self, dfa: Dfa, label_map: dict[State, str]):
        # Initialize super class and stores attributes.
        self.dfa = dfa
        self.label_map = {State(state):label for (state,label) in label_map.items()}
        labels = frozenset(self.label_map.values())

        super().__init__(self.dfa.alphabet, labels)

        # Perform checks to ensure a well formed Labelling Dfa.
        if not set(self.label_map.keys()) == self.dfa.accepting_states:
            for target_state in label_map.keys():
                if target_state not in self.dfa.accepting_states:
                    raise ValueError("The accepting state '" + target_state + "' is missing an associated label in label_map")

        for label in labels:
            if not isinstance(label, str):
                print("'" + str(label) + "' is not a string, and therefore cannot be a label.")

        # Initialize cache values to None
        self.decomp_labelling_func = None

    ####################################################################################################
    # LabellingFunc Functions
    ####################################################################################################

    def label(self, word) -> Optional[str]:
        """ Returns the appropriate label for a word. This label is
        found by first checking if the interior Dfa accepts a word.
        If it does, then the accepting state that the Dfa terminates
        in is mapped through label_map to determine the label for
        that word. If the word has no label, i.e. it is not accepted
        by this Dfa, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The label associated with this word.
        """
        if self.dfa.accepts(word):
            return self.label_map[self.dfa.get_terminal_state(word)]

        return None

    def decompose(self) -> dict[str, Dfa]:
        """ Decomposes this labelling function into a Dfa indicator
        function for each label.

        :returns: A dictionary mapping each label to a Dfa Spec that
            accepts if and only if a word has that label.
        """
        # Check if value is cached.
        if self.decomp_labelling_func is not None:
            return self.decomp_labelling_func

        # Compute decomposed labelling function
        self.decomp_labelling_func = {}

        # Compute a mapping from each label to its associated accepting states.
        label_states_mapping = {label:set() for label in self.labels}

        for accepting_state in self.dfa.accepting_states:
            label = self.label_map[accepting_state]
            label_states_mapping[label].add(accepting_state)

        # Compute the indicator function for each label
        for label in self.labels:
            indicator_dfa = Dfa(self.dfa.alphabet, self.dfa.states, label_states_mapping[label], \
                                self.dfa.start_state, self.dfa.transitions)
            self.decomp_labelling_func[label] = indicator_dfa.minimize()

        return self.decomp_labelling_func

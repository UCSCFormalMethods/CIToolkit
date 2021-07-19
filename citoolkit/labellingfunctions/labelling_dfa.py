""" Contains the LabellingDfa class"""

from __future__ import annotations
from typing import Union

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.labellingfunctions.labelling_func import LabellingFunc

class LabellingDfa(LabellingFunc):
    """ Class encoding a Labelling Deterministic Finite Automata.
    This is represented with a Dfa that has each accepting state
    mapped to a label. A word is labelled if it is accepted by the
    Dfa and the label associated with that word is the one that is
    associated with that accepting state.
    """
    def __init__(self, alphabet: set[str], states: set[Union[str, State]], accepting_states: set[Union[str, State]], \
                 start_state: Union[str, State], transitions: dict[tuple[State, str], State], \
                 labels: set[str], label_map: dict[State, str]):
        super().__init__(alphabet, labels)

    def label_word(self, word) -> str:
        """ Returns the appropriate label for a word. This label is
        found by first checking if the interior Dfa accepts a word.
        If it does, then the accepting state that the Dfa terminates
        in is mapped through label_map to determine the label for
        that word. If the word has no label, i.e. it is not accepted
        by this Dfa, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The label associated with this word.
        """
        raise NotImplementedError()

    def decompose(self) -> dict[str, Dfa]:
        """ Decomposes this labelling function into a Dfa indicator
        function for each label.

        :returns: A dictionary mapping each label to a Dfa Spec that
        accepts if and only if a word has that label.

        """
        raise NotImplementedError()

    @staticmethod
    def recompose(decomp_labelling_func) -> LabellingFunc:
        """ Takes a decomposed LabellingDfa and uses it to
        reconstruct the associated LabellingDfa.

        :param decomp_labelling_func: A dictionary mapping each label
            to a Dfa object that accepts only words labelled with that
            label.
        :returns: A LabellingDfa that maps each word accepted
            by one of the Dfas in decomp_labelling_func to
            the label that maps to that Dfa.
        """

""" Contains the LabelFunc class, from which all labelling functions should inherit."""

from __future__ import annotations
from typing import Optional

from abc import ABC, abstractmethod

from citoolkit.specifications.spec import Spec

class LabellingFunc(ABC):
    """ The LabelFunc class is a parent class to all labelling functions.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: set[str], labels: set[str]) -> None:
        self.alphabet = frozenset(alphabet)
        self.labels = frozenset(labels)

    @abstractmethod
    def label(self, word: tuple[str, ...]) -> Optional[str]:
        """ Returns the appropriate label for a word. If the word
        has no label, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The label associated with this word.
        """

    @abstractmethod
    def decompose(self) -> dict[str, Spec]:
        """ Decompose this labelling function into a Spec object for
        each label that accepts only on words with that label.

        :returns: A dictionary mapping each label to a Spec object that
            accepts only words labelled with that label by this labelling function.
        """

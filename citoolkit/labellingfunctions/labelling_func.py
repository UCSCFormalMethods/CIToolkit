""" Contains the LabelFunc class, from which all labelling functions should inherit."""

from __future__ import annotations
from typing import Optional

from abc import ABC, abstractmethod

from citoolkit.specifications.spec import ExactSpec, ApproxSpec, UniverseSpec

class ExactLabellingFunc(ABC):
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
    def decompose(self) -> dict[str, ExactSpec]:
        """ Decompose this labelling function into an ExactSpec object for
        each label that accepts only on words with that label.

        :returns: A dictionary mapping each label to an ExactSpec object that
            accepts only words labelled with that label by this labelling function.
        """

class ApproxLabelFunc(ABC):
    """ The ApproximateCostFunc class is a parent class to all approximate cost functions.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: set[str], labels) -> None:
        self.alphabet = frozenset(alphabet)
        self.labels = labels

    @abstractmethod
    def realize(self, label) -> ApproxSpec:
        """ Realize this cost function into an ApproximateSpec object that accepts
        only words with cost in the range [min_cost, max_cost].

        :param min_cost: The minimum cost accepted by the realized cost function.
        :param max_cost: The maximum cost accepted by the realized cost function.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """

class TrivialLabellingFunc(ExactLabellingFunc):
    """ The TrivialLabelFunc class assigns the label "TrivialLabel" to every string. """
    def __init__(self) -> None:
        super().__init__(alphabet=frozenset(), labels=frozenset(["TrivialLabel"]))

    def label(self, word: tuple[str, ...]) -> Optional[str]:
        """ Returns the appropriate label for a word. If the word
        has no label, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The label associated with this word.
        """
        return "TrivialLabel"

    def decompose(self) -> dict[str, ExactSpec]:
        """ Decompose this labelling function into an ExactSpec object for
        each label that accepts only on words with that label.

        :returns: A dictionary mapping each label to an ExactSpec object that
            accepts only words labelled with that label by this labelling function.
        """
        return {"TrivialLabel": UniverseSpec()}

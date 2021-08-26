""" Contains the CostFunc class, from which all cost functions should inherit."""

from __future__ import annotations
from typing import Optional

from numbers import Rational

from abc import ABC, abstractmethod

from citoolkit.specifications.spec import Spec

class CostFunc(ABC):
    """ The CostFunc class is a parent class to all cost functions.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: set[str], costs: set[float]) -> None:
        self.alphabet = frozenset(alphabet)
        self.costs = frozenset(costs)

        for cost in costs:
            if not isinstance(cost, Rational):
                raise ValueError("'" + str(cost) + "' is not of the type Rational, and therefore cannot be a cost." +\
                                 " Consider constructing one using the 'fractions' library.")
            if cost < 0:
                raise ValueError("'" + str(cost) + "' is less than zero, and therefore cannot be a cost.")

    @abstractmethod
    def cost(self, word: tuple[str, ...]) -> Optional[float]:
        """ Returns the appropriate cost for a word. If the word
        has no cost, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The cost associated with this word.
        """

    @abstractmethod
    def decompose(self) -> dict[str, Spec]:
        """ Decompose this cost function into a Spec object for
        each cost that accepts only on words with that cost.

        :returns: A dictionary mapping each cost to a Spec object that
            accepts only words assigned that cost by this cost function.
        """

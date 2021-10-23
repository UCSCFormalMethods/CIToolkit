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
    def __init__(self, alphabet: set[str], costs: set[Rational]) -> None:
        self.alphabet = frozenset(alphabet)
        self.costs = frozenset(costs)

        for cost in costs:
            if not isinstance(cost, Rational):
                raise ValueError("'" + str(cost) + "' is not of the type Rational, and therefore cannot be a cost." +\
                                 " Consider constructing one using the 'fractions' library.")
            if cost < 0:
                raise ValueError("'" + str(cost) + "' is less than zero, and therefore cannot be a cost.")

    @abstractmethod
    def cost(self, word: tuple[str, ...]) -> Optional[Rational]:
        """ Returns the appropriate cost for a word. If the word
        has no cost, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The cost associated with this word.
        """

    @abstractmethod
    def decompose(self) -> dict[Rational, Spec]:
        """ Decompose this cost function into a Spec object for
        each cost that accepts only on words with that cost.

        :returns: A dictionary mapping each cost to a Spec object that
            accepts only words assigned that cost by this cost function.
        """

class SoftConstraintCostFunc(CostFunc):
    """ The SoftConstraintCostFunc class takes in a soft constraint Spec and
    produces an equivalent cost function for use in Quantitative CI. The new
    cost function assigns all words that are accepted by that Spec cost 1.
    All other words are assigned cost 0 (Note that the Spec must support the
    negation operation).

    :param soft_constraint: The soft constraint for which an equivalent cost
        function will be constructed.
    """
    def __init__(self, soft_constraint: Spec) -> None:
        self.soft_constraint = soft_constraint

        super().__init__(alphabet=frozenset(), costs=frozenset([0,1]))

    def cost(self, word: tuple[str, ...]) -> Optional[Rational]:
        """ Returns the appropriate cost for a word. If the word
        has no cost, returns None.

        :param word: A word over this cost function's alphabet.
        :returns: The cost associated with this word.
        """
        if self.soft_constraint.accepts(word):
            return 0
        else:
            return 1

    def decompose(self) -> dict[Rational, Spec]:
        """ Decompose this cost function into a Spec object for
        each cost that accepts only on words with that cost.

        :returns: A dictionary mapping each cost to a Spec object that
            accepts only words assigned that cost by this cost function.
        """
        decomposed_cost_func = {}

        decomposed_cost_func[0] = self.soft_constraint
        decomposed_cost_func[1] = ~self.soft_constraint

        return decomposed_cost_func

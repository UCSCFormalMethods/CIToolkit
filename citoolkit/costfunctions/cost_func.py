""" Contains the ExactCostFunc and ApproxCostFunc class, from which all cost functions should inherit."""

from __future__ import annotations

from numbers import Rational
from abc import ABC, abstractmethod

from citoolkit.specifications.spec import ExactSpec, ApproxSpec, Alphabet

class ExactCostFunc(ABC):
    """ The CostFunc class is a parent class to all cost functions.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: set[str], costs: set[Rational]) -> None:
        self.alphabet = Alphabet.create_alphabet(alphabet)
        self.costs = frozenset(costs)

        for cost in costs:
            if not isinstance(cost, Rational):
                raise ValueError("'" + str(cost) + "' is not of the type Rational, and therefore cannot be a cost." +\
                                 " Consider constructing one using the 'fractions' library.")
            if cost < 0:
                raise ValueError("'" + str(cost) + "' is less than zero, and therefore cannot be a cost.")

    @abstractmethod
    def cost(self, word: tuple[str, ...]) -> Rational | None:
        """ Returns the appropriate cost for a word. If the word
        has no cost, returns None.

        :param word: A word over this labelling function's alphabet.
        :returns: The cost associated with this word.
        """

    @abstractmethod
    def decompose(self, num_threads) -> dict[Rational, ExactSpec]:
        """ Decompose this cost function into an ExactSpec object for
        each cost that accepts only on words with that cost.

        :returns: A dictionary mapping each cost to an ExactSpec object that
            accepts only words assigned that cost by this cost function.
        """

class ApproxCostFunc(ABC):
    """ The ApproximateCostFunc class is a parent class to all approximate cost functions.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: set[str]) -> None:
        self.alphabet = Alphabet.create_alphabet(alphabet)

    @abstractmethod
    def realize(self, min_cost, max_cost) -> ApproxSpec:
        """ Realize this cost function into an ApproximateSpec object that accepts
        only words with cost in the range [min_cost, max_cost].

        :param min_cost: The minimum cost accepted by the realized cost function.
        :param max_cost: The maximum cost accepted by the realized cost function.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """

class SoftConstraintCostFunc(ExactCostFunc):
    """ The SoftConstraintCostFunc class takes in a soft constraint ExactSpec and
    produces an equivalent cost function for use in Quantitative CI. The new
    cost function assigns all words that are accepted by that ExactSpec cost 1.
    All other words are assigned cost 0 (Note that the ExactSpec must support the
    negation operation).

    :param soft_constraint: The soft constraint for which an equivalent cost
        function will be constructed.
    """
    def __init__(self, soft_constraint: ExactSpec) -> None:
        self.soft_constraint = soft_constraint

        super().__init__(alphabet=frozenset(), costs=frozenset([0,1]))

    def cost(self, word: tuple[str, ...]) -> Rational | None:
        """ Returns the appropriate cost for a word. If the word
        has no cost, returns None.

        :param word: A word over this cost function's alphabet.
        :returns: The cost associated with this word.
        """
        if self.soft_constraint.accepts(word):
            return 0
        else:
            return 1

    def decompose(self, num_threads=1) -> dict[Rational, ExactSpec]:
        """ Decompose this cost function into a ExactSpec object for
        each cost that accepts only on words with that cost.

        :returns: A dictionary mapping each cost to a ExactSpec object that
            accepts only words assigned that cost by this cost function.
        """

        decomposed_cost_func = {}

        decomposed_cost_func[0] = self.soft_constraint
        decomposed_cost_func[1] = ~self.soft_constraint

        return decomposed_cost_func

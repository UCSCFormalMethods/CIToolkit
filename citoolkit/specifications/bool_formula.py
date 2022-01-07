""" Contains the BoolFormula approximate specification class."""

from __future__ import annotations
from collections.abc import Iterable

import pyapproxmc
import pyunigen

from citoolkit.specifications.spec import ApproxSpec


class BoolFormula(ApproxSpec):
    """ The BoolFormula class encodes a boolean formula specification in CNF form.

    :param clauses: An iterable of CNF clause. Each CNF clause should be composed
        of one or more nonzero integers, following the DIMACS format. Clauses should
        not be terminated with 0.
    :param main_vars: If provided, only variables in this iterator will be counted
        and sampled over. If not provided or the iterator is empty, all variables
        will be counted and sampled over.
    """
    def __init__(self, clauses, main_vars = None):
        # Initialize superclass
        super().__init__([0,1])

        # Check that parameters are well formed.
        variables = set()
        for clause in clauses:
            for literal in clause:
                if not isinstance(literal, int):
                    raise ValueError("All literals must be integers.")
                if literal == 0:
                    raise ValueError("All literals must be nonzero.")

                variables.add(abs(literal))

        if main_vars is None:
            main_vars = set()
        elif isinstance(main_vars, Iterable):
            main_vars = frozenset(main_vars)
        else:
            raise ValueError("The main_vars parameter must be None or an Iterable.")

        if not main_vars.issubset(variables):
            raise ValueError("The main_vars parameter contains variables not in clauses.")

        # Store class attributes.
        self.clauses = [tuple(clause) for clause in clauses]
        self.main_vars = main_vars
        self.variables = frozenset(variables)

        # Initialize cache values to None
        self._counts = None

    def accepts(self, word) -> bool:
        raise NotImplementedError()

    def language_size(self, tolerance=0.8, confidence=0.2, seed=1, min_length: int=None, max_length: int=None) -> int:
        """ Approximately computes the number of solutions to this formula.
            With probability 1 - confidence, the following holds true,
            true_count*(1 + confidence)^-1 <= returned_count <= true_count*(1 + confidence)

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param min_length: Not applicable to boolean formula so ignored.
        :param max_length: Not applicable to boolean formula so ignored.
        :returns: The approximate number of solutions to this formula.
        """
        # Check if count has already been computed
        if self._counts is not None:
            return self._counts

        # Create a new ApproxMC counter and approximately count the number of solutions.
        counter = pyapproxmc.Counter(epsilon=tolerance, delta=confidence, sampling_set=self.main_vars, seed=seed)

        for clause in self.clauses:
            counter.add_clause(clause)

        counts = counter.count()

        # Cache counts and return total count.
        self._counts = counts

        return counts[0] * 2**counts[1]


    def sample(self, tolerance=15, seed=1 , min_length: int=None, max_length: int=None) -> tuple[int,...]:
        """ Generate a solution to this boolean formula approximately uniformly.
            Let true_prob be 1/true_count and returned_prob be the probability of sampling
            any particular solution. With probability 1 - confidence, the following holds true,
            1/(1 + tolerance) * true_prob <= returned_prob <= (1 + tolerance) / true_prob

            language_size() must be called before sample().

        :param tolerance: The tolerance of the count.
        :param min_length: Not applicable to boolean formula so ignored.
        :param max_length: Not applicable to boolean formula so ignored.
        :returns: An approximately uniformly sampled solution to this formula.
        """
        # Check that tolerance isn't too small for UniGen and that count has already
        # been computed.
        if tolerance < 6.84:
            raise ValueError("The almost uniform sampling library used (UniGen) does not currently support a tolerance of less than 6.84.")

        if self._counts is None:
            raise RuntimeError("You must call language_size with an appropriate tolerance and confidence before calling sample.")

        # Find an appropriate kappa value.
        kappa = BoolFormula._find_kappa(tolerance)

        # Create a new Unigen sampler and approximately uniformly sample a solution.
        sampler = pyunigen.Sampler(kappa=kappa, sampling_set=self.main_vars, seed=seed)

        for clause in self.clauses:
            sampler.add_clause(clause)

        sample = tuple(sampler.sample(*self._counts))

        return sample

    @staticmethod
    def _find_kappa(tolerance):
        """ Finds a kappa value that is in the range (0.99*tolerance, tolerance] using the bisection method.

        :param tolerance: The desired tolerance for the uniform sample.
        :returns: A kappa that is equivalent to an epsilon in the range (0.99*tolerance, tolerance]
        """
        tol_func = lambda k: (1 + k)*(7.44 + (0.392)/((1-k)**2)) - 1

        lower_bound, upper_bound = (0,1)

        while True:
            curr_kappa = (upper_bound-lower_bound)/2 + lower_bound

            curr_tol = tol_func(curr_kappa)

            if 0.99*tolerance < curr_tol <= tolerance:
                return curr_kappa

            if curr_tol > tolerance:
                upper_bound = curr_kappa
            else:
                lower_bound = curr_kappa

class UnsatBoolFormula(BoolFormula):
    """ A class representing a minimal unsatisfiable boolean formula."""
    def __init__(self):
        super().__init__(clauses=[])

    def accepts(self, word):
        return False

    def language_size(self, tolerance, confidence, min_length: int = None, max_length: int = None) -> int:
        return 0

    def sample(self, tolerance, confidence, min_length: int = None, max_length: int = None) -> tuple[int,...]:
        raise ValueError("Cannot sample from an unsatisfiable boolean formula.")
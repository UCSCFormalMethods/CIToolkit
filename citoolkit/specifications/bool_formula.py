""" Contains the BoolFormula approximate specification class."""

from __future__ import annotations
from collections.abc import Iterable

import random

import pyapproxmc
import pyunigen

from citoolkit.specifications.spec import ApproxSpec, Alphabet


class BoolFormula(ApproxSpec):
    """The BoolFormula class encodes a boolean formula specification in CNF form.

    :param clauses: An iterable of CNF clause. Each CNF clause should be composed
        of one or more nonzero integers, following the DIMACS format. Clauses should
        not be terminated with 0.
    :param main_vars: If provided, only variables in this iterator will be counted
        and sampled over. If not provided or the iterator is empty, all variables
        will be counted and sampled over.
    :raises ValueError: Raised if an input is malformed.
    """

    def __init__(
        self, clauses: Iterable[Iterable[int]], main_vars: Iterable[int] = None
    ):
        # Initialize superclass
        super().__init__(BoolFormulaAlphabet())

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
            raise ValueError(
                "The main_vars parameter contains variables not in clauses."
            )

        # Store class attributes.
        self.clauses = [tuple(clause) for clause in clauses]
        self.main_vars = main_vars
        self.variables = frozenset(variables)

        # Initialize cache values to None
        self._counts = None

    ####################################################################################################
    # ApproxSpec Functions
    ####################################################################################################

    def accepts(self, word: dict[int, bool]) -> bool:
        raise NotImplementedError()

    def language_size(
        self, tolerance: float = 0.8, confidence: float = 0.2, seed: int = None
    ) -> int:
        """Approximately computes the number of solutions to this formula.
        With probability 1 - confidence, the following holds true,
        true_count*(1 + tolerance)^-1 <= returned_count <= true_count*(1 + tolerance)

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param seed: The randomized seed. By default this is equal to None, which means the
            internal random state will be used.
        :returns: The approximate number of solutions to this formula.
        """
        # Compute count if not already computed
        if self._counts is None:
            # If seed is None, get random seed from internal random state.
            if seed is None:
                seed = random.getrandbits(32)

            # Create a new ApproxMC counter and approximately count the number of solutions.
            counter = pyapproxmc.Counter(
                epsilon=tolerance,
                delta=confidence,
                sampling_set=self.main_vars,
                seed=seed,
            )

            for clause in self.clauses:
                counter.add_clause(clause)

            self._counts = counter.count()

        return self._counts[0] * 2 ** self._counts[1]

    def sample(self, tolerance: float = 15, seed: int = None) -> dict[int, bool]:
        """Generate a solution to this boolean formula almost uniformly.
        Let true_prob be 1/true_count and returned_prob be the probability of sampling
        any particular solution. With probability 1 - confidence, the following holds true,
        1/(1 + tolerance) * true_prob <= returned_prob <= (1 + tolerance) / true_prob

        NOTE: language_size() must be called before sample(), as the confidence of sampling
        depends on the confidence of the count.

        :param tolerance: The tolerance of the count. Due to limitations of Unigen this must
            be greater than or equal to 6.84.
        :param seed: The randomized seed. By default this is equal to None, which means the
            internal random state will be used.
        :raises ValueError: Raised if too low a tolerance is requested.
        :raises RuntimeError: Raised if tolerance is too low or if language_size has
            not already been called for this class.
        :returns: An approximately uniformly sampled solution to this formula.
        """
        # If seed is None, get random seed from internal random state.
        if seed is None:
            seed = random.getrandbits(32)

        # Check that tolerance isn't too small for UniGen and that count has already
        # been computed.
        if tolerance < 6.84:
            raise ValueError(
                "The almost uniform sampling library used (UniGen) does not currently"
                " support a tolerance of less than 6.84."
            )

        if self._counts is None:
            raise RuntimeError(
                "You must call language_size with an appropriate tolerance and"
                " confidence before calling sample."
            )

        # Find an appropriate kappa value.
        kappa = BoolFormula._find_kappa(tolerance)

        # Create a new Unigen sampler and approximately uniformly sample a solution.
        sampler = pyunigen.Sampler(kappa=kappa, sampling_set=self.main_vars, seed=seed)

        for clause in self.clauses:
            sampler.add_clause(clause)

        sample = tuple(sampler.sample(*self._counts))

        return sample

    ####################################################################################################
    # Helper Functions
    ####################################################################################################

    @staticmethod
    def _find_kappa(tolerance: float) -> float:
        """Finds a kappa value that is in the range (0.99*tolerance, tolerance] using the bisection method.

        :param tolerance: The desired tolerance for the uniform sample.
        :returns: A kappa that is equivalent to an epsilon in the range (0.99*tolerance, tolerance]
        """
        tol_func = lambda k: (1 + k) * (7.44 + (0.392) / ((1 - k) ** 2)) - 1

        lower_bound, upper_bound = (0, 1)

        while True:
            curr_kappa = (upper_bound - lower_bound) / 2 + lower_bound

            curr_tol = tol_func(curr_kappa)

            if 0.99 * tolerance < curr_tol <= tolerance:
                return curr_kappa

            if curr_tol > tolerance:
                upper_bound = curr_kappa
            else:
                lower_bound = curr_kappa


class UnsatBoolFormula(BoolFormula):
    """A class representing a minimal unsatisfiable boolean formula."""

    def __init__(self):
        super().__init__(clauses=[])

    def accepts(self, word: Any):
        """An UnsatBoolFormula never accepts.

        :param word: The word which is checked for acceptance.
        :returns: False
        """
        return False

    def language_size(self, tolerance: float, confidence: float, seed: int) -> int:
        """An UnsatBoolFormula has an empty language.

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param seed: The randomized seed.
        :returns: 0
        """
        return 0

    def sample(self, tolerance: float, seed: int) -> dict[int, bool]:
        """An UnsatBoolFormula cannot be sampled from.

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param seed: The randomized seed.
        :raises ValueError: Raised as the UnsatBoolFormula cannot be
            sampled from.
        :returns: None
        """
        raise ValueError("Cannot sample from an unsatisfiable boolean formula.")


class BoolFormulaAlphabet(Alphabet):
    """Alphabet class representing the abstract alphabet of CNF formulas,
    which is a mapping from each variable number to a truth assignment.
    """

    def __eq__(self, other: Any):
        if isinstance(other, BoolFormulaAlphabet):
            return True

        return NotImplemented

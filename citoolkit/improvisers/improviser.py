""" Contains the Improviser class, from which all CI improvisers should inherit."""

from __future__ import annotations
from typing import Iterator

from abc import ABC, abstractmethod

class Improviser(ABC):
    """ The Improviser class is a parent class to all CI improvisers. """

    @abstractmethod
    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """

    def generator(self) -> Iterator[tuple[str,...]]:
        """ Create a generator that continually improvises words.

        :returns: An iterable that will indefinitely improvise words.
        """
        while True:
            yield self.improvise()

class InfeasibleImproviserError(Exception):
    """ An exception raised when an improvisation problem is infeasible."""

class InfeasibleCostError(InfeasibleImproviserError):
    """ An exception raised when there is no improvising distribution that meets
    the required cost bound.

    :param best_cost: The lowest cost achievable.
    """
    def __init__(self, msg, best_cost):
        super().__init__(msg)
        self.best_cost = best_cost

class InfeasibleSoftConstraintError(InfeasibleImproviserError):
    """ An exception raised when there is no improvising distribution that satisfies
    the soft constraint requirement.

    :param best_prob: The maximum soft constraint probability achievable.
    """
    def __init__(self, msg, best_prob) -> None:
        super().__init__(msg)
        self.best_prob = best_prob

class InfeasibleRandomnessError(InfeasibleImproviserError):
    """ An exception that is raised when there is no improvising distribution that
    satisfies the randomness requirement(s).

    :param set_size: The size of the set which lacks sufficient randomness.
    """
    def __init__(self, msg, set_size) -> None:
        super().__init__(msg)
        self.set_size = set_size

class InfeasibleLabelRandomnessError(InfeasibleRandomnessError):
    """ An exception that is raised when there is no improvising distribution that
    satisfies the randomness over labels requirement.
    """

class InfeasibleWordRandomnessError(InfeasibleRandomnessError):
    """ An exception that is raised when there is no improvising distribution that
    satisfies the randomness over words requirement.
    """

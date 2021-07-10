"""Contains the Improviser class, from which all CI improvisers should inherit."""

from __future__ import annotations
from typing import Tuple, Iterator

class Improviser:
    """ The Improviser class is a parent class to all CI improvisers. """

    def improvise(self) -> Tuple[str]:
        """ Improvise a single word."""
        raise NotImplementedError(self.__class__.__name__ + " has not implemented 'generate'.")

    def generator(self) -> Iterator[Tuple[str]]:
        """ Returns an iterable that will indefinitely improvise words."""
        while True:
            yield self.improvise()

class InfeasibleImproviserError(Exception):
    """ An exception raised when an improvisation problem is infeasible."""

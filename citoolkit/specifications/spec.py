"""Contains the Spec class, from which all specifications must inherit,
and the AbstractSpec class, which allows one to perform the union,
intersection, and negation operations on specifications."""

import copy
from enum import Enum
from typing import Set

class Spec:
    """ The Specification class is a parent class to all specifications.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: Set[str]) -> None:
        self.alphabet = alphabet

    def accepts(self, word) -> bool:
        raise NotImplementedError(self.__class__.__name__ + " has not implemented 'accepts'.")

    def __or__(self, other):
        """ Computes an abstract specification that accepts only words accepted
            by self or accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be unioned with self.
        """
        return AbstractSpec(self, other, SpecOp.UNION)

    def __and__(self, other):
        """ Computes an abstract specification that accepts only words accepted
            by self and accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be intersected with self.
        """
        return AbstractSpec(self, other, SpecOp.INTERSECTION)

    def __invert__(self):
        """ Computes an abstract specification that accepts only words not accepted
            by self. The returned specification will be the logical negation of self.
        """
        return AbstractSpec(self, None, SpecOp.NEGATION)

class SpecOp(Enum):
    UNION = 1
    INTERSECTION = 2
    NEGATION = 3

class AbstractSpec(Spec):
    def __init__(self, spec_1: Spec, spec_2: Spec, operation: SpecOp):
        # Performs checks to make sure that AbstractSpec is being used
        # correctly.
        alphabet = spec_1.alphabet & spec_2.alphabet
        if alphabet != spec_1.alphabet or alphabet != spec_2.alphabet:
            raise ValueError("Cannot perform operations on specifications with different alphabets.")

        if operation not in SpecOp:
            raise ValueError("Unsupported specification operation.")

        if (operation in [SpecOp.UNION, SpecOp.INTERSECTION]) \
          and (not isinstance(spec_1, Spec) or not isinstance(spec_2, Spec)):
            raise ValueError("The union and intersection operations require two specifications as input.")

        if (operation == SpecOp.NEGATION) \
          and (not isinstance(spec_1, Spec) or spec_2 is not None):
            raise ValueError("The negation operation require one specification as input.")

        # Intializes super class and stores all parameters, or their
        # copies if appropriate.
        super().__init__(alphabet)

        self.spec_1 = copy.deepcopy(spec_1)
        self.spec_2 = copy.deepcopy(spec_2)
        self.operation = operation

    def accepts(self, word) -> bool:
        if self.operation == SpecOp.UNION:
            return self.spec_1.accepts(word) or self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return self.spec_1.accepts(word) and self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return not self.spec_1.accepts(word)

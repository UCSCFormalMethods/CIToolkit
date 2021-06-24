"""Contains the Spec class, from which all specifications must inherit,
and the AbstractSpec class, which allows one to perform the union,
intersection, and negation operations on specifications."""

import copy
from enum import Enum
from typing import List, Set

class Spec:
    """ The Spec class is a parent class to all specifications.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet: Set[str]) -> None:
        self.alphabet = frozenset(alphabet)

    def accepts(self, word: List[str]) -> bool:
        """ Returns true if the specification accepts word, and false otherwise.

        :param word: The word which is checked for membership in the lanugage
            of this specification.
        """
        raise NotImplementedError(self.__class__.__name__ + " has not implemented 'accepts'.")

    def language_size(self) -> int:
        """ Computes the number of strings accepted by this specification."""
        raise NotImplementedError(self.__class__.__name__ + " has not implemented 'language_size'.")

    def __or__(self, other: Spec) -> "AbstractSpec":
        """ Computes an abstract specification that accepts only words accepted
            by self or accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be unioned with self.
        """
        return AbstractSpec(self, other, SpecOp.UNION)

    def __and__(self, other: Spec) -> "AbstractSpec":
        """ Computes an abstract specification that accepts only words accepted
            by self and accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be intersected with self.
        """
        return AbstractSpec(self, other, SpecOp.INTERSECTION)

    def __invert__(self) -> "AbstractSpec":
        """ Computes an abstract specification that accepts only words not accepted
            by self. The returned specification will be the logical negation of self.
        """
        return AbstractSpec(self, None, SpecOp.NEGATION)

    def __sub__(self, other: Spec) -> "AbstractSpec":
        """ Computes an abstract specification that accepts only words accepted
            by self and not accepted by other. The returned specification will be the
            logical difference of self and other. This is shorthand for using the
            intersection and complement functions.

        :param other: The specification whose complement will be intersected with self.
        """
        complement_spec_2 = AbstractSpec(other, None, SpecOp.NEGATION)

        return AbstractSpec(self, complement_spec_2, SpecOp.INTERSECTION)

class AbstractSpec(Spec):
    """ The AbstractSpec class represents the language that results from a SpecOp
    on one or two specs.

    :param spec_1: The first specification in the operation.
    :param spec_2: The second specificaton in the operation. In the case of a
        unary operation, spec_2 is None.
    :param operation: The operation to be performed on spec_1/spec_2.
    """
    def __init__(self, spec_1: Spec, spec_2: Spec, operation: SpecOp):
        # Performs checks to make sure that AbstractSpec is being used
        # correctly.
        alphabet = spec_1.alphabet | spec_2.alphabet
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

    def accepts(self, word: List[str]) -> bool:
        """ Returns true if the specification accepts word, and false otherwise.

        :param word: The word which is checked for membership in the lanugage
            of this specification.
        """
        if self.operation == SpecOp.UNION:
            return self.spec_1.accepts(word) or self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return self.spec_1.accepts(word) and self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return not self.spec_1.accepts(word)
        else:
            raise NotImplementedError(str(self.operation) + " is not currently supported.")

    def language_size(self) -> int:
        """ Computes the number of strings accepted by this specification.
            For an AbstractSpec, we first try to compute it's explicit form,
            in which case we can rely on the subclasses' counting method.
            Otherwise, we make as much of the AbstractSpec tree as explicit as
            possible, and then check if we have a "hack" to compute the
            size of the language anyway.
        """
        raise NotImplementedError()

    def explicit(self):
        """ Computes an explicit form for this AbstractSpec, raising an exception
            if this is not possible.
        """
        raise NotImplementedError()

class SpecOp(Enum):
    """ An enum enconding the different operations that can be
        performed on specifications.
    """
    UNION = 1
    INTERSECTION = 2
    NEGATION = 3

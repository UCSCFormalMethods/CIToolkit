"""Contains the Spec class, from which all specifications must inherit,
and the AbstractSpec class, which allows one to perform the union,
intersection, and negation operations on specifications."""

from __future__ import annotations
from typing import List, Set, Tuple

import copy
from enum import Enum

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

    def sample(self) -> Tuple[str]:
        """ Generate a word uniformly at random from this specification."""
        raise NotImplementedError(self.__class__.__name__ + " has not implemented 'sample'.")

    def __or__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
            by self or accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be unioned with self.
        """
        return AbstractSpec(self, other, SpecOp.UNION)

    def __and__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
            by self and accepted by other. The returned specification will be the
            logical intersection of self and other.

        :param other: The specification that will be intersected with self.
        """
        return AbstractSpec(self, other, SpecOp.INTERSECTION)

    def __invert__(self) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words not accepted
            by self. The returned specification will be the logical negation of self.
        """
        return AbstractSpec(self, None, SpecOp.NEGATION)

    def __sub__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
            by self and not accepted by other. The returned specification will be the
            logical difference of self and other. This is shorthand for using the
            intersection and complement functions.

        :param other: The specification whose complement will be intersected with self.
        """
        complement_spec_2 = AbstractSpec(other, None, SpecOp.NEGATION)

        return AbstractSpec(self, complement_spec_2, SpecOp.INTERSECTION)

class SpecOp(Enum):
    """ An enum enconding the different operations that can be
        performed on specifications.
    """
    UNION = 1
    INTERSECTION = 2
    NEGATION = 3

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

        if spec_2 is not None:
            alphabet = spec_1.alphabet | spec_2.alphabet
            if alphabet != spec_1.alphabet or alphabet != spec_2.alphabet:
                raise ValueError("Cannot perform operations on specifications with different alphabets.")
        else:
            alphabet = spec_1.alphabet

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

        if spec_2 is not None:
            self.spec_2 = copy.deepcopy(spec_2)
        else:
            self.spec_2 = None

        self.operation = operation

        # Initializes explicit form to None. It will be assigned when computed
        self.explicit_form = None

    def accepts(self, word: List[str]) -> bool:
        """ Returns true if the specification accepts word, and false otherwise.

        :param word: The word which is checked for membership in the lanugage
            of this specification.
        """
        if self.operation == SpecOp.UNION:
            return self.spec_1.accepts(word) or self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return self.spec_1.accepts(word) and self.spec_2.accepts(word)
        elif self.operation == SpecOp.NEGATION:
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
        # Attempt to compute explicit form, and if so rely on the explicit form's
        # language_size implementation
        try:
            explicit_form = self.explicit()
            return explicit_form.language_size()
        except NotImplementedError:
            pass

        # Check if have a "hack" to compute language_size anyway


        # Otherwise, raise a NotImplementedError
        if self.spec_1.explicit is not None:
            refined_spec_1 = self.spec_1.explicit
        else:
            refined_spec_1 = self.spec_1

        if self.spec_2.explicit is not None:
            refined_spec_2 = self.spec_2.explicit
        else:
            refined_spec_2 = self.spec_2

        raise NotImplementedError("Computation if language_size for abstract specifications of types '" \
                                  + refined_spec_1.__class__.__name__ + "' and '" + refined_spec_2.__class__.__name__ \
                                  + " with operation " + str(self.operation) + " is not supported.")

    def sample(self) -> Tuple[str]:
        """ Samples uniformly at random from the language of this specification.
            For an AbstractSpec, we first try to compute it's explicit form,
            in which case we can rely on the subclasses' sample method.
            Otherwise, we make as much of the AbstractSpec tree as explicit as
            possible, and then check if we have a "hack" to sample from the
            language anyway.
        """
        # Attempt to compute explicit form, and if so rely on the explicit form's
        # sample implementation
        try:
            explicit_form = self.explicit()
            return explicit_form.sample()
        except NotImplementedError:
            pass

        # Check if have a "hack" to compute language_size anyway


        # Otherwise, raise a NotImplementedError
        if self.spec_1.explicit is not None:
            refined_spec_1 = self.spec_1.explicit
        else:
            refined_spec_1 = self.spec_1

        if self.spec_2.explicit is not None:
            refined_spec_2 = self.spec_2.explicit
        else:
            refined_spec_2 = self.spec_2

        raise NotImplementedError("Uniform sampling for abstract specifications of types '" \
                                  + refined_spec_1.__class__.__name__ + "' and '" + refined_spec_2.__class__.__name__ \
                                  + " with operation " + str(self.operation) + " is not supported.")

    def explicit(self) -> Spec:
        """ Computes an explicit form for this AbstractSpec, raising an exception
            if this is not possible.
        """
        if self.explicit_form is not None:
            return self.explicit_form

        # Import explicit specification classes. (Done here to avoid circular import)
        from citoolkit.specifications.dfa import Dfa

        # Ensures that children are in explicit form and assign them to shorthand variables.
        if isinstance(self.spec_1, AbstractSpec):
            spec_1_explicit = self.spec_1.explicit()
        else:
            spec_1_explicit = self.spec_1

        if isinstance(self.spec_2, AbstractSpec):
            spec_2_explicit = self.spec_2.explicit()
        else:
            spec_2_explicit = self.spec_2

        # Attempts to make an explicit specification, raising an error
        # if such a construction is not supported.
        if isinstance(spec_1_explicit, Dfa) and (spec_2_explicit is None or isinstance(spec_2_explicit, Dfa)):
            if self.operation == SpecOp.UNION:
                self.explicit_form = Dfa.union_construction(spec_1_explicit, spec_2_explicit)
            elif self.operation == SpecOp.INTERSECTION:
                self.explicit_form = Dfa.intersection_construction(spec_1_explicit, spec_2_explicit)
            elif self.operation == SpecOp.NEGATION:
                self.explicit_form = spec_1_explicit.negation()
            else:
                raise NotImplementedError("Explict construction for '" + spec_1_explicit.__class__.__name__ + \
                                      "' and '" + spec_2_explicit.__class__.__name__ + "' with operation '" + \
                                      str(self.operation) + "' is not supported.")
        else:
            raise NotImplementedError("Explict constructions for '" + spec_1_explicit.__class__.__name__ + \
                                      "' and '" + spec_2_explicit.__class__.__name__ + " are not supported.")

        return self.explicit_form

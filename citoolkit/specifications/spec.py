"""Contains the Spec class, from which all specifications must inherit,
and the AbstractSpec class, which allows one to perform the union,
intersection, and negation operations on specifications."""

from __future__ import annotations
from typing import Any

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum

####################################################################################################
# Alphabet Classes
####################################################################################################

class Alphabet:
    """ The base class for all alphabets. Alphabets are used to determine compatibility
    when combining specifications"""
    @staticmethod
    def create_alphabet(alphabet_input: Any) -> Alphabet:
        """ Generic static function providing the ability to create an alphabet from different
        inputs. If the input is already an alphabet, that alphabet is simply returned. If the
        input is an iterable, a SymbolAlphabet is created over that iterable and returned. Otherwise,
        an error is raised.

        :param alphabet_input: The input that should be converted to an alphabet.
        :returns: An alphabet encoding of alphabet_input or NotImplemented.
        :raises ValueError: Raised when an alphabet_input with unsupported format is passed.
        """
        if isinstance(alphabet_input, Alphabet):
            # Already an alphabet, simply return it.
            return alphabet_input
        elif isinstance(alphabet_input, Iterable):
            # An iterable of symbols, create an alphabet from it.
            return SymbolAlphabet(alphabet_input)
        else:
            raise ValueError("Cannot create an alphabet from type " + str(type(alphabet_input)))

    def __and__(self, other):
        """ Provides a generic way to combine two alphabets. By default, equality is checked and
        if the two objects are equal, then this object is returned. If the two objects are not equal
        NotImplemented is returned.

        :param other: The other object to combine with this object.
        :returns: A combined alphabet or NotImplemented
        """
        if self != other:
            return NotImplemented

        return self

class SymbolAlphabet(Alphabet):
    """ A class representing a generic alphabet of symbols. Typically symbols are strings.

    :param symbols: An iterable containing whatever symbols are in this alphabet.
    """
    def __init__(self, symbols: Iterable):
        self.symbols = frozenset(symbols)

    def __eq__(self, other):
        """ Checks compatibility with another alphabet. If the other object is a SymbolAlphabet
        with the same symbols set, return True. Otherwise return NotImplemented.

        :param other: The other comparison object.
        :returns: Whether or not this alphabet is compatible with other.
        """
        if isinstance(other, SymbolAlphabet):
            return other.symbols == self.symbols

        return NotImplemented

    def __iter__(self):
        return self.symbols.__iter__()

    def __len__(self):
        return len(self.symbols)

class UniversalAlphabet(Alphabet):
    """ A class representing an alphabet compatible with any other alphabet"""
    def __eq__(self, other):
        """ Checks equality with another object. If the other object is an Alphabet,
        return True. Otherwise return NotImplemented.

        :param other: The other comparison object.
        :returns: Whether or not this alphabet is compatible to other.
        """
        if isinstance(other, Alphabet):
            return True

        return NotImplemented

    def __and__(self, other):
        return other

####################################################################################################
# Explicit Spec Classes
####################################################################################################

class Spec(ABC):
    """ The Spec class is a parent class to all exact and approximate specifications.
    All specifications must support the accepts function, which checks membership in
    the language the spec represents. Using this Spec objects can be combined into
    AbstractSpec objects with the Or (|), And (&), Not (~), and Difference (-) operators. 
    Certain compatible combinations of Spec objects in AbstractSpec form can also be
    collapsed into a single Spec again, by calling the explicit method. For more details,
    see the AbstractSpec class.

    :param alphabet: The alphabet this specification is defined over.
    """
    def __init__(self, alphabet):
        self.alphabet = Alphabet.create_alphabet(alphabet)

    @abstractmethod
    def accepts(self, word) -> bool:
        """ Returns true if the specification accepts word, and false otherwise.

        :param word: The word which is checked for membership in the lanugage
            of this specification.
        :returns: True if this Spec accepts word and false otherwise.
        """

    def explicit(self) -> Spec:
        """ The default implementation of explicit for all Spec objects, which
        is to simply return itself as it is already explicit. The only class
        that should need to override this is AbstractSpec.
        """
        return self

    def __or__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
        by self or accepted by other. The returned specification will be the
        logical intersection of self and other.

        :param other: The specification that will be unioned with self.
        :returns: An AbstractSpec that accepts words accepted by this Spec or other.
        """
        return AbstractSpec(self, other, SpecOp.UNION)

    def __and__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
        by self and accepted by other. The returned specification will be the
        logical intersection of self and other.

        :param other: The specification that will be intersected with self.
        :returns: An AbstractSpec that accepts words accepted by this Spec and other.
        """
        return AbstractSpec(self, other, SpecOp.INTERSECTION)

    def __invert__(self) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words not accepted
        by self. The returned specification will be the logical negation of self.

        :returns: An AbstractSpec that accepts words not accepted by this Spec.
        """
        return AbstractSpec(self, None, SpecOp.NEGATION)

    def __sub__(self, other: Spec) -> AbstractSpec:
        """ Computes an abstract specification that accepts only words accepted
        by self and not accepted by other. The returned specification will be the
        logical difference of self and other. This is shorthand for using the
        intersection and complement functions.

        :param other: The specification whose complement will be intersected with self.
        :returns: An AbstractSpec that accepts words accepted by this Spec and not by other.
        """
        complement_other_spec = AbstractSpec(other, None, SpecOp.NEGATION)

        return AbstractSpec(self, complement_other_spec, SpecOp.INTERSECTION)

class ExactSpec(Spec):
    """ The ExactSpec class is the parent class to all classes that support
    exact language size counting and sampling. These operations also require a lower
    and upper bound on word length.
    """
    @abstractmethod
    def language_size(self, min_length: int=None, max_length: int=None) -> int:
        """ Computes the number of words accepted by this specification.

        :param min_length: An inclusive lower bound on word size to consider.
        :param max_length: An inclusive upper bound on word size to consider.
        :returns: The size of the language accepted by this Spec.
        """

    @abstractmethod
    def sample(self, min_length: int=None, max_length: int=None, seed=None) -> Any:
        """ Generate a word uniformly at random from this specification. What
        a word is depends on the alphabet.

        :param min_length: An inclusive lower bound on word size to consider.
        :param max_length: An inclusive upper bound on word size to consider.
        :returns: A uniformly sampled word from the language of this Spec.
        """

class ApproxSpec(Spec):
    """ The ApproximateSpec class is the parent class to all classes that support
    approximate language size counting and sampling. As these operations are approximate
    they require a tolerance and/or confidence to be specified. The word length for these
    specs is also constant, and so the bounds on word length are omitted.
    """
    @abstractmethod
    def language_size(self, tolerance, confidence, seed=1) -> int:
        """ Approximately computes the number of words accepted by this specification.
            With probability 1 - confidence, the following holds true,
            true_count*(1 + confidence)^-1 <= returned_count <= true_count*(1 + confidence)

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :returns: The approximate size of the language accepted by this Spec.
        """

    @abstractmethod
    def sample(self, tolerance, seed=None) -> Any:
        """ Generate a word approximately uniformly at random from this specification. What
            a word is depends on the alphabet. Let true_prob be 1/true_count and returned_prob 
            be the probability of sampling any particular solution. With probability 1 - confidence, 
            the following holds true, 
            1/(1 + tolerance) * true_prob <= returned_prob <= (1 + tolerance) / true_prob

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :returns: An approximately uniformly sampled word from the language of this Spec.
        """

####################################################################################################
# Abstract Spec Classes
####################################################################################################

class SpecOp(Enum):
    """ An enum enconding the different operations that can be performed on specifications."""
    UNION = 1
    INTERSECTION = 2
    NEGATION = 3

class AbstractSpec(Spec):
    """ The AbstractSpec class represents the language that results from a SpecOp
    on one or two Specs.

    :param spec_1: The first specification in the operation.
    :param spec_2: The second specificaton in the operation. In the case of a
        unary operation, spec_2 is None.
    :param operation: The operation to be performed on the specs.
    :raises ValueError: Raised if a parameter is not supported or incompatible.
    """
    def __init__(self, spec_1: Spec, spec_2: Spec | None, operation: SpecOp):
        # Performs checks to make sure that AbstractSpec is being used correctly.
        # Determine if alphabet is well defined and if so freezes it.
        if (spec_2 is not None) and (spec_1.alphabet != spec_2.alphabet):
            raise ValueError("Cannot perform operations on specifications with incompatible alphabets.")

        if operation not in SpecOp:
            raise ValueError("Unsupported specification operation.")

        if (operation in [SpecOp.UNION, SpecOp.INTERSECTION]) \
          and (not isinstance(spec_1, Spec) or not isinstance(spec_2, Spec)):
            raise ValueError("The union and intersection operations require two specifications as input.")

        if (operation == SpecOp.NEGATION) \
          and (not isinstance(spec_1, Spec) or spec_2 is not None):
            raise ValueError("The negation operation requires exactly one specification as input.")

        # Intializes super class and stores all attributes, or their
        # copies if appropriate.
        if spec_2 is None:
            super().__init__(spec_1.alphabet)
        else:
            super().__init__(spec_1.alphabet & spec_2.alphabet)

        self.spec_1 = spec_1
        self.spec_2 = spec_2
        self.operation = operation

        # Initializes explicit form to None. It will be assigned when computed
        self.explicit_form = None

    def accepts(self, word: tuple[str,...]) -> bool:
        """ Returns true if the specification accepts word, and false otherwise.

        :param word: The word which is checked for membership in the language
            of this specification.
        :raises NotImplementedError: If an operation is passed that is not yet
            implemented.
        :returns: True if this AbstractSpec accepts word and False otherwise.
        """
        if self.operation == SpecOp.UNION:
            return self.spec_1.accepts(word) or self.spec_2.accepts(word)
        elif self.operation == SpecOp.INTERSECTION:
            return self.spec_1.accepts(word) and self.spec_2.accepts(word)
        elif self.operation == SpecOp.NEGATION:
            return not self.spec_1.accepts(word)
        else:
            raise NotImplementedError(str(self.operation) + " is not currently supported.")

    def language_size(self, *args, **kwargs) -> int:
        """ Computes the number of strings accepted by this specification.
        For an AbstractSpec, we first try to compute it's explicit form,
        in which case we can rely on the subclasses' counting method.
        Otherwise, we make as much of the AbstractSpec tree as explicit as
        possible, and then check if we have a "hack" to compute the
        size of the language anyway. Parameters dependent on whether the
        resulting spec is expact or abstract.

        :raises NotImplementedError: Raised if there is no method to compute
            language size for this choice of specification and operation.
        :returns: The size of the language accepted by this AbstractSpec.
        """
        # Attempt to compute explicit form, and if so rely on the explicit form's
        # language_size implementation
        try:
            explicit_form = self.explicit()
            return explicit_form.language_size(*args, **kwargs)
        except NotImplementedError:
            pass

        # Check if have a "hack" to compute language_size anyway


        # Otherwise, raise a NotImplementedError
        if isinstance(self.spec_1, AbstractSpec):
            spec_1_explicit = self.spec_1.explicit()
        else:
            spec_1_explicit = self.spec_1

        if isinstance(self.spec_2, AbstractSpec):
            spec_2_explicit = self.spec_2.explicit()
        else:
            spec_2_explicit = self.spec_2

        raise NotImplementedError("Computation if language_size for abstract specifications of types '" \
                                  + spec_1_explicit.__class__.__name__ + "' and '" + spec_2_explicit.__class__.__name__ \
                                  + " with operation " + str(self.operation) + " is not supported.")

    def sample(self, *args, **kwargs) -> Any:
        """ Samples uniformly at random from the language of this specification.
        For an AbstractSpec, we first try to compute it's explicit form,
        in which case we can rely on the subclasses' sample method.
        Otherwise, we make as much of the AbstractSpec tree as explicit as
        possible, and then check if we have a "hack" to sample from the
        language anyway. Parameters dependent on whether the resulting spec is
        expact or abstract.

        :raises NotImplementedError: Raised if there is no method to sample uniformly
            from this choice of specification and operation.
        :returns: A uniformly sampled word from the language of this AbstractSpec.
        """
        # Attempt to compute explicit form, and if so rely on the explicit form's
        # sample implementation
        try:
            explicit_form = self.explicit()
            return explicit_form.sample(*args, **kwargs)
        except NotImplementedError:
            pass

        # Check if have a "hack" to uniformly sample anyway


        # Otherwise, raise a NotImplementedError
        if isinstance(self.spec_1, AbstractSpec):
            spec_1_explicit = self.spec_1.explicit()
        else:
            spec_1_explicit = self.spec_1

        if isinstance(self.spec_2, AbstractSpec):
            spec_2_explicit = self.spec_2.explicit()
        else:
            spec_2_explicit = self.spec_2

        raise NotImplementedError("Uniform sampling for abstract specifications of types '" \
                                  + spec_1_explicit.__class__.__name__ + "' and '" + spec_2_explicit.__class__.__name__ \
                                  + " with operation " + str(self.operation) + " is not supported.")

    def explicit(self) -> Spec:
        """ Computes an explicit form for this AbstractSpec, raising an exception
        if this is not possible.

        :raises NotImplementedError: Raised if a necessary operation is not supported for
            a pair of specifications.
        :returns: An explicit subclass of Spec that represents the same language as this
            AbstractSpec.
        """
        if self.explicit_form is not None:
            return self.explicit_form

        # Import explicit specification classes. (Done here to avoid circular import)
        from citoolkit.specifications.dfa import Dfa
        from citoolkit.specifications.z3_formula import Z3Formula

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
        if isinstance(spec_1_explicit, UniverseSpec) or isinstance(spec_2_explicit, UniverseSpec):
            ## At least one specification is a UniverseSpec.

            # Pick the spec that is not a UniverseSpec (or any spec
            # if all are UniverseSpecs).
            if isinstance(spec_1_explicit, UniverseSpec):
                target_spec = spec_2_explicit
            else:
                target_spec = spec_1_explicit

            if self.operation == SpecOp.UNION:
                self.explicit_form = UniverseSpec()
            elif self.operation == SpecOp.INTERSECTION:
                self.explicit_form = target_spec
            elif self.operation == SpecOp.NEGATION:
                self.explicit_form = NullSpec()
            else:
                raise NotImplementedError("Explict construction for '" + spec_1_explicit.__class__.__name__ + \
                                      "' and '" + spec_2_explicit.__class__.__name__ + "' with operation '" + \
                                      str(self.operation) + "' is not supported.")

        elif isinstance(spec_1_explicit, NullSpec) or isinstance(spec_2_explicit, NullSpec):
            ## At least one specification is a NullSpec.

            # Pick the spec that is not a NullSpec (or any spec
            # if all are NullSpecs).
            if isinstance(spec_1_explicit, NullSpec):
                target_spec = spec_2_explicit
            else:
                target_spec = spec_1_explicit

            if self.operation == SpecOp.UNION:
                self.explicit_form = target_spec
            elif self.operation == SpecOp.INTERSECTION:
                self.explicit_form = NullSpec()
            elif self.operation == SpecOp.NEGATION:
                self.explicit_form = UniverseSpec()
            else:
                raise NotImplementedError("Explict construction for '" + spec_1_explicit.__class__.__name__ + \
                                      "' and '" + spec_2_explicit.__class__.__name__ + "' with operation '" + \
                                      str(self.operation) + "' is not supported.")

        elif isinstance(spec_1_explicit, Dfa) and (spec_2_explicit is None or isinstance(spec_2_explicit, Dfa)):
            ## All specifications are DFAs.

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

        elif isinstance(spec_1_explicit, Z3Formula) and (spec_2_explicit is None or isinstance(spec_2_explicit, Z3Formula)):
            ## All specifications are Z3 Forumlas.

            if self.operation == SpecOp.UNION:
                self.explicit_form = Z3Formula.union_construction(spec_1_explicit, spec_2_explicit)
            elif self.operation == SpecOp.INTERSECTION:
                self.explicit_form = Z3Formula.intersection_construction(spec_1_explicit, spec_2_explicit)
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

####################################################################################################
# Utility Spec Classes
####################################################################################################

class UniverseSpec(Spec):
    """ The UniverseSpec class represents a Spec that accepts
    all strings. It has the properties (Spec & UniverseSpec = Spec),
    (Spec | UniverseSpec = UniverseSpec), and (~UniverseSpec = NullSpec).
    """
    def __init__(self):
        # Since the UniverseSpec is defined regardless of alphabet,
        # we set the alphabet to be the UniversalAlphabet.
        super().__init__(UniversalAlphabet())

    def accepts(self, word) -> bool:
        """ The UniverseSpec has a universal language, so this always returns
        true.

        :param word: The word which is checked for membership in the language
            of this specification.
        :returns: True
        """
        return True

    def __eq__(self, other: object) -> bool:
        """ Checks equality with another object.

        :param other: The other spec with which to check equality.
        :returns: True if other is a UniverseSpec object and NotImplemented otherwise.
        """
        if isinstance(other, UniverseSpec):
            return True
        else:
            return NotImplemented


class NullSpec(Spec):
    """ The NullSpec class represents a Spec that accepts
    no strings. It has the properties (Spec & NulleSpec = NullSpec),
    (Spec | NullSpec = UniverseSpec), and (~NullSpec = UniverseSpec).
    """
    def __init__(self):
        # Since the NullSpec is defined regardless of alphabet,
        # we set the alphabet to be the UniversalAlphabet.
        super().__init__(UniversalAlphabet())

    def accepts(self, word) -> bool:
        """ The NullSpec has an empty language, so this always returns
        false.

        :param word: The word which is checked for membership in the language
            of this specification.
        :returns: False
        """
        return False

    def __eq__(self, other: object) -> bool:
        """ Checks equality with another object.

        :param other: The other spec with which to check equality.
        :returns: True if other is a NullSpec object and NotImplemented otherwise.
        """
        if isinstance(other, NullSpec):
            return True
        else:
            return NotImplemented

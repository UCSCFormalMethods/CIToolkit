""" Contains the Z3Formula approximate specification class."""

from __future__ import annotations
from collections.abc import Iterable

import math
import random

import z3

from citoolkit.specifications.spec import ApproxSpec, Alphabet
from citoolkit.specifications.bool_formula import BoolFormula, UnsatBoolFormula

class Z3Formula(ApproxSpec):
    """ The Z3Formula class encodes a Z3 formula specification. The Z3Formula must support
    the bitblasting tactic as all the language_size() and sample() computations are performed
    by transforming the Z3Formula spec into an equivalent BoolFormula Spec. All main_vars must
    be either Bool or BitVec variables.

    As an additional restriction, the specification reserves all variable names of the form
    "CI_Internal_(*)_(*)" where * is any sequence of characters. These variables names are used
    internally and so if they are present in the formula parameter, behaviour is undefined.

    It is important to note that each Z3Formula can maintain it's own context, though by default 
    it will use whatever context. This is because the memory overhead of contexts is significant
    and this is only required when accessing different Z3Formulas concurrently. If a new_context
    is desired, one can use the new_context parameter.

    :param formula: A Z3 Formula object.
    :param main_vars: Only variables in this iterator will be counted and sampled over.
    :param new_context: If set to True, the formula and main_variables will be translated to
        a fresh context.
    :param lazy: If set to True, the internal boolean representation of this Z3Formula will not
        be computed until needed.
    :raises ValueError: Returned if invalid parameters are passed to the constructor.
    """
    def __init__(self, formula, main_variables, new_context=False, lazy=True):
        super().__init__(Z3FormulaAlphabet())

        # Check validity of parameters.
        if not isinstance(main_variables, Iterable):
            raise ValueError("main_variables must be an iterator (See docstring for details).")

        # Create a new context for this formula
        if new_context:
            self.context = z3.Context()
        else:
            self.context = None

        # Store parameters in our new context.
        if self.context is not None:
            self.formula = formula.translate(self.context)
            self.orig_formula = formula.translate(self.context)
            self.main_variables = {var.translate(self.context) for var in main_variables}
        else:
            self.formula = formula
            self.orig_formula = formula
            self.main_variables = frozenset(main_variables)

        # Create equivalent BoolFormulaSpec and the mapping used to convert elements of that
        # spec to this one.
        self._bool_formula_spec = None
        self._main_variables_map = None

        self._feasible = None

        # If not lazy, compute the bool formula spec now
        if not lazy:
            self._create_bool_formula_spec()

    ####################################################################################################
    # ApproxSpec Functions
    ####################################################################################################

    def accepts(self, word) -> bool:
        """ Returns True if and only if a word is accepted by this specification.
        A word is accepted when, if each variable in word is assigned to the corresponding
        value in word, the resulting formula is satisfiable.

        :param word: A dictionary mapping variables to values.
        :returns: Whether or not the word is accepted by this spec.
        """
        if self.context is not None:
            translated_word = {var.translate(self.context): val for var,val in word.items()}
        else:
            translated_word = word

        formula = self.formula

        for var, val in translated_word.items():
            formula = z3.And(formula, var == val)

        solver = z3.Solver(ctx=self.context)
        solver.add(formula)

        return solver.check() == z3.sat

    def language_size(self, tolerance=0.8, confidence=0.2, seed=None) -> int:
        """ Approximately computes the number of solutions to this formula.
        With probability 1 - confidence, the following holds true,
        true_count*(1 + confidence)^-1 <= returned_count <= true_count*(1 + confidence)

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param seed: The randomized seed. By default this is equal to None, which means the
            internal random state will be used.
        :returns: The approximate number of solutions to this formula.
        """
        return self.bool_formula_spec.language_size(tolerance=tolerance, confidence=confidence, seed=seed)

    def sample(self, tolerance=15, seed=None):
        """ Generate a solution to this boolean formula approximately uniformly.
        Let true_prob be 1/true_count and returned_prob be the probability of sampling
        any particular solution. With probability 1 - confidence, the following holds true,
        1/(1 + tolerance) * true_prob <= returned_prob <= (1 + tolerance) / true_prob

        language_size() must be called before sample().

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param seed: The randomized seed. By default this is equal to None, which means the
            internal random state will be used.
        :returns: An approximately uniformly sampled solution to this formula.
        """
        return self._extract_main_vars(self.bool_formula_spec.sample(tolerance=tolerance, seed=seed))

    ####################################################################################################
    # Property Functions
    ####################################################################################################

    @property
    def feasible(self):
        """ Determines whether or not this formula has a solution.

        :returns: Returns True if this Z3Formula has a solution and False otherwise.
        """
        if self._feasible is not None:
            return self._feasible

        solver = z3.Solver(ctx=self.context)
        solver.add(self.formula)

        self._feasible = solver.check() == z3.sat

        return self._feasible

    @property
    def bool_formula_spec(self):
        """ Fetches the internal boolean formula representation of this spec.

        :returns: The internal boolean formula representing this spec.
        """
        self._create_bool_formula_spec()
        return self._bool_formula_spec

    @property
    def main_variables_map(self):
        """ Fetches the mapping from main variables to boolean values for this spec.

        :returns: The mapping from main variables to boolean values for this spec.
        """
        self._create_bool_formula_spec()
        return self._main_variables_map

    ####################################################################################################
    # Modification Functions
    ####################################################################################################

    def set_bool_formula_spec(self, spec):
        """ Sets the internal boolean formula representation of this spec.

        :param spec: The boolean formula spec to store internally
        """
        if not isinstance(spec, BoolFormula):
            raise ValueError("The spec parameter must be a BoolFormula spec.")

        self._bool_formula_spec = spec

    ####################################################################################################
    # Helper Functions
    ####################################################################################################

    def _create_bool_formula_spec(self):
        """ Augments the formula with Bool variables needed to track the
        variables in main_variables and then converts the Z3 formula to an
        equivalent BoolFormulaSpec while creating a mapping to convert CNF
        solutions back to the main_variables.
        """
        if self._bool_formula_spec is not None:
            return

        z3.set_param("seed", 1)

        # Check if Z3 formula is UNSAT. If so, set self._bool_formula_spec
        # to an unsat spec.
        if not self.feasible:
            self._bool_formula_spec = UnsatBoolFormula()
            self._main_variables_map = None
            return

        # Expression is SAT, add additional variables to ensure we can
        # extract values properly.
        internal_vars = []

        # Sort the main variable list for reproducibility
        main_var_list = sorted(self.main_variables, key = lambda x: str(x))

        for main_var in main_var_list:
            if isinstance(main_var, z3.z3.BoolRef):
                internal_var = self._create_internal_var(str(main_var), 0)
                target_var = z3.Bool(str(main_var), ctx=self.context)

                self.formula = z3.And(self.formula, (internal_var == target_var))

                internal_vars.append(str(internal_var))
            elif isinstance(main_var, z3.z3.BitVecRef):
                target_var = z3.BitVec(str(main_var), main_var.size(), ctx=self.context)

                # For each bit in the Bitvector, create a boolean equal to that bits value.
                for index in range(main_var.size()):
                    internal_var = self._create_internal_var(str(main_var), index)
                    bitmask = z3.BitVecSort(main_var.size(), ctx=self.context).cast(math.pow(2,index))

                    self.formula = z3.And(self.formula, (internal_var == ((target_var & bitmask) == bitmask)))

                    internal_vars.append(str(internal_var))
            else:
                raise ValueError(str(type(main_var)) + " is not a supported Z3 variable type and so cannot be a main variable.")

        # Bitblast Z3 expression and convert to DIMACS format.
        tactic = z3.Then('simplify', 'bit-blast', 'tseitin-cnf', ctx=self.context)
        goal = tactic(self.formula)

        assert len(goal) == 1

        dimacs_output = goal[0].dimacs(include_names=True)
        raw_clauses = [line.split() for line in dimacs_output.split("\n")]

        # Pop header clause
        raw_clauses.pop(0)

        # Parse raw clauses
        clauses = []
        var_mapping = {}

        for raw_clause in raw_clauses:
            if raw_clause[0] == "c":
                # Comment clause for a var mapping
                var_mapping[raw_clause[2]] = int(raw_clause[1])
            else:
                # Formula clause
                clauses.append([int(var) for var in raw_clause[:-1]])

        # Create BoolFormulaSpec using DIMACS format CNF
        dimacs_internal_vars = {var_mapping[internal_var] for internal_var in internal_vars}

        self._bool_formula_spec = BoolFormula(clauses, dimacs_internal_vars)

        # Create a mapping the name for each main variable to one or a list of
        # dimacs variables, encoding a boolean or bitvector respectively.
        self._main_variables_map = {}

        for main_var in main_var_list:
            if isinstance(main_var, z3.z3.BoolRef):
                internal_var_name = str(self._create_internal_var(str(main_var), 0))
                dimacs_var = var_mapping[internal_var_name]

                self._main_variables_map[str(main_var)] = dimacs_var
            else: # Bitvector case
                dimacs_var_list = []

                # For each bit in the Bitvector, create a boolean equal to that bits value.
                for index in range(main_var.size()):
                    internal_var_name = str(self._create_internal_var(str(main_var), index))

                    dimacs_var_list.append(var_mapping[internal_var_name])


                self._main_variables_map[str(main_var)] = tuple(dimacs_var_list)

    def _create_internal_var(self, name, index):
        """ Given a name and an index, returns a Z3 Boolean variable corresponding to it.
        Primarily used in _create_bool_formula_spec.

        :param name: The name of the original Z3 variable.
        :param index: The index in that Z3 variable.
        :returns: A Z3 Boolean variable corresponding to the name and index.
        """
        return z3.Bool("CI_Internal_(" + name + ")_(" + str(index) + ")", ctx=self.context)

    def _extract_main_vars(self, dimacs_sample):
        """ Given a sampled solution in DIMACS format, extracts the main variables of this Z3 formula
        from it.

        :param dimacs_sample: A sampled value from the internal boolean formula corresponding to this
            Z3 formula.
        :returns: A dictionary mapping each variable in main_variables to properly encoded sampled value.
        """
        # Ensure that we have computed the main variables mapping
        assert self._main_variables_map is not None

        # Convert the sample to dictionary form, with each variable mapping to 1 or 0.
        sample_dict = {abs(val): 1 if val > 0 else 0 for val in dimacs_sample}

        # Map all main variables to the appropriate value: A truth value for booleans
        # and an integer for bitvectors.
        main_var_values = dict()

        for main_var in self.main_variables:
            if isinstance(main_var, z3.z3.BoolRef):
                main_var_values[main_var] = bool(sample_dict[self._main_variables_map[str(main_var)]])
            else: # Bitvector case
                bitvector = [sample_dict[dimacs_var] for dimacs_var in self._main_variables_map[str(main_var)]]

                # Convert endianess
                bitvector.reverse()

                bitvector_val = 0

                for bit in bitvector:
                    bitvector_val = bit | (bitvector_val << 1)

                main_var_values[main_var] = bitvector_val

        return main_var_values

    ####################################################################################################
    # Constructor Functions
    ####################################################################################################

    @staticmethod
    def union_construction(formula_a, formula_b):
        """ Create a Z3 Formula that is the union of two other Z3 formulas.
        The new main_variables is the set of main variables in either formula.

        :param formula_a: The first input formula
        :param formula_b: The second input formula
        :returns: A Z3 formula that accepts models that satisfy either input formula.
        """
        if formula_a.context != formula_b.context:
            new_context = z3.Context()

            t_variables_a = [var.translate(new_context) for var in formula_a.main_variables]
            t_variables_b = [var.translate(new_context) for var in formula_b.main_variables]

            t_formula_a = formula_a.formula.translate(new_context)
            t_formula_b = formula_b.formula.translate(new_context)
        else:
            t_variables_a = formula_a.main_variables
            t_variables_b = formula_b.main_variables

            t_formula_a = formula_a.formula
            t_formula_b = formula_b.formula

        main_variables = set(t_variables_a) | set(t_variables_b)
        union_formula = z3.Or(t_formula_a, t_formula_b)

        return Z3Formula(union_formula, main_variables)

    @staticmethod
    def intersection_construction(formula_a, formula_b):
        """ Create a Z3 Formula that is the intersection of two other Z3 formulas.
        The new main_variables is the set of main variables in either formula.

        :param formula_a: The first input formula
        :param formula_b: The second input formula
        :returns: A Z3 formula that accepts models that satisfy both input formulas.
        """
        if formula_a.context != formula_b.context:
            new_context = z3.Context()

            t_variables_a = [var.translate(new_context) for var in formula_a.main_variables]
            t_variables_b = [var.translate(new_context) for var in formula_b.main_variables]

            t_formula_a = formula_a.formula.translate(new_context)
            t_formula_b = formula_b.formula.translate(new_context)
        else:
            t_variables_a = formula_a.main_variables
            t_variables_b = formula_b.main_variables

            t_formula_a = formula_a.formula
            t_formula_b = formula_b.formula

        main_variables = set(t_variables_a) | set(t_variables_b)
        intersection_formula = z3.And(t_formula_a, t_formula_b)

        return Z3Formula(intersection_formula, main_variables)

    @staticmethod
    def negation_construction(formula):
        """ Create a Z3 Formula that is the negation if the input Z3 formula.
        The main variables are the same as the main variables in the input formula.

        :param formula: The input formula
        :returns: A Z3 formula that accepts models that do not satisfy the input formulas.
        """
        main_variables = set(formula.main_variables)
        negation_formula = z3.Not(formula.formula)

        return Z3Formula(negation_formula, main_variables)

class Z3FormulaAlphabet(Alphabet):
    """ Alphabet class representing the abstract alphabet of a Z3 formula,
        which is a mapping from each variable number to a truth assignment
        or integer.
    """
    def __eq__(self, other):
        if isinstance(other, Z3FormulaAlphabet):
            return True

        return NotImplemented

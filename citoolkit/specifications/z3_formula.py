""" Contains the Z3Formula approximate specification class."""

from __future__ import annotations
from collections.abc import Iterable

import math
import random

import z3

from citoolkit.specifications.spec import ApproxSpec, Alphabet
from citoolkit.specifications.bool_formula import BoolFormula, UnsatBoolFormula
from citoolkit.util.logging import cit_log

class Z3Formula(ApproxSpec):
    """ The Z3Formula class encodes a Z3 formula specification. The Z3Formula must support
    the bitblasting tactic as all the language_size() and sample() computations are performed
    by transforming the Z3Formula spec into an equivalent BoolFormula Spec. All main_vars must
    also be either Bool or BitVec variables.

    As an additional restriction, the specification reserves all variable names of the form
    "CI_Internal_(*)_(*)" where * is any sequence of characters. These variables names are used
    internally and so if they are present in the formula parameter, behaviour is undefined.

    It is important to note that each Z3Formula maintains it's own context and all formulas
    will be translated into that context.

    :param formula: A Z3 Formula object.
    :param main_vars: Only variables in this iterator will be counted and sampled over.
    """
    def __init__(self, formula, main_variables, lazy_bool_spec=False):
        super().__init__([0,1])

        # Check validity of parameters.
        if not isinstance(main_variables, Iterable):
            raise ValueError("main_variables must be an iterator (See docstring for details).")

        # Create a new context for this formula
        self.context = z3.Context()

        # Store parameters in our new context.
        self.formula = formula.translate(self.context)
        self.orig_formula = formula.translate(self.context)
        self.main_variables = [var.translate(self.context) for var in main_variables]

        # Create equivalent BoolFormulaSpec and the mapping between them.
        self.bool_formula_spec = None
        self.main_variables_map = None

        self.feasible = None

        # If not lazy, compute the bool formula spec now
        if not lazy_bool_spec:
            self._create_bool_formula_spec()

    def _create_bool_formula_spec(self):
        """ Augments the formula with Bool variables needed to track the
        variables in main_variables and then converts the Z3 formula to an
        equivalent BoolFormulaSpec while creating a mapping to convert CNF
        solutions back to the main_variables.
        """
        if self.bool_formula_spec is not None:
            return

        # Check if Z3 formula is UNSAT. If so, set self.bool_formula_spec
        # to an unsat spec.
        solver = z3.Solver(ctx=self.context)
        solver.add(self.formula)

        if solver.check() != z3.sat:
            self.feasible = False
            self.bool_formula_spec = UnsatBoolFormula()
            self.main_variables_map = None
            return

        self.feasible = True

        # Expression is SAT, add additional variables to ensure we can
        # extract values properly.
        internal_vars = []

        for main_var in self.main_variables:
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

        self.bool_formula_spec = BoolFormula(clauses, dimacs_internal_vars)

        # Create a mapping the name for each main variable to one or a list of
        # dimacs variables, encoding a boolean or bitvector respectively.
        self.main_variables_map = {}

        for main_var in self.main_variables:
            if isinstance(main_var, z3.z3.BoolRef):
                internal_var_name = str(self._create_internal_var(str(main_var), 0))
                dimacs_var = var_mapping[internal_var_name]

                self.main_variables_map[str(main_var)] = dimacs_var
            else: # Bitvector case
                dimacs_var_list = []

                # For each bit in the Bitvector, create a boolean equal to that bits value.
                for index in range(main_var.size()):
                    internal_var_name = str(self._create_internal_var(str(main_var), index))

                    dimacs_var_list.append(var_mapping[internal_var_name])


                self.main_variables_map[str(main_var)] = tuple(dimacs_var_list)

    def _create_internal_var(self, name, index):
        """ Given a name and an index, returns a Z3 Boolean variable corresponding to it.
            Primarily used in _create_bool_formula_spec.

        :param name: The name of the original Z3 variable.
        :param index: The index in that Z3 variable.
        :returns: A Z3 Boolean variable corresponding to the name and index.
        """
        return z3.Bool("CI_Internal_(" + name + ")_(" + str(index) + ")", ctx=self.context)

    def extract_main_vars(self, dimacs_sample):
        """ Given a sampled solution in DIMACS format, extracts the main variables of this Z3 formula
        from it.

        :param dimacs_sample: A sampled value from the internal boolean formula corresponding to this
            Z3 formula.
        :returns: A dictionary mapping each variable in main_variables to properly encoded sampled value.
        """
        # Ensure that we have computed the main variables mapping
        assert self.main_variables_map is not None

        # Convert the sample to dictionary form, with each variable mapping to 1 or 0.
        sample_dict = {abs(val): 1 if val > 0 else 0 for val in dimacs_sample}

        # Map all main variables to the appropriate value: A truth value for booleans
        # and an integer for bitvectors.
        main_var_values = dict()

        for main_var in self.main_variables:
            if isinstance(main_var, z3.z3.BoolRef):
                main_var_values[str(main_var)] = bool(sample_dict[self.main_variables_map[str(main_var)]])
            else: # Bitvector case
                bitvector = [sample_dict[dimacs_var] for dimacs_var in self.main_variables_map[str(main_var)]]

                # Convert endianess
                bitvector.reverse()

                bitvector_val = 0

                for bit in bitvector:
                    bitvector_val = bit | (bitvector_val << 1)

                main_var_values[str(main_var)] = bitvector_val

        return main_var_values

    def accepts(self, word) -> bool:
        raise NotImplementedError()

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
        # Ensure the internal bool spec has been created
        self._create_bool_formula_spec()

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
        # Ensure the internal bool spec has been created
        self._create_bool_formula_spec()

        return self.extract_main_vars(self.bool_formula_spec.sample(tolerance=tolerance, seed=seed))

    def __getstate__(self):
        # Initialize state_dict
        state_dict = {}

        # Get the smt2 encoding of the actual formula
        solver = z3.Solver(ctx=self.context)
        solver.add(self.formula)
        state_dict["formula"] = solver.to_smt2()

        # Get the smt2 encoding of the original formula
        solver = z3.Solver(ctx=self.context)
        solver.add(self.orig_formula)
        state_dict["orig_formula"] = solver.to_smt2()

        # Encode the main variables to tuple representations
        packed_vars = []

        for main_var in self.main_variables:
            if isinstance(main_var, z3.z3.BoolRef):
                packed_vars.append((str(main_var), "Bool", 0))
            elif isinstance(main_var, z3.z3.BitVecRef):
                packed_vars.append((str(main_var), "BitVec", main_var.size()))

        state_dict["main_variables"] = packed_vars

        # Store the internal boolean formula spec, and main variables map.
        state_dict["bool_formula_spec"] = self.bool_formula_spec
        state_dict["main_variables_map"] = self.main_variables_map
        state_dict["feasible"] = self.feasible

        return state_dict

    def __setstate__(self, state):
        # Initialize super class
        super().__init__([0,1])

        # Create new context for this Z3Formula
        self.context = z3.Context()

        # Unpack actual and original formula
        self.formula = z3.parse_smt2_string(state["formula"], ctx=self.context)[0]
        self.orig_formula = z3.parse_smt2_string(state["orig_formula"], ctx=self.context)[0]

        # Unpack main variables
        unpacked_vars = []

        for packed_vars in state["main_variables"]:
            if packed_vars[1] == "Bool":
                unpacked_vars.append(z3.Bool(packed_vars[0], ctx=self.context))
            else:
                unpacked_vars.append(z3.BitVec(packed_vars[0], packed_vars[2], ctx=self.context))

        self.main_variables = frozenset(unpacked_vars)

        # Unpack internal variables
        self.bool_formula_spec = state["bool_formula_spec"]
        self.main_variables_map = state["main_variables_map"]
        self.feasible = state["feasible"]

    @staticmethod
    def union_construction(formula_a, formula_b):
        """ Create a Z3 Formula that is the union of two other Z3 formulas.
        The new main_variables is the set of main variables in either formula.

        :param formula_a: The first input formula
        :param formula_b: The second input formula
        :returns: A Z3 formula that accepts models that satisfy either input formula.
        """
        new_context = z3.Context()

        t_variables_a = [var.translate(new_context) for var in formula_a.main_variables]
        t_variables_b = [var.translate(new_context) for var in formula_b.main_variables]

        t_formula_a = formula_a.formula.translate(new_context)
        t_formula_b = formula_b.formula.translate(new_context)

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
        new_context = z3.Context()

        t_variables_a = [var.translate(new_context) for var in formula_a.main_variables]
        t_variables_b = [var.translate(new_context) for var in formula_b.main_variables]

        t_formula_a = formula_a.formula.translate(new_context)
        t_formula_b = formula_b.formula.translate(new_context)

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
        new_context = z3.Context()

        t_variables = [var.translate(new_context) for var in formula.main_variables]

        t_formula = formula.formula.translate(new_context)

        main_variables = set(t_variables)
        negation_formula = z3.Not(t_formula)

        return Z3Formula(negation_formula, main_variables)

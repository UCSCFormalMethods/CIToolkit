""" Contains the Z3Formula approximate specification class."""

from __future__ import annotations
from collections.abc import Iterable

import math

import z3

from citoolkit.specifications.spec import ApproximateSpec
from citoolkit.specifications.bool_formula import BoolFormula, UnsatBoolFormula

class Z3Formula(ApproximateSpec):
    """ The Z3Formula class encodes a Z3 formula specification. The Z3Formula must support
    the bitblasting tactic as all the language_size() and sample() computations are performed
    by transforming the Z3Formula spec into an equivalent BoolFormula Spec. All main_vars must
    also be either Bool or BitVec variables.

    As an additional restriction, the specification reserves all variable names of the form
    "CI_Internal_(*)_(*)" where * is any sequence of characters. These variables names are used
    internally and so if they are present in the formula parameter, behaviour is undefined.

    :param formula: A Z3 Formula object.
    :param main_vars: Only variables in this iterator will be counted and sampled over. Each
        element in this iterator should be a tuple of the form (VariableName, Type, Length)
        where VariableName is the name used when creating the variable, Type is either "Bool"
        or "BitVec", and Length is 1 if a Bool and the length of the BitVec if a BitVec.
    """
    def __init__(self, formula, main_variables):
        # Check validity of parameters.
        if (not isinstance(main_variables, Iterable)) or (len(main_variables) == 0):
            raise ValueError("main_variables must be a non empty iterator (See docstring for details).")

        for var_tuple in main_variables:
            var_name, var_type, var_length = var_tuple

            if not isinstance(var_name, str):
                raise ValueError("The first element of every tuple in main_variables must be the " +
                                 "string name of the variable.")

            if var_type == "Bool":
                if not var_length == 1:
                    raise ValueError("A tuple with the variable type Bool must have length 1.")
            elif var_type == "BitVec":
                if not isinstance(var_length, int):
                    raise ValueError("The length portion of tuple in main_variables must " +
                                     "be an integer.")
            else:
                raise ValueError("The second element of every tuple in main_variables must be " +
                                 "the type of the variable in string form. (\"Bool\" or \"BitVec\")")

        # Store parameters.
        self.formula = formula
        self.main_variables = list(main_variables)

        # Create equivalent BoolFormulaSpec and the mapping between them.
        self.bool_formula_spec = None
        self.main_variables_map = None

        self._create_bool_formula_spec()

    def _create_bool_formula_spec(self):
        """ Augments the formula with Bool variables needed to track the
        variables in main_variables and then converts the Z3 formula to an
        equivalent BoolFormulaSpec while creating a mapping to convert CNF
        solutions back to the main_variables.
        """
        # Check if Z3 formula is UNSAT. If so, set self.bool_formula_spec
        # to an unsat spec.
        solver = z3.Solver()
        solver.add(self.formula)

        if solver.check() != z3.sat:
            self.bool_formula_spec = UnsatBoolFormula()
            self.main_variables_map = None
            return

        # Expression is SAT, add additional variables to ensure we can
        # extract values properly.
        internal_vars = []

        for main_var in self.main_variables:
            var_name, var_type, var_length = main_var

            if var_type == "Bool":
                internal_var = Z3Formula._create_internal_var(var_name, 0)
                target_var = z3.Bool(var_name)

                self.formula = z3.And(self.formula, (internal_var == target_var))

                internal_vars.append(str(internal_var))
            else: # Bitvector case
                target_var = z3.BitVec(var_name, var_length)

                # For each bit in the Bitvector, create a boolean equal to that bits value.
                for index in range(var_length):
                    internal_var = Z3Formula._create_internal_var(var_name, index)
                    bitmask = z3.BitVecSort(var_length).cast(math.pow(2,index))

                    self.formula = z3.And(self.formula, (internal_var == ((target_var & bitmask) == bitmask)))

                    internal_vars.append(str(internal_var))

        # Bitblast Z3 expression and convert to CNF form.
        tactic = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')
        goal = tactic(self.formula)

        assert len(goal) == 1

        clauses = goal[0]

        # Create a mapping between Z3 variables and positive integers
        # in line with DIMACS format.
        mapping_context = ({}, 1)

        # Safe function that adds variable to mapping if it is not
        # already present and returns updated context.
        def update_context(var_name, context):
            mapping = context[0]
            next_var = context[1]

            if var_name not in mapping.keys():
                mapping[var_name] = next_var
                next_var += 1

            return (mapping, next_var)

        # Map all internal variables first.
        for var in internal_vars:
            mapping_context = update_context(var, mapping_context)

        # Map all remaining variables
        for clause in clauses:
            if z3.is_or(clause):
                # Compound clause
                for literal in clause.children():
                    if z3.is_not(literal):
                        # Negated literal
                        mapping_context = update_context(str(literal.children()[0]), mapping_context)
                    else:
                        # Positive literal
                        mapping_context = update_context(str(literal), mapping_context)
            elif z3.is_not(clause):
                # Negated unit clause
                mapping_context = update_context(str(clause.children()[0]), mapping_context)
            else:
                # Positive unit clause
                mapping_context = update_context(str(clause), mapping_context)

        # Convert Z3 CNF to DIMACS format CNF
        var_mapping = mapping_context[0]

        dimacs_clauses = []

        for clause in clauses:
            new_clause = []

            if z3.is_or(clause):
                # Compound clause
                for literal in clause.children():
                    if z3.is_not(literal):
                        # Negated literal
                        var_num = var_mapping[str(literal.children()[0])]
                        new_clause.append(-var_num)
                    else:
                        # Positive literal
                        var_num = var_mapping[str(literal)]
                        new_clause.append(var_num)
            elif z3.is_not(clause):
                # Negative unit clause
                var_num = var_mapping[str(clause.children()[0])]
                new_clause.append(-var_num)
            else:
                # Positive unit clause
                var_num = var_mapping[str(clause)]
                new_clause.append(var_num)

            dimacs_clauses.append(tuple(new_clause))

        dimacs_clauses = tuple(dimacs_clauses)

        # Create BoolFormulaSpec using DIMACS format CNF
        dimacs_internal_vars = {var_mapping[internal_var] for internal_var in internal_vars}

        self.bool_formula_spec = BoolFormula(dimacs_clauses, dimacs_internal_vars)

        # Create a mapping the name for each main variable to one or a list of
        # dimacs variables, encoding a boolean or bitvector respectively.
        self.main_variables_map = {}

        for main_var in self.main_variables:
            var_name, var_type, var_length = main_var

            if var_type == "Bool":
                internal_var_name = str(Z3Formula._create_internal_var(var_name, 0))
                dimacs_var = var_mapping[internal_var_name]

                self.main_variables_map[var_name] = dimacs_var
            else: # Bitvector case
                dimacs_var_list = []

                # For each bit in the Bitvector, create a boolean equal to that bits value.
                for index in range(var_length):
                    internal_var_name = str(Z3Formula._create_internal_var(var_name, index))

                    dimacs_var_list.append(var_mapping[internal_var_name])


                self.main_variables_map[var_name] = tuple(dimacs_var_list)

    @staticmethod
    def _create_internal_var(name, index):
        return z3.Bool("CI_Internal_(" + name + ")_(" + str(index) + ")")

    def extract_main_vars(self, dimacs_sample):
        """ Given a sampled solution in DIMACS format, extracts the main variables of this Z3 formula
        from it.
        """
        # Ensure that we have computed the main variables mapping
        assert self.main_variables_map is not None

        # Convert the sample to dictionary form, with each variable mapping to 1 or 0.
        sample_dict = {abs(val): 1 if val > 0 else 0 for val in dimacs_sample}

        # Map all main variables to the appropriate value: A truth value for booleans
        # and an integer for bitvectors.
        main_var_values = dict()

        for main_var in self.main_variables:
            var_name, var_type, _ = main_var

            if var_type == "Bool":
                main_var_values[var_name] = bool(sample_dict[self.main_variables_map[var_name]])
            else: # Bitvector case
                bitvector = [sample_dict[dimacs_var] for dimacs_var in self.main_variables_map[var_name]]

                # Convert endian
                bitvector.reverse()

                bitvector_val = 0

                for bit in bitvector:
                    bitvector_val = bit | (bitvector_val << 1)

                main_var_values[var_name] = bitvector_val

        return main_var_values

    def accepts(self, word) -> bool:
        raise NotImplementedError()

    def language_size(self, tolerance=0.8, confidence=0.2, seed=1, min_length: int = None, max_length: int = None) -> int:
        """ Approximately computes the number of solutions to this formula.
            With probability 1 - confidence, the following holds true,
            true_count*(1 + confidence)^-1 <= returned_count <= true_count*(1 + confidence)

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param min_length: Not applicable to boolean formula so ignored.
        :param max_length: Not applicable to boolean formula so ignored.
        :returns: The approximate number of solutions to this formula.
        """
        return self.bool_formula_spec.language_size(tolerance, confidence, seed=seed)

    def sample(self, tolerance=15, seed=1, min_length: int = None, max_length: int = None):
        """ Generate a solution to this boolean formula approximately uniformly.
            Let true_prob be 1/true_count and returned_prob be the probability of sampling
            any particular solution. With probability 1 - confidence, the following holds true,
            1/(1 + tolerance) * true_prob <= returned_prob <= (1 + tolerance) / true_prob

            language_size() must be called before sample().

        :param tolerance: The tolerance of the count.
        :param confidence: The confidence in the count.
        :param min_length: Not applicable to boolean formula so ignored.
        :param max_length: Not applicable to boolean formula so ignored.
        :returns: An approximately uniformly sampled solution to this formula.
        """
        return self.extract_main_vars(self.bool_formula_spec.sample(tolerance, seed=seed))

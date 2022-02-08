from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.costfunctions.cost_func import ApproxCostFunc

import z3

class Z3CostFormula(ApproxCostFunc):
    """ A Z3 Formula encoding cost.

    :param var_name: The name for a Z3 BitVec variable encoding cost.
    :param num_bits: The number of bits in the Z3 BitVec variable encoding cost.
    """
    def __init__(self, var_name, num_bits) -> None:
        self.var_name = var_name
        self.num_bits = num_bits
        self.max_cost = 2**num_bits - 1

        super().__init__([0,1])

    def realize(self, min_cost, max_cost) -> Z3Formula:
        """ Realize this cost function into a Z3Formula object that accepts
        only words with cost in the range [min_cost, max_cost].

        :param min_cost: The minimum cost accepted by the realized cost function.
        :param max_cost: The maximum cost accepted by the realized cost function.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """
        min_bound = z3.UGE(z3.BitVec(self.var_name, self.num_bits), min_cost)
        max_bound = z3.ULE(z3.BitVec(self.var_name, self.num_bits), max_cost)

        formula = z3.And(min_bound, max_bound)

        return Z3Formula(formula, [(self.var_name, "BitVec", self.num_bits)])

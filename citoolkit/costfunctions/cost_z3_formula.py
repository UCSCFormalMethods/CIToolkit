from citoolkit.specs.spec import ApproximateSpec
from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.costfunctions.cost_func import ApproximateCostFunc

class Z3CostFormula(ApproximateCostFunc):
    """ A Z3 Formula encoding cost.

    :param formula: A Z3 formula which is assumed to fix a Z3 bitvector
        named cost.
    """
    def __init__(self, formula, num_bits) -> None:
        self.num_bits = num_bits

        super().__init__(None)

    def realize(self, min_cost, max_cost) -> ApproximateSpec:
        """ Realize this cost function into a Z3Formula object that accepts
        only words with cost in the range [min_cost, max_cost].

        :param min_cost: The minimum cost accepted by the realized cost function.
        :param max_cost: The maximum cost accepted by the realized cost function.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """
        raise NotImplementedError()

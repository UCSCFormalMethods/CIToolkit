import z3

from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.labellingfunctions.labelling_func import ApproxLabelFunc

class Z3LabelFormula(ApproxLabelFunc):
    """ A Z3 Formula encoding cost.

    :param label_map: A mapping from label names to a unique integer expressable
        by a bitvector of size num_bits.
    :param var_name: The name for a Z3 BitVec variable encoding the label.
    :param num_bits: The number of bits in the Z3 BitVec variable encoding label.
    """
    def __init__(self, label_map, var_name, num_bits) -> None:
        self.label_map = label_map
        self.var_name = var_name
        self.num_bits = num_bits

        labels = frozenset(self.label_map.keys())

        super().__init__([0,1], labels)

    def realize(self, label) -> Z3Formula:
        """ Realize this cost function into a Z3Formula object that accepts
        only words with the label .

        :param label: The name of the label to be required.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """
        label_num = self.label_map[label]

        formula = z3.BitVec(self.var_name, self.num_bits) == label_num

        return Z3Formula(formula, [(self.var_name, "BitVec", self.num_bits)])

""" Contains the LabellingZ3Formula Class"""

import z3

from citoolkit.specifications.z3_formula import Z3Formula, Z3FormulaAlphabet
from citoolkit.labellingfunctions.labelling_func import ApproxLabelFunc

class LabellingZ3Formula(ApproxLabelFunc):
    """ A Z3 Formula encoding cost.

    :param label_map: A mapping from label names to a unique integer expressable
        by a bitvector of size num_bits.
    :param var_name: The name for a Z3 BitVec variable encoding the label.
    :param num_bits: The number of bits in the Z3 BitVec variable encoding label.
    """
    def __init__(self, label_map, label_var):
        self.label_map = label_map
        self.var_name = str(label_var)
        self.var_size = label_var.size()

        labels = frozenset(self.label_map.keys())

        super().__init__(Z3FormulaAlphabet(), labels)

    def realize(self, label) -> Z3Formula:
        """ Realize this cost function into a Z3Formula object that accepts
        only words with the label .

        :param label: The name of the label to be required.
        :returns: An ApproximateSpec object that accepts only words with cost
            in the range [min_cost, max_cost].
        """
        label_num = self.label_map[label]
        label_var = z3.BitVec(self.var_name, self.var_size)

        formula = label_var == label_num

        return Z3Formula(formula, [label_var])

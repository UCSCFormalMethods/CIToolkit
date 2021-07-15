""" Contains the ClassicCI class, which acts as an improviser
for the Classic CI problem.
"""

from __future__ import annotations
from typing import Tuple

import random

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError
from citoolkit.specifications.spec import Spec

class ClassicCI(Improviser):
    """ An improviser for the "Control Improvisation" problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param soft_constraint: A specification that must accept improvisations with
        probability 1 - epsilon.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param epsilon: The allowed tolerance with which we can not satisfy the soft constraint.
    :param prob_bounds: A tuple containing lower and upper bounds on the probability with
        which we can generate a word.
    :raises ValueError: If passed parameters are not of the correct type.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, soft_constraint: Spec, length_bounds: Tuple[int, int], \
                 epsilon: float, prob_bounds: Tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(soft_constraint, Spec):
            raise ValueError("The soft_constraint parameter must be a member of the Spec class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("The epsilon parameter should be between 0 and 1 inclusive.")

        if (len(prob_bounds) != 2) or (prob_bounds[0] < 0) or (prob_bounds[0] > prob_bounds[1]) or (prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Initializes improviser values. In this case i refers to I\A instead of I
        self.length_bounds = length_bounds

        self.i_spec = hard_constraint - soft_constraint
        self.a_spec = hard_constraint & soft_constraint

        i_size = self.i_spec.language_size(*self.length_bounds)
        a_size = self.a_spec.language_size(*self.length_bounds)

        min_prob, max_prob = prob_bounds

        self.i_prob = max(1 - max_prob * a_size, min_prob * i_size)
        self.a_prob = 1 - self.i_prob

        # Checks that improviser is feasible. If not raise an InfeasibleImproviserError.
        if (i_size + a_size) < (1/max_prob) or (min_prob != 0 and (i_size + a_size) > (1/min_prob)):
            if min_prob == 0:
                inv_min_prob = float("inf")
            else:
                inv_min_prob = 1/min_prob

            raise InfeasibleImproviserError("Violation of condition 1/max_prob <= (i_size + a_size) <= 1/min_prob. Instead, " \
                                            + str(1/max_prob) + " <= " + str(i_size + a_size) + " <= " + str(inv_min_prob))

        if (1 - epsilon)/max_prob > a_size:
            raise InfeasibleImproviserError("Violation of condition (1 - epsilon)/max_prob <= a_size. Instead, " \
                                            + str((1 - epsilon)/max_prob) + " <= " + str(a_size))

        if min_prob != 0 and epsilon/min_prob < i_size:
            raise InfeasibleImproviserError("Violation of condition epsilon/min_prob >= i_size. Instead, " \
                                            + str(epsilon/max_prob) + " >= " + str(i_size))

    def improvise(self) -> Tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        spec_choice = random.random()

        if spec_choice < self.i_prob:
            return self.i_spec.sample(*self.length_bounds)
        else:
            return self.a_spec.sample(*self.length_bounds)

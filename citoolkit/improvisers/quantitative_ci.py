""" Contains the QuantitativeCI class, which acts as an improviser
for the Quantitative CI problem.
"""

from __future__ import annotations

import random

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.costfunctions.cost_func import CostFunc

import multiprocessing
import gc
import time


class QuantitativeCI(Improviser):
    """ An improviser for the Quantitative Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param cost_bound: The maximum allowed expected cost for our improviser.
    :param prob_bounds: A tuple containing lower and upper bounds on the
        probability with which we can generate a word.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, cost_func: CostFunc, length_bounds: tuple[int, int], \
                 cost_bound: float, prob_bounds: tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(cost_func, CostFunc):
            raise ValueError("The cost_func parameter must be a member of the CostFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (len(prob_bounds) != 2) or (prob_bounds[0] < 0) or (prob_bounds[0] > prob_bounds[1]) or (prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Store all constructor parameters
        self.hard_constraint = hard_constraint
        self.cost_func = cost_func
        self.length_bounds = length_bounds
        self.cost_bound = cost_bound
        self.prob_bounds = prob_bounds

        # Initialize cost class specs.
        cost_specs = cost_func.decompose()

        self.i_specs = {}

        for cost in cost_func.costs:
            self.i_specs[cost] = hard_constraint & cost_specs[cost]

        with multiprocessing.Pool(min(multiprocessing.cpu_count() - 2, 32)) as p:
            func_input = [(cost, spec, length_bounds) for (cost, spec) in self.i_specs.items()]
            spec_items = p.map(get_language_size, func_input, chunksize=1)

            p.close()
            p.join()

            print("Done computing language sizes")
            print("Total CPU Time:", sum([x[2] for x in spec_items]))

            self.i_specs = {key:spec for (key,spec, _) in spec_items}

        # Compute the size of I.
        i_size = sum([self.i_specs[cost].language_size(*length_bounds) for cost in cost_func.costs])

        # Compute the number of words that can be assigned max probability.
        if prob_bounds[0] == prob_bounds[1]:
            num_max_prob_words = i_size
        else:
            num_max_prob_words = (1 - prob_bounds[0]*i_size)/(prob_bounds[1] - prob_bounds[0])

        # Assign probabilities to each cost class using greedy construction.
        # Sort costs in increasing order and assign default minimal probability to each cost class.
        # Cost classes with higher probabilities will have this overriden.
        words_assigned = 0
        self.sorted_costs = sorted(cost_func.costs)
        self.cost_probs = {cost:prob_bounds[0]*self.i_specs[cost].language_size(*length_bounds) for cost in cost_func.costs}

        for cost in self.sorted_costs:
            cost_class_size = self.i_specs[cost].language_size(*length_bounds)

            # Check if assigning maximum probability to this cost class would put us over budget.
            # If so, assign as much as possible and break.
            if words_assigned + cost_class_size >= num_max_prob_words:
                self.cost_probs[cost] = prob_bounds[1]*(num_max_prob_words - words_assigned) + prob_bounds[0]*(words_assigned + cost_class_size - num_max_prob_words)
                break

            # Otherwise, assign max probability to this cost class and continue.
            self.cost_probs[cost] = prob_bounds[1] * cost_class_size
            words_assigned += cost_class_size

        # Place improviser values in form used by improvise function and computes expected cost
        self.sorted_costs_weights = [self.cost_probs[cost] for cost in sorted(cost_func.costs)]

        self.expected_cost = sum(cost*self.cost_probs[cost] for cost in cost_func.costs)

        # Checks that improviser is feasible. If not raise an InfeasibleImproviserError.
        if i_size < (1/prob_bounds[1]) or (prob_bounds[0] != 0 and i_size > (1/prob_bounds[0])):
            if prob_bounds[0] == 0:
                inv_min_prob = float("inf")
            else:
                inv_min_prob = 1/prob_bounds[0]

            raise InfeasibleImproviserError("Violation of condition 1/prob_bounds[1] <= i_size <= 1/prob_bounds[0]. Instead, " \
                                            + str(1/prob_bounds[1]) + " <= " + str(i_size)  + " <= " + str(inv_min_prob))

        if self.expected_cost > self.cost_bound:
            raise InfeasibleImproviserError("Greedy construction does not satisfy cost constraint, meaning no improviser can. Minimum cost achieved was " + str(self.expected_cost) + ".")

    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_cost = random.choices(population=self.sorted_costs, weights=self.sorted_costs_weights, k=1)[0]

        return self.i_specs[selected_cost].sample(*self.length_bounds)

def get_language_size(param):
    start_time = time.process_time()
    cost, spec, length_bounds = param
    print("Computing Spec for Cost:", cost)
    spec = spec.explicit()
    print("Counting Spec for Cost:", cost)
    gc.collect()
    print("Cost: " + str(cost) + ", Size: " + str(spec.language_size(*length_bounds)))
    return (cost, spec, time.process_time() - start_time)

""" Contains the LabelledQuantitativeCI class, which acts as an improviser
for the Labelled Quantitative CI problem.
"""

from __future__ import annotations

import multiprocessing

import warnings
import random
import cvxpy as cp
import numpy as np

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.costfunctions.cost_func import CostFunc
from citoolkit.labellingfunctions.labelling_func import LabellingFunc

class LabelledQuantitativeCI(Improviser):
    """ An improviser for the Labelled Quantitative Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param cost_bound: The maximum allowed expected cost for our improviser.
    :param label_prob_bounds: A tuple containing lower and upper bounds on the
        marginal probability with which we can generate a word with a particular label.
    :param word_prob_bounds: A dictionary mapping each label in label_func to a tuple. Each
        tuple contains a lower and upper bound on the conditional probability of selecting
        a word with the associated label conditioned on the fact that we do select a word
        with label.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, cost_func: CostFunc, label_func: LabellingFunc, \
                 length_bounds: tuple[int, int], cost_bound: float, \
                 label_prob_bounds: tuple[float, float], word_prob_bounds: dict[str, tuple[float, float]]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(cost_func, CostFunc):
            raise ValueError("The cost_func parameter must be a member of the CostFunc class.")

        if not isinstance(label_func, LabellingFunc):
            raise ValueError("The label_func parameter must be a member of the LabellingFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (len(label_prob_bounds) != 2) or (label_prob_bounds[0] < 0) or (label_prob_bounds[0] > label_prob_bounds[1]) or (label_prob_bounds[1] > 1):
            raise ValueError("The label_prob_bounds parameter should contain two floats, with 0 <= label_prob_bounds[0] <= label_prob_bounds[1] <= 1.")

        for label in label_func.labels:
            target_prob_bounds = word_prob_bounds[label]

            if label not in word_prob_bounds.keys():
                raise ValueError("The word_prob_bounds parameter is missing conditional probability bounds for the label '" + label + "'.")

            if (len(target_prob_bounds) != 2) or (target_prob_bounds[0] < 0) or (target_prob_bounds[0] > target_prob_bounds[1]) or (target_prob_bounds[1] > 1):
                raise ValueError("The word_prob_bounds parameter should contain two floats, with 0 <= word_prob_bounds[" + label + "][0] <= word_prob_bounds[" + label + "][1] <= 1.")

        # Store all constructor parameters
        self.hard_constraint = hard_constraint
        self.cost_func = cost_func
        self.label_func = label_func
        self.length_bounds = length_bounds
        self.cost_bound = cost_bound
        self.label_prob_bounds = label_prob_bounds
        self.word_prob_bounds = word_prob_bounds

        # Decompose cost and label functions
        label_specs = label_func.decompose()

        # Create an improviser for each label class that has a distribution
        # that minimizes cost.
        self.label_improvisers = {}

        for label in label_func.labels:
            label_class_spec = hard_constraint & label_specs[label]

            self.label_improvisers[label] = self.LabelClassImproviser(label, label_class_spec, cost_func, length_bounds, word_prob_bounds[label])

            # If the label classes is empty, check that label_prob_bounds[0] == 0, as otherwise the improviser is infeasible.
            if self.label_improvisers[label].label_class_size == 0 and label_prob_bounds[0] > 0:
                raise InfeasibleImproviserError("The label class associated with '" + label + "' is empty, but label_prob_bounds[0] > 0.")

        # Create a set of all labels that have non empty label classes.
        feasible_labels = frozenset([label for label in label_func.labels if self.label_improvisers[label].label_class_size != 0])

        # Compute number of label classes that are assigned .
        # Note: The enumerate below is 0 indexed unlike in paper.
        if label_prob_bounds[0] == label_prob_bounds[1]:
            max_prob_classes = len(feasible_labels)
        else:
            max_prob_classes = int((1 - len(feasible_labels)*label_prob_bounds[0])/(label_prob_bounds[1] - label_prob_bounds[0]))

        # Compute probability assigned to the class at index (num_max_prob_classes + 1) in a sorted list of label classes.
        mid_class_prob = 1 - label_prob_bounds[1]*max_prob_classes - label_prob_bounds[0]*(len(feasible_labels) - max_prob_classes - 1)

        # Compute probability assigned to each label class
        label_class_probs = {}

        for label_num, label in enumerate(sorted(feasible_labels, key=lambda x: self.label_improvisers[x].expected_cost)):
            if label_num < max_prob_classes:
                label_class_probs[label] = label_prob_bounds[1]
            elif label_num == max_prob_classes:
                label_class_probs[label] = mid_class_prob
            else:
                label_class_probs[label] = label_prob_bounds[0]

        # Create values used by improvise method.
        self.sorted_labels = sorted(feasible_labels)
        self.sorted_label_weights = [label_class_probs[label] for label in self.sorted_labels]

        # Checks that this improviser is feasible. If not raise an InfeasibleImproviserError.
        if len(feasible_labels) < (1/label_prob_bounds[1]) or (label_prob_bounds[0] != 0 and len(feasible_labels) > (1/label_prob_bounds[0])):
            if label_prob_bounds[0] == 0:
                inv_min_label_prob = float("inf")
            else:
                inv_min_label_prob = 1/label_prob_bounds[0]

            raise InfeasibleImproviserError("Violation of condition 1/label_prob_bounds[1] <= len(label_func.labels) <= 1/label_prob_bounds[0]. Instead, " \
                                            + str(1/label_prob_bounds[1]) + " <= " + str(len(label_func.labels)) + " <= " + str(inv_min_label_prob))

        for label in feasible_labels:
            label_class_size = self.label_improvisers[label].label_class_size
            min_word_prob, max_word_prob = word_prob_bounds[label]

            if label_class_size < (1/max_word_prob) or (min_word_prob != 0 and label_class_size > (1/min_word_prob)):
                if min_word_prob == 0:
                    inv_min_word_prob = float("inf")
                else:
                    inv_min_word_prob = 1/min_word_prob

                raise InfeasibleImproviserError("Violation for label '" + label + "' " +
                                                "of condition 1/word_prob_bounds[" + label + "][1] <= label_class_size <= 1/word_prob_bounds[" + label + "][0]." +
                                                " Instead, " + str(1/max_word_prob) + " <= " + str(label_class_size) + " <= " + str(inv_min_word_prob))

        self.expected_cost = sum(label_class_probs[label]*self.label_improvisers[label].expected_cost for label in feasible_labels)

        if self.expected_cost > cost_bound:
            raise InfeasibleImproviserError("Greedy construction does not satisfy cost_bound, meaning no improviser can."\
                                            + " Minimum expected cost was " + str(self.expected_cost) + ".")

    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_label = random.choices(population=self.sorted_labels, weights=self.sorted_label_weights, k=1)[0]

        return self.label_improvisers[selected_label].improvise()


    ####################################################################################################
    # Label Class Improviser Inner Class
    ####################################################################################################

    class LabelClassImproviser(Improviser):
        """ An improviser for the for a Label Class in the Labelled Quantitative CI problem. The
        distribution of the improviser always minimizes expected cost.

        :param base_label: The label corresponding to this label class.
        :param base_spec: A specification that accepts all words that satisfy a hard constraint
            and have the label defining this label class.
        :param cost_func: A cost function that must associate a rational cost
            with all improvisations.
        :param length_bounds: A tuple containing lower and upper bounds on the length
            of a generated word.
        :param cost_bound: The maximum allowed expected cost for our improviser.
        :param prob_bounds: A tuple containing lower and upper bounds on the
            probability with which we can generate a word.
        :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
        """
        def __init__(self, base_label: str, base_spec: Spec, cost_func: CostFunc, \
                     length_bounds: tuple[int, int], prob_bounds: tuple[float, float]) -> None:
            # Store all constructor parameters
            self.base_label = base_label
            self.hard_constraint = base_spec
            self.cost_func = cost_func
            self.length_bounds = length_bounds
            self.prob_bounds = prob_bounds

            # Decompose cost_specs and create cost class specs for each one.
            cost_specs = cost_func.decompose()
            self.cost_class_specs = {}

            for cost in cost_func.costs:
                self.cost_class_specs[cost] = base_spec & cost_specs[cost]

            # Compute the size of each cost class and the size of the complete label class.
            cost_class_sizes = {cost:self.cost_class_specs[cost].language_size(*length_bounds) for cost in cost_func.costs}
            self.label_class_size = sum(cost_class_sizes.values())

            if self.label_class_size == 0 and prob_bounds[0] > 0:
                raise InfeasibleImproviserError("No strings are labelled with label '" + base_label + "', but word_prob_bounds[" + base_label + "][0] > 0.")

            # Compute the number of words (including partial words) that can be assigned max probability.
            if prob_bounds[0] == prob_bounds[1]:
                # All words must be assigned exactly prob_bounds[0]/prob_bounds[1], so all words get
                # "max" probability
                max_prob_words = self.label_class_size
            else:
                # Compute number of words (including partial words) that can be given prob_bounds[1]
                # while leaving at least prob_bounds[0] for all remaining words.
                max_prob_words = (1 - prob_bounds[0]*self.label_class_size)/(prob_bounds[1] - prob_bounds[0])

            # Assign probabilities to each cost class using greedy construction.
            # Sort costs in increasing order and assign default minimal probability to each cost class.
            # Cost classes with higher probabilities will then have this overriden.
            words_assigned = 0
            cost_class_probs = {cost:(prob_bounds[0] * cost_class_sizes[cost]) for cost in cost_func.costs}

            for cost in sorted(cost_func.costs):
                # Check if assigning maximum probability to this cost class would put us over budget.
                # If so, assign as much as possible and break.
                if words_assigned + cost_class_sizes[cost] >= max_prob_words:
                    cost_class_probs[cost] = prob_bounds[1]*(max_prob_words - words_assigned) \
                                           + prob_bounds[0]*(words_assigned + cost_class_sizes[cost] - max_prob_words)
                    break

                # Otherwise, assign max probability to this cost class and continue.
                cost_class_probs[cost] = prob_bounds[1] * cost_class_sizes[cost]

                words_assigned += cost_class_sizes[cost]

            # Create values used by improvise method and parent class.
            self.expected_cost = sum(cost*cost_class_probs[cost] for cost in cost_func.costs)

            self.sorted_costs = sorted(cost_func.costs)
            self.sorted_cost_weights = [cost_class_probs[cost] for cost in self.sorted_costs]

            # Check if improviser is "randomness-feasible", i.e. that it satisfies the randomness bounds.
            # Cost feasibility will be checked by the parent LQCI improviser.
            if self.label_class_size < (1/prob_bounds[1]) or (prob_bounds[0] != 0 and self.label_class_size > (1/prob_bounds[0])):
                if prob_bounds[0] == 0:
                    inv_min_prob = float("inf")
                else:
                    inv_min_prob = 1/prob_bounds[0]

                raise InfeasibleImproviserError("Violation of condition 1/word_prob_bounds[" + self.base_label \
                                                + "][1] <= label_class_size <= 1/word_prob_bounds[" + self.base_label + "][0]. " \
                                                + "Instead, " + str(1/prob_bounds[1]) + " <= " + str(self.label_class_size)  + " <= " + str(inv_min_prob))

        def improvise(self) -> tuple[str,...]:
            """ Improvise a single word.

            :returns: A single improvised word.
            """
            assert self.label_class_size != 0

            selected_cost = random.choices(population=self.sorted_costs, weights=self.sorted_cost_weights, k=1)[0]

            return self.cost_class_specs[selected_cost].sample(*self.length_bounds)

class MaxEntropyLabelledQuantitativeCI(Improviser):
    """ An improviser for the Maximum Entropy Labelled Quantitative Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param cost_bound: The maximum allowed expected cost for our improviser.
    :param label_prob_bounds: A tuple containing lower and upper bounds on the
        marginal probability with which we can generate a word with a particular label.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, cost_func: CostFunc, label_func: LabellingFunc, \
                 length_bounds: tuple[int, int], cost_bound: float, \
                 label_prob_bounds: tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(cost_func, CostFunc):
            raise ValueError("The cost_func parameter must be a member of the CostFunc class.")

        if not isinstance(label_func, LabellingFunc):
            raise ValueError("The label_func parameter must be a member of the LabellingFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (len(label_prob_bounds) != 2) or (label_prob_bounds[0] < 0) or (label_prob_bounds[0] > label_prob_bounds[1]) or (label_prob_bounds[1] > 1):
            raise ValueError("The label_prob_bounds parameter should contain two floats, with 0 <= label_prob_bounds[0] <= label_prob_bounds[1] <= 1.")

        if len(label_func.labels) == 0:
            raise InfeasibleImproviserError("This problem has no labels and therefore is infeasible. MELQCI problems must have at least one label.")

        if len(cost_func.costs) == 0:
            raise InfeasibleImproviserError("This problem has no costs and therefore is infeasible. MELQCI problems must have at least one cost.")

        # Store all constructor parameters
        self.hard_constraint = hard_constraint
        self.cost_func = cost_func
        self.label_func = label_func
        self.length_bounds = length_bounds
        self.cost_bound = cost_bound
        self.label_prob_bounds = label_prob_bounds

        # Compute cost class specs and their sizes.
        label_specs = label_func.decompose()
        cost_specs = cost_func.decompose()

        cost_class_specs = {}
        cost_class_sizes = {}

        print("Labels: ", label_func.labels)
        print("Costs: ", cost_func.costs)

        # for label in label_func.labels:
        #     label_class_spec = hard_constraint & label_specs[label]

        #     for cost in cost_func.costs:
        #         cost_class_specs[(label, cost)] = label_class_spec & cost_specs[cost]
        #         cost_class_sizes[(label, cost)] = cost_class_specs[(label, cost)].language_size(*length_bounds)

        for label in label_func.labels:
            label_class_spec = hard_constraint & label_specs[label]

            for cost in cost_func.costs:
                cost_class_specs[(label, cost)] = label_class_spec & cost_specs[cost]

        with multiprocessing.Pool(min(multiprocessing.cpu_count() - 2, 48)) as p:
            func_input = [(label, cost, spec, length_bounds) for ((label, cost),spec) in cost_class_specs.items()]
            spec_items = p.map(get_language_size, func_input)

            p.close()
            p.join()

            print("Done computing language sizes")

            cost_class_specs = {key:spec for (key,spec) in spec_items}

        for label in label_func.labels:
            for cost in cost_func.costs:
                cost_class_sizes[(label, cost)] = cost_class_specs[(label, cost)].language_size(*length_bounds)

        print("Making optimization problem...")

        # Create optimization variables and constants. Assuming n labels and m costs, the variable at position
        # x*m + y represents the probability allocated to words with label x and cost y.
        x = cp.Variable(len(label_func.labels)*len(cost_func.costs), nonneg=True)

        cost_class_sizes_vector = [max(1, cost_class_sizes[(label, cost)]) for label in sorted(label_func.labels) for cost in sorted(cost_func.costs)]

        entropy_equation = - cp.sum(cp.entr(x) + cp.multiply(x, np.log(cost_class_sizes_vector)))

        objective = cp.Minimize(entropy_equation)

        # Create constraints list
        constraints = []

        # (C1) Satisfaction of the cost bound
        expected_cost_equation = cp.sum(cp.multiply(x, sorted(cost_func.costs)*len(label_func.labels)))
        constraints.append(expected_cost_equation <= cost_bound)

        # (C2) - (C3) Randomness over Labels lower and upper bound
        for label_iter, label in enumerate(sorted(label_func.labels)):
            label_prob_equation = cp.sum(cp.multiply(x, np.concatenate([([1]*len(cost_func.costs) if i == label_iter else [0]*len(cost_func.costs)) for i in range(len(label_func.labels))])))

            constraints.append(label_prob_equation >= self.label_prob_bounds[0])
            constraints.append(label_prob_equation <= self.label_prob_bounds[1])

        # (C4) Non negative probability
        constraints.append(x >= 0)

        # (C5) Probability distribution sums to 1
        constraints.append(cp.sum(x) == 1)

        # (C6) Empty Cost Classes have 0 probability
        for label_iter, label in enumerate(sorted(label_func.labels)):
            for cost_iter, cost in enumerate(sorted(cost_func.costs)):
                if cost_class_sizes[(label, cost)] == 0:
                    empty_cost_class_vector = [0]*(len(label_func.labels)*len(cost_func.costs))
                    empty_cost_class_vector[label_iter*len(cost_func.costs) + cost_iter] = 1
                    constraints.append(cp.multiply(x, empty_cost_class_vector) == 0)

        print("Solving optimization problem...")

        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        print("Done solving....")

        # Check if problem is feasible. If not, raise an InfeasibleImproviserError.
        if "infeasible" in prob.status:
            raise InfeasibleImproviserError()

        if prob.status != "optimal":
            warnings.warn("Got unexpected value '" + prob.status + "' as optimizer output.")

        # Store improvisation variables
        self.sorted_cost_class_specs = [cost_class_specs[(label, cost)] for label in sorted(label_func.labels) for cost in sorted(cost_func.costs)]
        self.sorted_cost_class_weights = list(x.value)
        self.entropy = -1*result
        self.status = prob.status

        # Set all sorted_cost_class_weights that have empty cost classes to absolutely zero instead of very near 0.
        for label_iter, label in enumerate(sorted(label_func.labels)):
            for cost_iter, cost in enumerate(sorted(cost_func.costs)):
                if cost_class_sizes[(label, cost)] == 0:
                    self.sorted_cost_class_weights[label_iter*len(cost_func.costs) + cost_iter] = 0

    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_cost_class = random.choices(population=self.sorted_cost_class_specs, weights=self.sorted_cost_class_weights, k=1)[0]

        return selected_cost_class.sample(*self.length_bounds)


def get_language_size(param):
    label, cost, spec, length_bounds = param
    spec.language_size(*length_bounds)
    return ((label, cost), spec)

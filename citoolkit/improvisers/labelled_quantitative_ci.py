""" Contains the LabelledQuantitativeCI class, which acts as an improviser
for the Labelled Quantitative CI problem.
"""

from __future__ import annotations
from typing import Optional

import time
import warnings
import random
from numbers import Rational

import cvxpy as cp
import numpy as np
from multiprocess import Pool

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError, InfeasibleCostError,\
                                             InfeasibleLabelRandomnessError, InfeasibleWordRandomnessError
from citoolkit.specifications.spec import Spec
from citoolkit.costfunctions.cost_func import CostFunc
from citoolkit.labellingfunctions.labelling_func import LabellingFunc
from citoolkit.util.logging import cit_log

class _LabelledQuantitativeCIBase(Improviser):
    """ The base class for the LabelledQuantitativeCI class and the
    MaxEntropyLabelledQuantitativeCI class.

    When this class is intialized, all the label/cost class specs are
    created and stored in the class_specs dictionary attribute which matches
    all (label,cost) tuples to the spec that recognizes words with that label
    and cost.

    All child classes must then initialize two list attributes: class_keys
    which contains a (label,cost) tuple for every label/cost class to be sampled
    from, and class_probabilities which contains the probability for the label/cost
    class at the same index in class_keys.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    """
    def __init__(self, hard_constraint: Spec, cost_func: CostFunc, \
                 label_func: LabellingFunc, length_bounds: tuple[int, int],
                 direct_specs: Optional[dict[tuple[str, Rational], Spec]],\
                 num_threads: int, verbose: bool) -> None:
        # Set verbosity level and num_threads.
        self.verbose = verbose
        self.num_threads = num_threads

        self.class_specs = {}
        self.class_keys = None
        self.class_probabilities = None
        self.length_bounds = length_bounds

        if direct_specs is None:
            # Compute spec for each label/cost class.
            if self.verbose:
                cit_log("Beginning function decomposition and abstract spec construction.")

            label_specs = label_func.decompose()
            cost_specs = cost_func.decompose()

            for label in label_func.labels:
                label_class_spec = hard_constraint & label_specs[label]

                for cost in cost_func.costs:
                    self.class_specs[(label, cost)] = label_class_spec & cost_specs[cost]

            if self.verbose:
                cit_log("Function decomposition and abstract spec construction completed.")
        else:
            # Use precomputed specs found in direct_specs after verifying completeness.
            if self.verbose:
                cit_log("Using precomputed specs.")

            for label in label_func.labels:
                for cost in cost_func.costs:
                    if (label, cost) not in direct_specs:
                        raise ValueError("Incomplete direct_specs dictionary provided. " + str((label,cost)) + " is missing an associated Spec.")

                    self.class_specs[(label, cost)] = direct_specs[(label, cost)]

        # Count the language size for each spec.
        if self.verbose:
            start_time = time.time()
            cit_log("Beginning language size counting. Using " + str(num_threads) + " thread(s).")

        if num_threads <= 1:
            cpu_time = "N/A"
            # 1 thread, so compute all sizes iteratively.
            for label in label_func.labels:
                for cost in cost_func.costs:
                    self.class_specs[(label, cost)].language_size(*length_bounds)
        else:
            # Multiple threads, so create wrapper and thread pool and map specs before
            # resaving specs containing cached language sizes.
            with Pool(self.num_threads) as pool:
                # Helper function for pool.map
                def count_wrapper(class_id):
                    process_start_time = time.process_time()
                    label, cost = class_id
                    spec = self.class_specs[(label, cost)]

                    spec.language_size(*length_bounds)

                    return (class_id, spec, time.process_time() - process_start_time)

                class_ids = self.class_specs.keys()

                pool_output = pool.map(count_wrapper, class_ids)

                # Extract relevant info from pool_output
                cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))
                self.class_specs = {class_id: class_spec for class_id, class_spec, _ in pool_output}

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("Language size counting completed. Wallclock Runtime: " + wall_time + "  CPU Runtime: " + cpu_time)

    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word. Base class must populate self.class_probabilities
        before this method is called.

        :returns: A single improvised word.
        """
        if (self.class_probabilities is None) or (self.class_keys is None):
            raise Exception("Improvise function was called without first computing self.class_probabilities or self.class_keys.")

        target_class = random.choices(population=self.class_keys, weights=self.class_probabilities, k=1)[0]

        return self.class_specs[target_class].sample(*self.length_bounds)

class LabelledQuantitativeCI(_LabelledQuantitativeCIBase):
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
                 label_prob_bounds: tuple[float, float], word_prob_bounds: dict[str, tuple[float, float]],
                 direct_specs: Optional[dict[tuple[str, Rational], Spec]]=None,\
                 num_threads: int =1, verbose: bool =False) -> None:
        # Checks that parameters are well formed.
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

        if len(label_func.labels) == 0:
            raise InfeasibleImproviserError("This problem has no labels and therefore no improvisations.")

        if len(cost_func.costs) == 0:
            raise InfeasibleImproviserError("This problem has no costs and therefore no improvisations.")

        # Initialize LQCI base class.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, direct_specs=direct_specs, num_threads=num_threads, verbose=verbose)

        if self.verbose:
            start_time = time.time()
            cit_log("Beginning LQCI distribution calculation.")

        # Extract label/cost class sizes from class specs.
        cost_class_sizes = {}
        label_class_sizes = {}

        for label in label_func.labels:
            label_class_sizes[label] = 0

            for cost in cost_func.costs:
                cost_class_size = self.class_specs[(label, cost)].language_size(*length_bounds)
                cost_class_sizes[(label, cost)] = cost_class_size
                label_class_sizes[label] += cost_class_size

            if label_class_sizes[label] == 0 and label_prob_bounds[0] > 0:
                raise InfeasibleLabelRandomnessError("No strings are labelled with label '" + label + "', but label_prob_bounds[0] > 0.", 0)

        # Create a set of all labels that have non empty label classes.
        feasible_labels = frozenset([label for label in label_func.labels if label_class_sizes[label] != 0])

        # Create conditional distributions over labels.
        conditional_distributions = {}

        for label in feasible_labels:
            prob_bounds = word_prob_bounds[label]

            # Compute the number of words (including partial words) that can be assigned max probability.
            if prob_bounds[0] == prob_bounds[1]:
                # All words must be assigned exactly prob_bounds[0]/prob_bounds[1], so all words get
                # "max" probability
                max_prob_words = label_class_sizes[label]
            else:
                # Compute number of words (including partial words) that can be given prob_bounds[1]
                # while leaving at least prob_bounds[0] for all remaining words.
                max_prob_words = (1 - prob_bounds[0]*label_class_sizes[label])/(prob_bounds[1] - prob_bounds[0])

            # Assign probabilities to each cost class using greedy construction.
            # Sort costs in increasing order and assign default minimal probability to each cost class.
            # Cost classes with higher probabilities will then have this overriden.
            words_assigned = 0
            cost_class_probs = {cost:(prob_bounds[0] * cost_class_sizes[(label,cost)]) for cost in cost_func.costs}

            for cost in sorted(cost_func.costs):
                # Check if assigning maximum probability to this cost class would put us over budget.
                # If so, assign as much as possible and break.
                if words_assigned + cost_class_sizes[(label,cost)] >= max_prob_words:
                    cost_class_probs[cost] = prob_bounds[1]*(max_prob_words - words_assigned) \
                                           + prob_bounds[0]*(words_assigned + cost_class_sizes[(label,cost)] - max_prob_words)
                    break

                # Otherwise, assign max probability to this cost class and continue.
                cost_class_probs[cost] = prob_bounds[1] * cost_class_sizes[(label,cost)]

                words_assigned += cost_class_sizes[(label,cost)]

            conditional_distributions[label] = cost_class_probs

        # Compute expected cost for each label distribution
        conditional_costs = {label: sum([cost*probability for cost,probability in conditional_distributions[label].items()]) for label in feasible_labels}

        # Compute number of label classes that are assigned.
        if label_prob_bounds[0] == label_prob_bounds[1]:
            max_prob_classes = len(feasible_labels)
        else:
            max_prob_classes = int((1 - len(feasible_labels)*label_prob_bounds[0])/(label_prob_bounds[1] - label_prob_bounds[0]))

        # Compute probability assigned to the class at index (num_max_prob_classes + 1) in a sorted list of label classes.
        mid_class_prob = 1 - label_prob_bounds[1]*max_prob_classes - label_prob_bounds[0]*(len(feasible_labels) - max_prob_classes - 1)

        # Compute probability assigned to each label class
        marginal_distribution = {}

        for label_num, label in enumerate(sorted(feasible_labels, key=lambda x: conditional_costs[x])):
            if label_num < max_prob_classes:
                marginal_distribution[label] = label_prob_bounds[1]
            elif label_num == max_prob_classes:
                marginal_distribution[label] = mid_class_prob
            else:
                marginal_distribution[label] = label_prob_bounds[0]

        # Store improvisation values.
        self.class_keys = [(label, cost) for label in sorted(feasible_labels) for cost in sorted(cost_func.costs)]
        self.class_probabilities = [marginal_distribution[label]*conditional_distributions[label][cost] for label,cost in self.class_keys]

        # Checks that this improviser is feasible. If not raise an InfeasibleImproviserError.
        if len(feasible_labels) < (1/label_prob_bounds[1]) or (label_prob_bounds[0] != 0 and len(feasible_labels) > (1/label_prob_bounds[0])):
            if label_prob_bounds[0] == 0:
                inv_min_label_prob = float("inf")
            else:
                inv_min_label_prob = 1/label_prob_bounds[0]

            raise InfeasibleLabelRandomnessError("Violation of condition 1/label_prob_bounds[1] <= len(feasible_labels) <= 1/label_prob_bounds[0]. Instead, " \
                                            + str(1/label_prob_bounds[1]) + " <= " + str(len(feasible_labels)) + " <= " + str(inv_min_label_prob), len(feasible_labels))

        for label in feasible_labels:
            label_class_size = label_class_sizes[label]
            min_word_prob, max_word_prob = word_prob_bounds[label]

            if label_class_size < (1/max_word_prob) or (min_word_prob != 0 and label_class_size > (1/min_word_prob)):
                if min_word_prob == 0:
                    inv_min_word_prob = float("inf")
                else:
                    inv_min_word_prob = 1/min_word_prob

                raise InfeasibleWordRandomnessError("Violation for label '" + label + "' " +
                                                "of condition 1/word_prob_bounds[" + label + "][1] <= label_class_size <= 1/word_prob_bounds[" + label + "][0]." +
                                                " Instead, " + str(1/max_word_prob) + " <= " + str(label_class_size) + " <= " + str(inv_min_word_prob), label_class_size)

        expected_cost = sum(marginal_distribution[label]*conditional_costs[label] for label in feasible_labels)

        if expected_cost > cost_bound:
            raise InfeasibleCostError("Greedy construction does not satisfy cost_bound, meaning no improviser can."\
                                      + " Minimum expected cost was " + str(expected_cost) + ".", expected_cost)

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("LQCI distribution calculation completed. Wallclock Runtime: " + wall_time)

class MaxEntropyLabelledQuantitativeCI(_LabelledQuantitativeCIBase):
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
                 length_bounds: tuple[int, int], cost_bound: float, label_prob_bounds: tuple[float, float],
                 direct_specs: Optional[dict[tuple[str, Rational], Spec]]=None,\
                 num_threads: int =1, verbose: bool =False) -> None:
        # Checks that parameters are well formed.
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
            raise InfeasibleImproviserError("This problem has no labels and therefore no improvisations.")

        if len(cost_func.costs) == 0:
            raise InfeasibleImproviserError("This problem has no costs and therefore no improvisations.")

        # Initialize LQCI base class.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, direct_specs=direct_specs, num_threads=num_threads, verbose=verbose)

        if self.verbose:
            start_time = time.time()
            cit_log("Beginning MELQCI distribution calculation.")

        # Extract class sizes from class specs.
        cost_class_sizes = {}

        for label in label_func.labels:
            for cost in cost_func.costs:
                cost_class_sizes[(label, cost)] = self.class_specs[(label, cost)].language_size(*length_bounds)

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

            constraints.append(label_prob_equation >= label_prob_bounds[0])
            constraints.append(label_prob_equation <= label_prob_bounds[1])

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

        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        if self.verbose:
            cit_log("Solving MELQCI distribution optimization problem.")

        result = prob.solve(verbose=verbose)

        # Check if problem is infeasible. If so, raise an InfeasibleImproviserError.
        if "infeasible" in prob.status:
            raise InfeasibleImproviserError("Optimization problem infeasible. For more details run the associated non ME problem.")

        # Check if problem result was not infeasible, but also not optimal. If so, raise a warning.
        if prob.status != "optimal":
            warnings.warn("Got unexpected value '" + prob.status + "' as optimizer output.")

        # Store improvisation variables
        self.class_keys = [(label, cost) for label in sorted(label_func.labels) for cost in sorted(cost_func.costs)]
        self.class_probabilities = list(x.value)
        self.entropy = -1*result
        self.status = prob.status

        # Set all sorted_cost_class_weights that have empty cost classes to absolutely zero instead of very near 0.
        for label_iter, label in enumerate(sorted(label_func.labels)):
            for cost_iter, cost in enumerate(sorted(cost_func.costs)):
                if cost_class_sizes[(label, cost)] == 0:
                    self.class_probabilities[label_iter*len(cost_func.costs) + cost_iter] = 0

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("MELQCI distribution calculation completed. Wallclock Runtime: " + wall_time)

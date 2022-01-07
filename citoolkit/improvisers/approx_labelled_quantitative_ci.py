from __future__ import annotations
from typing import Optional

import time
import random
import math

from multiprocess import Pool

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError, InfeasibleCostError,\
                                             InfeasibleLabelRandomnessError, InfeasibleWordRandomnessError
from citoolkit.specifications.spec import ApproxSpec
from citoolkit.costfunctions.cost_func import ApproxCostFunc
from citoolkit.labellingfunctions.labelling_func import ApproxLabelFunc
from citoolkit.util.logging import cit_log


class _ApproxLabelledQuantitativeCIBase(Improviser):
    def __init__(self, hard_constraint: ApproxSpec, cost_func: ApproxCostFunc, \
                 label_func: ApproxLabelFunc, bucket_ratio, counting_tol, counting_conf, sampling_tol, \
                 num_threads: int, verbose: bool, length_bounds=None) -> None:
        # Set verbosity level and num_threads.
        self.verbose = verbose
        self.num_threads = num_threads

        self.class_specs = {}
        self.class_keys = None
        self.class_probabilities = None
        self.length_bounds = length_bounds

        # Compute bucket dimensions and total number of buckets.
        last_bucket_max_cost = 1
        self.bucket_bounds_list = []
        while last_bucket_max_cost < cost_func.max_cost:
            min_bucket_cost = last_bucket_max_cost+1
            max_bucket_cost = min(math.floor(min_bucket_cost*bucket_ratio), cost_func.max_cost)
            last_bucket_max_cost = max_bucket_cost

            self.bucket_bounds_list.append((min_bucket_cost, max_bucket_cost))

        self.num_buckets = len(self.bucket_bounds_list)

        # Compute spec for each label/cost class.
        if self.verbose:
            cit_log("Beginning compound Spec construction.")

        for label in label_func.labels:
            for bucket_iter, bucket_bounds in enumerate(self.bucket_bounds_list):
                self.class_specs[(label, bucket_iter)] = hard_constraint & label_func.realize(label) & cost_func.realize(*bucket_bounds)

        if self.verbose:
            cit_log("Compound Spec construction completed.")

        # Count the language size for each spec.
        if self.verbose:
            start_time = time.time()
            cit_log("Beginning language size counting. Using " + str(num_threads) + " thread(s).")

        if num_threads <= 1:
            cpu_time = "N/A"
            # 1 thread, so compute all sizes iteratively.
            for label in label_func.labels:
                for cost in cost_func.costs:
                    self.class_specs[(label, cost)].language_size(tolerance=counting_tol, confidence=counting_conf, seed=random.getrandbits(32))
        else:
            # Multiple threads, so create wrapper and thread pool and map specs before
            # resaving specs containing cached language sizes.
            with Pool(self.num_threads) as pool:
                # Create and extract bool_formula for each z3_formula
                bool_formula_specs = {}
                for class_id in self.class_specs.keys():
                    bool_formula_specs[class_id] = self.class_specs[class_id].bool_formula_spec

                # Helper function for pool.map
                def count_wrapper(class_id):
                    process_start_time = time.process_time()
                    spec = bool_formula_specs[class_id]

                    spec.language_size(tolerance=counting_tol, confidence=counting_conf, seed=random.getrandbits(32))

                    return (class_id, spec, time.process_time() - process_start_time)

                class_ids = self.class_specs.keys()

                pool_output = pool.map(count_wrapper, class_ids)

                # Extract relevant info from pool_output
                cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))

                for class_id, class_spec in self.class_specs.values():
                    class_spec[class_id].bool_formula_spec = {class_id: class_spec for class_id, class_spec, _ in pool_output}[class_id][1]

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

class ApproxLabelledQuantitativeCI(_ApproxLabelledQuantitativeCIBase):
    def __init__(self, hard_constraint: ApproxSpec, cost_func: ApproxCostFunc, \
                 label_func: ApproxLabelFunc, cost_bound, label_prob_bounds, word_prob_bounds, \
                 bucket_ratio, counting_tol, counting_conf, sampling_tol, \
                 num_threads: int, verbose: bool) -> None:
        # Checks that parameters are well formed.
        if not isinstance(hard_constraint, ApproxSpec):
            raise ValueError("The hard_constraint parameter must be a member of the ApproxSpec class.")

        if not isinstance(cost_func, ApproxCostFunc):
            raise ValueError("The cost_func parameter must be a member of the ApproxCostFunc class.")

        if not isinstance(label_func, ApproxLabelFunc):
            raise ValueError("The label_func parameter must be a member of the ApproxLabelFunc class.")

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
        super().__init__(self, hard_constraint, cost_func, label_func, \
                         bucket_ratio, counting_tol, counting_conf, sampling_tol, \
                         num_threads, verbose)

        if self.verbose:
            start_time = time.time()
            cit_log("Beginning LQCI distribution calculation.")

        # Extract label/cost class sizes from class specs.
        cost_class_sizes = {}
        label_class_sizes = {}

        for label in label_func.labels:
            label_class_sizes[label] = 0

            for bucket_iter in range(self.num_buckets):
                cost_class_size = self.class_specs[(label, bucket_iter)].language_size()
                cost_class_sizes[(label, bucket_iter)] = cost_class_size
                label_class_sizes[label] += cost_class_size

            if label_class_sizes[label] == 0 and label_prob_bounds[0] > 0:
                raise InfeasibleLabelRandomnessError("No strings are labelled with label '" + label + "', but label_prob_bounds[0] > 0.", 0)

        # Create a set of all labels that have non empty label classes.
        feasible_labels = frozenset([label for label in label_func.labels if label_class_sizes[label] != 0])

        # Initialize conditional weights to alpha for each word and sum up current marginal label probabilities.
        conditional_weights = {}

        for class_id in self.class_specs.keys():
            label, bucket_iter = class_id
            conditional_weights[class_id] = cost_class_sizes[class_id] * (word_prob_bounds[label][0]/1+counting_tol)

        label_sum_probs = {label:sum([prob for ((l, _), prob) in conditional_weights.items() if l == label]) for label, bucket_iter in self.class_specs.keys()}

        # Check if we've already broken probability bounds.
        for label_prob, label in label_sum_probs:
            assert label_prob <= 1
            #TODO Change this to exception with appropriate error message.

        # Add probabilites to appropriate classes (up to beta per word) without assigning more than 1 over each label.
        for label in label_func.labels:
            for bucket_iter in range(self.num_buckets):
                class_id = (label, bucket_iter)
                new_cost = min((1 + conditional_weights) * word_prob_bounds[label][0] * cost_class_sizes[class_id], 1 - label_sum_probs[label])

                conditional_weights[class_id] = new_cost

                # Update label sum probability
                label_sum_probs[label] = sum([prob for ((l, _), prob) in conditional_weights.items() if l == label])

        # Check if we've now broken probability bounds
        for label in label_func.labels:
            assert label_sum_probs[label] == 1
            #TODO Change this to exception with appropriate error message.

        # Calculate conditional exptected costs
        conditional_costs = {}
        for label in label_func.labels:
            conditional_costs[label] = sum([conditional_weights[(label,bucket_iter)]*self.bucket_bounds_list[bucket_iter][0] for bucket_iter in range(self.num_buckets)])

        # Now calculate marginal weights.
        marginal_weights = []

        u = math.floor((1 - len(label_func.labels)*label_prob_bounds[0])/(label_prob_bounds[1] - label_prob_bounds[0]))

        for label_iter in sorted(range(len(label_func.labels)), key=lambda x: conditional_costs[label_iter]):
            if label_iter < u:
                marginal_weights.append(label_prob_bounds[1])
            elif label_iter == u:
                marginal_weights.append(1 - label_prob_bounds[1]*u - label_prob_bounds[0]*(len(label_func.labels) - u - 1))
            else:
                marginal_weights.append(label_prob_bounds[0])

        assert sum(marginal_weights) == 1
        #TODO Change this to exception with appropriate error message.

        expected_cost = sum([marginal_weights[label] * conditional_costs[label] for label in label_func.labels])

        print("Marginal Weights:", marginal_weights)
        print("Expected Cost:", expected_cost)
        print()

        # sorted_label_weights = marginal_weights
        # sorted_labels = range(len(lo_locs))

        # sorted_cost_weights = {label_iter:[conditional_weights[(label_iter, cost_iter)] for cost_iter in range(max_r)] for label_iter in range(len(lo_locs))}
        # sorted_costs = range(max_r)

        # print("Sorted Label Weights:", sorted_label_weights)
        # print("Sorted Cost Weights:", sorted_cost_weights)

        # Store improvisation values.
        self.class_keys = [(label, bucket) for label in sorted(label_func.labels) for bucket in range(self.num_buckets)]
        self.class_probabilities = [marginal_weights[label]*conditional_weights[label][cost] for label,cost in self.class_keys]

        # # Checks that this improviser is feasible. If not raise an InfeasibleImproviserError.
        # if len(feasible_labels) < (1/label_prob_bounds[1]) or (label_prob_bounds[0] != 0 and len(feasible_labels) > (1/label_prob_bounds[0])):
        #     if label_prob_bounds[0] == 0:
        #         inv_min_label_prob = float("inf")
        #     else:
        #         inv_min_label_prob = 1/label_prob_bounds[0]

        #     raise InfeasibleLabelRandomnessError("Violation of condition 1/label_prob_bounds[1] <= len(feasible_labels) <= 1/label_prob_bounds[0]. Instead, " \
        #                                     + str(1/label_prob_bounds[1]) + " <= " + str(len(feasible_labels)) + " <= " + str(inv_min_label_prob), len(feasible_labels))

        # for label in feasible_labels:
        #     label_class_size = label_class_sizes[label]
        #     min_word_prob, max_word_prob = word_prob_bounds[label]

        #     if label_class_size < (1/max_word_prob) or (min_word_prob != 0 and label_class_size > (1/min_word_prob)):
        #         if min_word_prob == 0:
        #             inv_min_word_prob = float("inf")
        #         else:
        #             inv_min_word_prob = 1/min_word_prob

        #         raise InfeasibleWordRandomnessError("Violation for label '" + label + "' " +
        #                                         "of condition 1/word_prob_bounds[" + label + "][1] <= label_class_size <= 1/word_prob_bounds[" + label + "][0]." +
        #                                         " Instead, " + str(1/max_word_prob) + " <= " + str(label_class_size) + " <= " + str(inv_min_word_prob), label_class_size)

        # expected_cost = sum(marginal_distribution[label]*conditional_costs[label] for label in feasible_labels)

        # if expected_cost > cost_bound:
        #     raise InfeasibleCostError("Greedy construction does not satisfy cost_bound, meaning no improviser can."\
        #                               + " Minimum expected cost was " + str(expected_cost) + ".", expected_cost)

        # if self.verbose:
        #     wall_time = "{:.4f}".format(time.time() - start_time)
        #     cit_log("LQCI distribution calculation completed. Wallclock Runtime: " + wall_time)
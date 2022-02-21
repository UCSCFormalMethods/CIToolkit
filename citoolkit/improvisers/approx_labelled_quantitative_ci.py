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
                 seed, num_threads: int, verbose: bool, lazy_counting=False) -> None:
        # Save random state and seed new state.
        old_state = random.getstate()
        random.seed(seed)

        # Set verbosity level and num_threads.
        self.verbose = verbose
        self.num_threads = num_threads

        # Save counting tolerance
        self.sampling_tol = sampling_tol

        self.class_specs = {}
        self.class_keys = None
        self.class_probabilities = None

        # Compute bucket dimensions and total number of buckets.
        last_bucket_max_cost = 0
        self.bucket_bounds_list = [(0,0)]
        while last_bucket_max_cost < cost_func.max_cost:
            min_bucket_cost = last_bucket_max_cost+1
            max_bucket_cost = min(math.floor(min_bucket_cost*bucket_ratio), cost_func.max_cost)
            last_bucket_max_cost = max_bucket_cost

            self.bucket_bounds_list.append((min_bucket_cost, max_bucket_cost))

        self.num_buckets = len(self.bucket_bounds_list)


        # Compute spec for each label/cost class.
        if self.verbose:
            cit_log("Total buckets over all labels: " + str(len(label_func.labels) * self.num_buckets))
            start_time = time.time()
            cit_log("Beginning compound Spec construction. Using " + str(num_threads) + " thread(s).")

        class_ids = set()

        for label in label_func.labels:
            for bucket_iter, bucket_bounds in enumerate(self.bucket_bounds_list):
                class_ids.add((label, bucket_iter))

        self.class_specs = {}

        if num_threads <= 1:
            cpu_time = "N/A"
            for class_id in class_ids:
                label, bucket_iter = class_id
                bucket_bounds = self.bucket_bounds_list[bucket_iter]

                self.class_specs[class_id] = (hard_constraint & label_func.realize(label) & cost_func.realize(*bucket_bounds)).explicit()
        else:
            # Multiple threads, so create wrapper and thread pool and map abstract specs before
            # resaving explicit specs.

            with Pool(self.num_threads) as pool:
                # Helper function for pool.map
                def explicit_wrapper(class_id):
                    process_start_time = time.process_time()

                    label, bucket_iter = class_id
                    bucket_bounds = self.bucket_bounds_list[bucket_iter]

                    spec = (hard_constraint & label_func.realize(label) & cost_func.realize(*bucket_bounds)).explicit()

                    return (class_id, spec, time.process_time() - process_start_time)

                pool_output = pool.map(explicit_wrapper, class_ids)

                # Extract relevant info from pool_output
                cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))

                self.class_specs = {class_id: class_spec for class_id, class_spec, _ in pool_output}


        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("Compound Spec construction completed. Wallclock Runtime: " + wall_time + "  CPU Runtime: " + cpu_time)

        if not lazy_counting:
            # Count the language size for each spec if precount is enabled.
            if self.verbose:
                start_time = time.time()
                cit_log("Beginning language size counting. Using " + str(num_threads) + " thread(s).")

            if num_threads <= 1:
                cpu_time = "N/A"
                # 1 thread, so compute all sizes iteratively.
                for spec in self.class_specs.values():
                    spec.language_size(tolerance=counting_tol, confidence=counting_conf, seed=random.getrandbits(32))
            else:
                # Multiple threads, so create wrapper and thread pool and map specs before
                # resaving specs containing cached language sizes.

                # TODO: Make this general with a special case for Z3 Formulas
                with Pool(self.num_threads) as pool:
                    # Sort class spec keys for reproducability
                    class_ids = sorted(self.class_specs.keys())

                    # Get a random seed for each language size calculation
                    random_seeds = {}
                    for class_id in class_ids:
                        random_seeds[class_id] = random.getrandbits(32)

                    # Helper function for pool.map
                    def count_wrapper(class_id):
                        process_start_time = time.process_time()
                        spec = self.class_specs[class_id]

                        spec.language_size(tolerance=counting_tol, confidence=counting_conf, seed=random_seeds[class_id])

                        return (class_id, spec, time.process_time() - process_start_time)

                    pool_output = pool.map(count_wrapper, class_ids)

                    # Extract relevant info from pool_output
                    cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))

                    self.class_specs = {class_id: class_spec for class_id, class_spec, _ in pool_output}

            if self.verbose:
                wall_time = "{:.4f}".format(time.time() - start_time)
                cit_log("Language size counting completed. Wallclock Runtime: " + wall_time + "  CPU Runtime: " + cpu_time)

        # Restore random state
        random.setstate(old_state)

    def improvise(self, seed=None) -> tuple[str,...]:
        """ Improvise a single word. Base class must populate self.class_probabilities
        before this method is called.

        :returns: A single improvised word.
        """
        if (self.class_probabilities is None) or (self.class_keys is None):
            raise Exception("Improvise function was called without first computing self.class_probabilities or self.class_keys.")

        if seed is None:
            seed = random.getrandbits(32)

        # Cache the random state, set it to the seed, pick a target class, sample, and finally restore the random state.
        old_state = random.getstate()
        random.seed(seed)

        target_class = random.choices(population=self.class_keys, weights=self.class_probabilities, k=1)[0]

        sample = self.class_specs[target_class].sample(tolerance=self.sampling_tol, seed=random.getrandbits(32))

        random.setstate(old_state)

        return sample

class ApproxLabelledQuantitativeCI(_ApproxLabelledQuantitativeCIBase):
    def __init__(self, hard_constraint: ApproxSpec, cost_func: ApproxCostFunc, \
                 label_func: ApproxLabelFunc, cost_bound, label_prob_bounds, word_prob_bounds, \
                 bucket_ratio, counting_tol, counting_conf, sampling_tol, \
                 seed=None, lazy_counting=False, num_threads: int =1, verbose: bool =False) -> None:
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

        # If the lower prob bound for all labels is 0, don't precount.
        # precount = False

        # for label in label_func.labels:
        #     if word_prob_bounds[label][0] > 0:
        #         precount = True

        # Initialize LQCI base class.
        super().__init__(hard_constraint, cost_func, label_func, \
                         bucket_ratio, counting_tol, counting_conf, sampling_tol, \
                         seed, num_threads, verbose, lazy_counting=lazy_counting)

        if self.verbose:
            start_time = time.time()
            cit_log("Beginning LQCI distribution calculation.")

        # Cache the random state and set seed.
        old_state = random.getstate()
        random.seed(seed)

        # TODO Fix feasible labels hack here
        # Extract label/cost class sizes from class specs.
        # cost_class_sizes = {}
        # label_class_sizes = {}

        # for label in label_func.labels:
        #     label_class_sizes[label] = 0

        #     for bucket_iter in range(self.num_buckets):
        #         cost_class_size = self.class_specs[(label, bucket_iter)].language_size()
        #         cost_class_sizes[(label, bucket_iter)] = cost_class_size
        #         label_class_sizes[label] += cost_class_size

        #     if label_class_sizes[label] == 0 and label_prob_bounds[0] > 0:
        #         raise InfeasibleLabelRandomnessError("No strings are labelled with label '" + label + "', but label_prob_bounds[0] > 0.", 0)

        # Create a set of all labels that have non empty label classes.
        feasible_labels = frozenset(label_func.labels) #frozenset([label for label in label_func.labels if label_class_sizes[label] != 0])

        # Initialize conditional weights to alpha for each word and sum up current marginal label probabilities.
        conditional_weights = {}

        print(word_prob_bounds)

        for class_id in self.class_specs:
            label, bucket_iter = class_id
            # If alpha is 0, we can just set everything to 0. Otherwise compute everything.
            if word_prob_bounds[label][0] == 0:
                conditional_weights[class_id] = 0
            else:
                class_count = self.class_specs[class_id].language_size(tolerance=counting_tol, confidence=counting_conf, seed=random.getrandbits(32))
                conditional_weights[class_id] = class_count * (word_prob_bounds[label][0]/(1+counting_tol))

        label_sum_probs = {label:sum([prob for ((l, _), prob) in conditional_weights.items() if l == label]) for label, bucket_iter in self.class_specs}

        # Check if we've already broken probability bounds.
        for label, label_prob in label_sum_probs.items():
            if label_prob > 1:
                raise InfeasibleWordRandomnessError("Assigning minimum probability (" + str(word_prob_bounds[label][0]) + ") to all words with label \""
                    + label + "\" rresults in a probability of " + str(label_prob) + ", which is greater than 1.",
                    None) #TODO Make proper count

        # Add probabilites to appropriate classes (up to beta per word) without assigning more than 1 over each label.
        for label in label_func.labels:
            print(label)
            for bucket_iter in range(self.num_buckets):
                class_id = (label, bucket_iter)
                class_count = self.class_specs[class_id].language_size(tolerance=counting_tol, confidence=counting_conf, seed=random.getrandbits(32))
                print(class_count)
                new_cost = min((1 + counting_tol) * word_prob_bounds[label][1] * class_count, 1 - label_sum_probs[label])

                conditional_weights[class_id] = new_cost

                # Update label sum probability
                label_sum_probs[label] = sum([prob for ((l, _), prob) in conditional_weights.items() if l == label])

                # If we assigned all probability, terminate early
                if label_sum_probs[label] == 1:
                    break

        # Check if we've now broken probability bounds
        for label, label_prob in label_sum_probs.items():
            if label_prob != 1:
                raise InfeasibleWordRandomnessError("Assigning maximum probability (" + str(word_prob_bounds[label][1]) + ") to all words with label \""
                    + label + "\" results in a probability of " + str(label_prob) + ", which is less than 1.",
                    None) #TODO Make proper count

        # Calculate conditional expected costs
        conditional_costs = {}
        for label in label_func.labels:
            conditional_costs[label] = sum([conditional_weights[(label,bucket_iter)]*self.bucket_bounds_list[bucket_iter][0] for bucket_iter in range(self.num_buckets)])

        # Now calculate marginal weights.
        marginal_weights = {}

        u = math.floor((1 - len(label_func.labels)*label_prob_bounds[0])/(label_prob_bounds[1] - label_prob_bounds[0]))

        for label_iter, label in enumerate(sorted(label_func.labels, key=lambda x: conditional_costs[x])):
            if label_iter < u:
                marginal_weights[label] = (label_prob_bounds[1])
            elif label_iter == u:
                marginal_weights[label] = (1 - label_prob_bounds[1]*u - label_prob_bounds[0]*(len(label_func.labels) - u - 1))
            else:
                marginal_weights[label] = (label_prob_bounds[0])

        min_expected_cost = sum([marginal_weights[label] * conditional_costs[label] for label in label_func.labels])

        # Restore old state
        random.setstate(old_state)

        # Store improvisation values.
        self.class_keys = [(label, bucket) for label in sorted(label_func.labels) for bucket in range(self.num_buckets)]
        self.class_probabilities = [marginal_weights[label]*conditional_weights[(label,cost)] for label,cost in self.class_keys]

        # Checks that this improviser is feasible. If not raise an InfeasibleImproviserError.
        if len(feasible_labels) < (1/label_prob_bounds[1]) or (label_prob_bounds[0] != 0 and len(feasible_labels) > (1/label_prob_bounds[0])):
            if label_prob_bounds[0] == 0:
                inv_min_label_prob = float("inf")
            else:
                inv_min_label_prob = 1/label_prob_bounds[0]

            raise InfeasibleLabelRandomnessError("Violation of condition 1/label_prob_bounds[1] <= len(feasible_labels) <= 1/label_prob_bounds[0]. Instead, " \
                                            + str(1/label_prob_bounds[1]) + " <= " + str(len(feasible_labels)) + " <= " + str(inv_min_label_prob), len(feasible_labels))

        if min_expected_cost > cost_bound:
            raise InfeasibleCostError("Greedy construction does not satisfy cost_bound, meaning no improviser can."\
                                      + " Minimum expected cost was " + str(min_expected_cost) + ".", min_expected_cost)

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("ApproxLQCI distribution calculation completed. Wallclock Runtime: " + wall_time)

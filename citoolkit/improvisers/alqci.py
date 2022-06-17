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


class _ALQCIBase(Improviser):
    """ The base class for the ALQCI class.

    When this class is intialized, all the label/cost class specs are
    created and stored in the class_specs dictionary attribute which matches
    all (label,cost) tuples to the spec that recognizes words with that label
    and cost.

    If lazy_counting is set to False, then all label/cost class specs are also
    counted.

    All child classes must then initialize two list attributes: class_keys
    which contains a (label,cost) tuple for every label/cost class to be sampled
    from, and class_probabilities which contains the probability for the label/cost
    class at the same index in class_keys.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a natural number cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param bucket_ratio: The ratio between successive buckets. Must be greater than 1.
    :param counting_tol: The desired counting tolerance.
    :param sampling_tol: The desired sampling tolerance.
    :param conf: The desired confidence in the counting/sampling. Used with the number
        of feasible buckets to determine counting_conf, which is the confidence we need
        from each bucket we actually count.
    :param seed: Random seed for reproducible results.
    :param num_threads: Number of threads to be used when computing improviser.
    :param verbose: If verbose is enabled, information will be printed to STDOUT.
    :param lazy_counting: Whether or not to count lazily. If true, all spec counts will
        be computed after spec construction.
    """
    def __init__(self, hard_constraint: ApproxSpec, cost_func: ApproxCostFunc, label_func: ApproxLabelFunc, \
                bucket_ratio:float , counting_tol: float, sampling_tol: float, conf: float, \
                seed: int, num_threads: int, verbose: bool, lazy_counting: bool) -> None:
        ## Initialization ##
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

        ## Spec Construction ##
        # Compute spec for each label/cost class.
        if self.verbose:
            start_time = time.time()
            cit_log("Beginning compound Spec construction. Using " + str(num_threads) + " thread(s).")

        class_ids = set()

        for label in label_func.labels:
            for bucket_iter, bucket_bounds in enumerate(self.bucket_bounds_list):
                class_ids.add((label, bucket_iter))

        if num_threads <= 1:
            cpu_time = "N/A"
            for class_id in class_ids:
                label, bucket_iter = class_id
                bucket_bounds = self.bucket_bounds_list[bucket_iter]

                # Create abstract spec
                self.class_specs[class_id] = hard_constraint & label_func.realize(label) & cost_func.realize(*bucket_bounds)

                # Try to make the spec explicit if possible
                try:
                    self.class_specs[class_id] = self.class_specs[class_id].explicit()
                except NotImplementedError:
                    pass
        else:
            # Multiple threads, so create wrapper and thread pool and map abstract specs before
            # resaving explicit specs.

            with Pool(self.num_threads) as pool:
                # Helper function for pool.map
                def explicit_wrapper(class_id):
                    process_start_time = time.process_time()

                    label, bucket_iter = class_id
                    bucket_bounds = self.bucket_bounds_list[bucket_iter]

                    # Create abstract spec
                    spec = hard_constraint & label_func.realize(label) & cost_func.realize(*bucket_bounds)

                    # Try to make the spec explicit if possible
                    try:
                        spec = spec.explicit()
                    except NotImplementedError:
                        pass

                    return (class_id, spec, time.process_time() - process_start_time)

                pool_output = pool.map(explicit_wrapper, class_ids)

                # Extract relevant info from pool_output
                cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))

                self.class_specs = {class_id: class_spec for class_id, class_spec, _ in pool_output}

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("Compound Spec construction completed. Wallclock Runtime: " + wall_time + "  CPU Runtime: " + cpu_time)

        ## Counting Confidence Calculation ##
        # Determine counting confidence needed
        feasible_buckets = sum([spec.feasible for spec in self.class_specs.values()])

        self.counting_conf = 1 - math.pow((1-conf), 1/feasible_buckets)

        if self.verbose:
            cit_log("Total Buckets: " + str(len(self.class_specs)))
            cit_log("Feasible Buckets: " + str(feasible_buckets))
            cit_log("Desired Confidence " + str(conf) + " requires counting confidence of " + str(self.counting_conf))

        ## Spec Counting ##
        if not lazy_counting:
            # Count the language size for each spec if precount is enabled.
            if self.verbose:
                start_time = time.time()
                cit_log("Beginning language size counting. Using " + str(num_threads) + " thread(s).")

            if num_threads <= 1:
                cpu_time = "N/A"
                # 1 thread, so compute all sizes iteratively.
                for spec in self.class_specs.values():
                    spec.language_size(tolerance=counting_tol, confidence=self.counting_conf, seed=random.getrandbits(32))
            else:
                # Multiple threads, so create wrapper and thread pool and map specs before
                # resaving specs containing cached language sizes.

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

                        spec.language_size(tolerance=counting_tol, confidence=self.counting_conf, seed=random_seeds[class_id])

                        return (class_id, spec, time.process_time() - process_start_time)

                    pool_output = pool.map(count_wrapper, class_ids)

                    # Extract relevant info from pool_output
                    cpu_time = "{:.4f}".format(sum([runtime for _,_,runtime in pool_output]))

                    self.class_specs = {class_id: class_spec for class_id, class_spec, _ in pool_output}

            if self.verbose:
                wall_time = "{:.4f}".format(time.time() - start_time)
                cit_log("Language size counting completed. Wallclock Runtime: " + wall_time + "  CPU Runtime: " + cpu_time)

        ## Cleanup ##
        # Restore random state
        random.setstate(old_state)

    def improvise(self, seed: int =None) -> tuple[str,...]:
        """ Improvise a single word. Base class must populate self.class_probabilities
        before this method is called.

        :returns: A single improvised word.
        """
        if (self.class_probabilities is None) or (self.class_keys is None):
            raise Exception("Improvise function was called without first computing self.class_probabilities or self.class_keys.")

        if seed is None:
            seed = random.getrandbits(32)

        # Cache the random state, set it to the seed, pick a target class, sample, and finally, restore the random state.
        old_state = random.getstate()
        random.seed(seed)

        target_class = random.choices(population=self.class_keys, weights=self.class_probabilities, k=1)[0]

        sample = self.class_specs[target_class].sample(tolerance=self.sampling_tol, seed=random.getrandbits(32))

        random.setstate(old_state)

        return sample

class ALQCI(_ALQCIBase):
    """ An improviser for the Approximate Labelled Quantitative Control Improvisation problem.

    When this class is intialized, all the label/cost class specs are
    created and stored in the class_specs dictionary attribute which matches
    all (label,cost) tuples to the spec that recognizes words with that label
    and cost.

    If lazy_counting is set to False, then all label/cost class specs are also
    counted.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a natural number cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param cost_bound: The desired maximum cost that should be returned. The expected
        cost of the resulting improvising distribution is guaranteed to be less than
        or equal to (cost_bound * bucket_ratio) with confidence conf.
    :param label_prob_bounds: A tuple containing lower and upper bounds on the
        marginal probability with which we can generate a word with a particular label.
        The marginal probability of each label in the resulting improvising distribution
        is guaranteed to be in this range with confidence conf.
    :param word_prob_bounds: A dictionary mapping each label in label_func to a tuple. Each
        tuple contains a desired lower and upper bound on the conditional probability of selecting
        a word with the associated label conditioned on the fact that we do select a word
        with that label. The conditional probability of each word is guaranteed to be between (inclusive)
        (word_prob_bounds[label][0] / ((1 + counting_tol)**2 (1 + sampling_tol))) and
        (word_prob_bounds[label][1] * ((1 + counting_tol)**2 (1 + sampling_tol)))
    :param bucket_ratio: The ratio between successive buckets. Must be greater than 1.
    :param counting_tol: The desired counting tolerance. Must be greater than 0.
    :param sampling_tol: The desired sampling tolerance. Must be greater than 0.
    :param conf: The desired confidence in the counting/sampling. Used with the number
        of feasible buckets to determine counting_conf, which is the confidence we need
        from each bucket we actually count. The guarantees for the improviser
        will hold with confidence (1 - conf). Must be in the range (0,1)
    :param seed: Random seed for reproducible results.
    :param num_threads: Number of threads to be used when computing improviser.
    :param verbose: If verbose is enabled, information will be printed to STDOUT.
    :param lazy_counting: Whether or not to count lazily. If true, all spec counts will
        be computed after spec construction. If false, all spec counts will be computed when needed.
        If None, the improviser will decide based on a heuristic.
    """
    def __init__(self, hard_constraint: ApproxSpec, cost_func: ApproxCostFunc, label_func: ApproxLabelFunc, \
                 cost_bound: float, label_prob_bounds: tuple[float, float], word_prob_bounds: dict[str, tuple[float, float]], \
                 bucket_ratio: float, counting_tol: float, sampling_tol: float, conf: float, \
                 seed=None, num_threads: int =1, verbose: bool =False, lazy_counting: Optional[bool] =None) -> None:
        ## Initialization ##
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

        if bucket_ratio <= 1:
            raise ValueError("The bucket_ratio parameter must be a number >= 1.")

        if counting_tol <= 0:
            raise ValueError("The counting_tol parameter must be a number > 0.")

        if sampling_tol <= 0:
            raise ValueError("The sampling_tol parameter must be a number > 0.")

        if conf <= 0 or conf >= 1:
            raise ValueError("The conf parameter must be a number in the range (0,1).")

        # If lazy_counting is set to the default None, set it to either True or False.
        # The heuristic is simple, if all words have zero minimum probability then count
        # lazily.
        if lazy_counting is None:
            if max([word_prob_bounds[label][0] for label in label_func.labels]) == 0:
                lazy_counting = True
            else:
                lazy_counting = False

        # Cache the random state and set seed.
        old_state = random.getstate()
        random.seed(seed)

        # Initialize LQCI base class.
        super().__init__(hard_constraint, cost_func, label_func, \
                         bucket_ratio, counting_tol, sampling_tol, conf, \
                         seed, num_threads, verbose, lazy_counting)

        if self.verbose:
            start_time = time.time()
            cit_log("Beginning ALQCI distribution calculation.")

        ## Distribution Computation ##
        feasible_labels = set()

        for label in label_func.labels:
            for bucket_iter, _ in enumerate(self.bucket_bounds_list):
                if self.class_specs[(label, bucket_iter)].feasible:
                    feasible_labels.add(label)
                    break
            if label_prob_bounds[0] > 0 and label not in feasible_labels:
                raise InfeasibleLabelRandomnessError("No strings are labelled with label '" + label + "', but label_prob_bounds[0] > 0.", 0)

        # Initialize conditional weights to alpha for each word and sum up current marginal label probabilities.
        conditional_weights = {}

        for class_id in self.class_specs:
            label, bucket_iter = class_id
            # If alpha is 0, we can just set everything to 0. Otherwise determine probability based on class count.
            if word_prob_bounds[label][0] == 0:
                conditional_weights[class_id] = 0
            else:
                class_count = self.class_specs[class_id].language_size(tolerance=counting_tol, confidence=self.counting_conf, seed=random.getrandbits(32))
                conditional_weights[class_id] = class_count * (word_prob_bounds[label][0]/(1+counting_tol))

        label_sum_probs = {label:sum([prob for ((l, _), prob) in conditional_weights.items() if l == label]) for label, bucket_iter in self.class_specs}

        # Check if we've already broken probability bounds.
        for label, label_prob in label_sum_probs.items():
            if label_prob > 1:
                raise InfeasibleWordRandomnessError("Assigning minimum probability (" + str(word_prob_bounds[label][0]) + ") to all words with label \""
                    + label + "\" results in a probability of " + str(label_prob) + ", which is greater than 1.",
                    None)

        # Add probabilites to appropriate classes (up to beta per word) without assigning more than 1 over each label.
        if lazy_counting and self.num_threads > 1 and len(label_func.labels) > 1:
            # Probabilities are not yet computed and we have several threads available and several labels to compute. Advantageous to multithread.
            with Pool(self.num_threads) as pool:
                # Helper function that counts buckets for a label and assigns probability.
                def assign_beta(func_input):
                    label, label_specs, cond_label_weights, seed = func_input

                    for bucket_iter in range(self.num_buckets):
                        # If we assigned all probability, terminate early
                        if sum(cond_label_weights.values()) == 1:
                            break

                        class_count = label_specs[bucket_iter].language_size(tolerance=counting_tol, confidence=self.counting_conf, seed=seed)

                        # Assign at least beta probability to all words in this bucket (accounting for counting errors), or
                        # all remaining probability (plus what we have already assigned), whichever is smaller.
                        new_cost = min((1 + counting_tol) * word_prob_bounds[label][1] * class_count, 1 - sum(cond_label_weights.values()) + cond_label_weights[cost])

                        cond_label_weights[bucket_iter] = new_cost

                    return (label, label_specs, cond_label_weights)

                # Create helper function inputs and run them through assign_beta
                func_inputs = []
                for label in label_func.labels:
                    label_specs = {b:(spec) for (l, b), spec in self.class_specs.items() if l == label}
                    cond_label_weights = {b:weight for (l, b), weight in conditional_weights.items() if l == label}
                    seed = random.getrandbits(32)
                    func_inputs.append((label, label_specs, cond_label_weights, seed))

                pool_output = pool.map(assign_beta, func_inputs)
                pool_dict = {label:(specs, weights) for label, specs, weights in pool_output}

                # Parse pool outputs back into appropriate variables
                for class_id in self.class_specs:
                    label, bucket_iter = class_id
                    self.class_specs[class_id] = pool_dict[label][0][bucket_iter]

                conditional_weights = {(l,b):pool_dict[l][1][b] for l,b in self.class_specs}

        else:
            # Probabilities already computed or only one thread, so no point in multithreading.
            for label in label_func.labels:
                for bucket_iter in range(self.num_buckets):
                    # If we assigned all probability, terminate early
                    if label_sum_probs[label] == 1:
                        break

                    class_id = (label, bucket_iter)
                    class_count = self.class_specs[class_id].language_size(tolerance=counting_tol, confidence=self.counting_conf, seed=random.getrandbits(32))

                    conditional_weights[class_id] =  min((1 + counting_tol) * word_prob_bounds[label][1] * class_count, 1 - label_sum_probs[label] + conditional_weights[class_id])

                    # Update label sum probability
                    label_sum_probs[label] = sum([prob for ((l, _), prob) in conditional_weights.items() if l == label])

        # Check if we've now broken probability bounds
        for label, label_prob in label_sum_probs.items():
            if label_prob != 1:
                raise InfeasibleWordRandomnessError("Assigning maximum probability (" + str(word_prob_bounds[label][1]) + ") to all words with label \""
                    + label + "\" results in a probability of " + str(label_prob) + ", which is less than 1.", None)

        # Calculate conditional expected costs
        conditional_costs = {}
        for label in label_func.labels:
            conditional_costs[label] = sum([conditional_weights[(label,bucket_iter)]*self.bucket_bounds_list[bucket_iter][0] for bucket_iter in range(self.num_buckets)])

        # Calculate marginal weights.
        marginal_weights = {}

        # Calculate u, the number of labels that can get maximum probability
        if label_prob_bounds[1] == label_prob_bounds[0]:
            # In this case all labels must get the same probability, which will be handled by case 3 of the if statement below.
            u = 0
        else:
            u = math.floor((1 - len(label_func.labels)*label_prob_bounds[0])/(label_prob_bounds[1] - label_prob_bounds[0]))

        for label_iter, label in enumerate(sorted(label_func.labels, key=lambda x: conditional_costs[x])):
            if label_iter < u:
                marginal_weights[label] = (label_prob_bounds[1])
            elif label_iter == u:
                marginal_weights[label] = (1 - label_prob_bounds[1]*u - label_prob_bounds[0]*(len(label_func.labels) - u - 1))
            else:
                marginal_weights[label] = (label_prob_bounds[0])

        min_expected_cost = sum([marginal_weights[label] * conditional_costs[label] for label in label_func.labels])

        if self.verbose:
            cit_log("Lower Bound on Calculated Distribution Cost: " + str(min_expected_cost))

        # Store improvisation values.
        self.class_keys = [(label, bucket) for label in sorted(label_func.labels) for bucket in range(self.num_buckets)]
        self.class_probabilities = [marginal_weights[label]*conditional_weights[(label,cost)] for label,cost in self.class_keys]

        if self.verbose:
            wall_time = "{:.4f}".format(time.time() - start_time)
            cit_log("ALQCI distribution calculation completed. Wallclock Runtime: " + wall_time)

        ## Cleanup ##
        # Restore old state
        random.setstate(old_state)

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

import z3
import logging
import time
import math
import random
import json
import copy
import itertools

from citoolkit.specifications.z3_formula import Z3Formula

from citoolkit.labellingfunctions.labelling_z3_formula import Z3LabelFormula
from citoolkit.costfunctions.cost_z3_formula import Z3CostFormula

from citoolkit.improvisers.approx_labelled_quantitative_ci import ApproxLabelledQuantitativeCI

class Vehicle:
    def __init__(self, name, id, type, mass, feasible_per_poi, survey_per_sortie_per_poi, travel_time_per_sortie_per_poi):
        self.name = name
        self.id = id
        self.mass = mass
        self.feasible_per_poi = feasible_per_poi
        self.survey_per_sortie_per_poi = survey_per_sortie_per_poi
        self.travel_time_per_sortie_per_poi = travel_time_per_sortie_per_poi
        self.type = type
        assert len(survey_per_sortie_per_poi) == len(travel_time_per_sortie_per_poi)

    def __str__(self):
        return str((self.name, self.id, self.mass, self.feasible_per_poi, self.survey_per_sortie_per_poi, self.travel_time_per_sortie_per_poi, self.type))

    def __repr__(self):
        return str(self)

class Poi:
    def __init__(self, name, id, area):
        self.name = name
        self.id = id
        self.area = area

    def __str__(self):
        return str((self.name, self.id, self.area))

    def __repr__(self):
        return str(self)

class SimpleOMTConstrainedPlanning():
    def __init__(self, vehicles, pois, mass_bound, time_bound, vec_bits=8):
        assert vec_bits >= 6

        # Initialize class attributes
        self.vec_bits = vec_bits

        self.vehicles, self.pois, self.mass_bound, self.time_bound = self.discretize_inputs(vehicles, pois, mass_bound, time_bound)

        self.base_constraint = self.create_base_constraint()

        self.main_variables = {self.var_sortie_count(veh, poi) for veh in self.vehicles for poi in self.pois}

    def feasible(self):
        s = z3.Solver()

        s.add(self.base_constraint)

        if s.check() == z3.sat:
            return True
        else:
            return False

    def get_model(self):
        s = z3.Solver()

        s.add(self.base_constraint)

        assert s.check() == z3.sat

        return s.model()

    def discretize_inputs(self, vehicles, pois, mass_bound, time_bound):
        # Determine maximum single values for mass, time, area
        max_mass = max(mass_bound, max([veh.mass for veh in vehicles]))
        max_time = max(time_bound, max([max(veh.travel_time_per_sortie_per_poi.values()) for veh in vehicles]))
        max_area = max(max([max(veh.survey_per_sortie_per_poi.values()) for veh in vehicles]), max([poi.area for poi in pois]))

        # Create a mapping function for each unit from value to bitvec value
        max_bitvec_val = 2**self.vec_bits - 1
        conv_funcs = {}
        self.conv_funcs = conv_funcs
        conv_funcs["mass"] = lambda x: math.ceil(x/max(1,((max_mass+1)/max_bitvec_val)))
        conv_funcs["time"] = lambda x: math.ceil(x/max(1,((max_time+1)/max_bitvec_val)))
        conv_funcs["area"] = lambda x: math.floor(x/max(1,((max_area+1)/max_bitvec_val)))

        # Sanity check to make sure we avoid overflow
        assert conv_funcs["mass"](max_mass) <= float(max_bitvec_val)
        assert conv_funcs["time"](max_time) <= float(max_bitvec_val)
        assert conv_funcs["area"](max_area) <= float(max_bitvec_val)

        # Convert vehicles
        d_vehicles = []

        for veh in vehicles:
            d_survey_area_map = {key:conv_funcs["area"](val) for key,val in veh.survey_per_sortie_per_poi.items()}
            d_travel_time_map = {key:conv_funcs["time"](val) for key,val in veh.travel_time_per_sortie_per_poi.items()}
            d_vehicles.append(Vehicle(veh.name, veh.id, veh.type, conv_funcs["mass"](veh.mass), veh.feasible_per_poi, d_survey_area_map, d_travel_time_map))

        # Convert pois
        d_pois = []

        for poi in pois:
            d_pois.append(Poi(poi.name, poi.id, conv_funcs["area"](poi.area)))

        return d_vehicles, d_pois, conv_funcs["mass"](mass_bound), conv_funcs["time"](time_bound)

    def create_base_constraint(self):
        # Initialize base constraint
        base_constraint = True

        ## (Constraint 1) Require that the sum of the mass of used vehicles is less than max_bound
        ## AND that the sum does not overflow at any point.
        #base_constraint = z3.And(base_constraint, self.create_constraint_1())

        ## (Constraint 2) Require that vehicles only make sorties to feasible pois
        base_constraint = z3.And(base_constraint, self.create_constraint_2())

        ## (Constraint 3) Require that if a vehicle's sortie count for any POI is larger than 0, that
        ## the vehicle used boolean is true.
        base_constraint = z3.And(base_constraint, self.create_constraint_3())

        ## (Constraint 4) Require that the sum of the area covered by all sorties to a poi covers the
        ## entire poi.
        base_constraint = z3.And(base_constraint, self.create_constraint_4())

        ## (Constraint 5) Require that for each vehicle, the sum of time spent surveying each waypoint
        ## is less than or equal to the time bound.
        base_constraint = z3.And(base_constraint, self.create_constraint_5())

        ## (Label Constraint) Create a bitvector that labels plans based on how many veh are used as follows:
        ## [1-3] vehicles -> 0
        ## [4-6] vehicles -> 1
        ## [7-10] vehicles -> 2
        base_constraint = z3.And(base_constraint, self.create_label_constraint())

        return base_constraint

    def create_constraint_1(self):
        ## (Constraint 1) Require that the sum of the mass of used vehicles is less than max_bound
        ## AND that the sum does not overflow at any point.
        sum_value = z3.BitVecVal(0, self.vec_bits)
        sum_constraint = True

        for veh in self.vehicles:
            # Require that the contributed mass of a vehicle be zero if the vehicle is not used
            # and the vehicles mass otherwise.
            sum_constraint = z3.And(sum_constraint, z3.Implies(z3.Not(self.var_veh_used(veh)), (self.var_veh_contributed_weight(veh) == 0)))
            sum_constraint = z3.And(sum_constraint, z3.Implies(self.var_veh_used(veh), (self.var_veh_contributed_weight(veh) == veh.mass)))

            # Require that adding the current vehicles contributed mass doesn't overflow the bitvector
            sum_constraint = z3.And(sum_constraint, z3.BVAddNoOverflow(sum_value, self.var_veh_contributed_weight(veh), False))

            # Accumulate the current sum of mass contributed by vehicles
            sum_value += self.var_veh_contributed_weight(veh)

        # Require that the sum of contributed mass be less than the mass bound
        sum_constraint = z3.And(sum_constraint, z3.ULT(sum_value, self.mass_bound))

        return sum_constraint

    def create_constraint_2(self):
        ## (Constraint 2) Require that vehicles only make sorties to feasible pois
        feas_constraint = True

        for poi in self.pois:
            for veh in self.vehicles:
                if not veh.feasible_per_poi[poi.id]:
                    feas_constraint = z3.And(feas_constraint, self.var_sortie_count(veh, poi) == 0)
                feas_constraint = z3.And(feas_constraint, z3.ULE(self.var_sortie_count(veh, poi), 1))

        return feas_constraint

    def create_constraint_3(self):
        ## (Constraint 3) Require that if a vehicle's sortie count for any POI is larger than 0, that
        ## the vehicle used boolean is true.
        sortie_constraints = True

        sum_of_bools = z3.BitVecVal(0, self.vec_bits)

        # Check the sortie count for each vehicle
        for veh in self.vehicles:
            veh_has_sorties = False

            for poi in self.pois:
                veh_has_sorties = z3.Or(veh_has_sorties, z3.UGT(self.var_sortie_count(veh, poi), 0))

            sortie_constraints = z3.And(sortie_constraints, veh_has_sorties == self.var_veh_used(veh))

            sum_of_bools = sum_of_bools + z3.If(self.var_veh_used(veh), z3.BitVecVal(1, self.vec_bits), z3.BitVecVal(0, self.vec_bits))

        #TODO Fix hard cap on vehs.
        sortie_constraints = z3.And(sortie_constraints, sum_of_bools == z3.BitVec("NumVehs", self.vec_bits))
        sortie_constraints = z3.And(sortie_constraints, z3.BitVec("NumVehs", self.vec_bits) == z3.BitVecVal(2, self.vec_bits))

        return sortie_constraints

    def create_constraint_4(self):
        ## (Constraint 4) Require that the sum of the area covered by all sorties to a poi covers the
        ## entire poi.
        area_constraints = True

        # Check that area is covered for each poi
        for poi in self.pois:
            # Sum area covered by each vehicle and its sorties for this poi.
            # Keep track of overflow, as if we overflow we know we
            # have satisfied the area.
            area_sum = z3.BitVecVal(0, self.vec_bits)
            no_overflow = True

            for veh in self.vehicles:
                # OPTIMIZATION: Require that if we've already overflowed or exceeded the area count, that no more sorties are sent.
                area_done = z3.Or(z3.Not(no_overflow), z3.UGT(area_sum, poi.area))
                area_constraints = z3.And(area_constraints, z3.Implies(area_done, self.var_sortie_count(veh, poi) == 0))

                # Compute product of sortie count and area covered per sortie while checking for overflow.
                sortie_area = z3.BitVecVal(veh.survey_per_sortie_per_poi[poi.id], self.vec_bits)

                no_overflow = z3.And(no_overflow, z3.BVMulNoOverflow(self.var_sortie_count(veh, poi), sortie_area, False))

                sortie_prod = self.var_sortie_count(veh, poi) * sortie_area

                # Add sortie product to area_sum while checking for overflow.
                no_overflow = z3.And(no_overflow, z3.BVAddNoOverflow(area_sum, sortie_prod, False))

                area_sum = area_sum + sortie_prod

            # Require that we have met our area requirements or that we overflowed (which implies we met our area requirements)
            enough_area_surveyed = z3.UGT(area_sum, poi.area)
            overflowed_area = z3.Not(no_overflow)

            good_survey = z3.Or(enough_area_surveyed, overflowed_area)

            area_constraints = z3.And(area_constraints, good_survey)

        return area_constraints

    def create_constraint_5(self):
        ## (Constraint 5) Require that for each vehicle, the sum of time spent surveying all waypoints
        ## is less than or equal to the time bound.
        time_constraints = True

        # Create mission time bitvector, which will be set to the largest time value over all vehicles.
        mission_time_var = self.var_cost()

        mission_time_equal_one = False
        mission_time_larger = True

        # Check that time bound is respected for each vehicle
        for veh in self.vehicles:
            # Sum time taken by each vehicle and its sorties.
            # Keep track of overflow, as if we overflow we know we
            # have violated the time bound.
            time_sum = z3.BitVecVal(0, self.vec_bits)
            no_overflow = True

            for poi in self.pois:
                # Compute product of sortie count and time taken per sortie while checking for overflow.
                sortie_time = z3.BitVecVal(veh.travel_time_per_sortie_per_poi[poi.id], self.vec_bits)

                no_overflow = z3.And(no_overflow, z3.BVMulNoOverflow(self.var_sortie_count(veh, poi), sortie_time, False))

                sortie_prod = self.var_sortie_count(veh, poi) * sortie_time

                # Add sortie product to area_sum while checking for overflow.
                no_overflow = z3.And(no_overflow, z3.BVAddNoOverflow(time_sum, sortie_prod, False))

                time_sum = time_sum + sortie_prod

            # Require that we are under our time requirements and that we have not overflowed
            # (which would imply exceeding our time requirement)
            under_time_bound = z3.ULT(time_sum, self.time_bound) # z3.ULT(z3.BitVec("TimeBound"), self.time_bound)
            good_survey = z3.And(under_time_bound, no_overflow)

            # Require that mission_time_var be at least as large as time_sum, and that mission_time_var be equal
            # to one of the time sums.
            mission_time_equal_one = z3.Or(mission_time_equal_one, mission_time_var == time_sum)
            mission_time_larger = z3.And(mission_time_larger, z3.UGE(mission_time_var, time_sum))

            time_constraints = z3.And(time_constraints, good_survey)

        time_constraints = z3.And(time_constraints, mission_time_equal_one)
        time_constraints = z3.And(time_constraints, mission_time_larger)

        return time_constraints

    def create_label_constraint(self):
        ## (Label Constraint) Create a bitvector that labels plans as follows:
        ## [1-3] vehicles -> 0
        ## [4-6] vehicles -> 1
        ## [7-10] vehicles -> 2

        # Create a value containing the number of vehicles used.
        # used_vehicles = z3.BitVecVal(0, self.vec_bits)

        # for veh in self.vehicles:
        #     used_vehicles = used_vehicles + z3.If(self.var_veh_used(veh), z3.BitVecVal(1, self.vec_bits), z3.BitVecVal(0, self.vec_bits))

        # label_var = self.var_label()

        # label_1_constraint = z3.And(z3.ULE(1, used_vehicles), z3.ULE(used_vehicles, 5)) == (label_var == 0)

        # label_2_constraint = z3.And(z3.ULE(6, used_vehicles), z3.ULE(used_vehicles, 10)) == (label_var == 1)

        # label_3_constraint = z3.And(z3.ULE(11, used_vehicles), z3.ULE(used_vehicles, 15)) == (label_var == 2)

        # label_constraints = z3.And(label_1_constraint, label_2_constraint, label_3_constraint)

        check_duplicates = False
        label_var = self.var_label()

        for veh_1, veh_2 in itertools.combinations(self.vehicles, 2):
            if veh_1.type == veh_2.type:
                #print(veh_1.id, veh_2.id)
                check_duplicates = z3.Or(check_duplicates, z3.And(self.var_veh_used(veh_1), self.var_veh_used(veh_2)))

        label_constraints = z3.If(check_duplicates, label_var == 2, label_var == 1)

        return label_constraints

    def var_veh_used(self, veh):
        return z3.Bool("VehicleUsed_" + str(veh.id))

    def var_sortie_count(self, veh, poi):
        return z3.BitVec("SortieCount_" + str(veh.id) + "_" + str(poi.id), self.vec_bits)

    def var_veh_contributed_weight(self, veh):
        return z3.BitVec("VehicleContribWeight_" + str(veh.id), self.vec_bits)

    def var_label(self):
        return z3.BitVec("LabelVar", self.vec_bits)

    def var_cost(self):
        return z3.BitVec("CostVar", self.vec_bits)


def plan(vehicles, pois):
    OPT_RANGE = 1.01 # How close we want to be when calculating optimal time
    TIME_UPPER_BOUND = 1e6 # Highest value at which to start our optimal time search
    TIME_RANGE = 4 # We consider all plans with time at most TIME_RANGE * optimal_time
    EXP_TIME_RANGE = 2 # We want the expected time of our plans to be within this of optimal_time

    lo_time = 0
    hi_time = TIME_UPPER_BOUND

    # Get within OPT_RANGE of the minimum time design.
    while hi_time - lo_time > OPT_RANGE * lo_time:
        mid_time =  (hi_time + lo_time)/2

        print("mid_time", mid_time)

        planner = SimpleOMTConstrainedPlanning(vehicles, pois, mass_bound=1, time_bound= mid_time)

        if planner.feasible():
            hi_time = mid_time
        else:
            lo_time = mid_time

    print("Best time:", hi_time)

    # Create hard constraint
    planner = SimpleOMTConstrainedPlanning(vehicles, pois, mass_bound=2500, time_bound=hi_time*TIME_RANGE, vec_bits=6)
    base_constraint = planner.base_constraint
    main_variables = [planner.var_sortie_count(veh, poi) for veh in planner.vehicles for poi in planner.pois]
    main_variables.append(planner.var_label())
    main_variables.append(z3.BitVec("NumVehs", planner.vec_bits))
    hard_constraint = Z3Formula(base_constraint, main_variables, lazy_bool_spec=True)

    # Create cost and label functions
    cost_func = Z3CostFormula(planner.var_cost())
    label_map = {"NoDuplicates": 1, "YesDuplicates": 2}
    label_func = Z3LabelFormula(label_map, planner.var_label())

    for label in label_func.labels:
        spec = (hard_constraint & label_func.realize(label)).explicit()
        print(label + " Feasible:", spec.feasible)

    # Create improvier
    opt_encoded_time = planner.conv_funcs["time"](hi_time)
    cost_bound = opt_encoded_time * EXP_TIME_RANGE
    word_prob_bounds = {label:(0,5e-1z) for label in label_func.labels}

    improviser = ApproxLabelledQuantitativeCI(hard_constraint, cost_func, label_func, \
                 cost_bound, (0.5,1), word_prob_bounds, \
                 1.05, 1, 0.2, 15, \
                 num_threads=10, lazy_counting=True, verbose=True)

    print(improviser.class_keys)
    print(improviser.class_probabilities)

    samples = {i:improviser.improvise() for i in range(4)}

    vehicle_dict = {veh.id:veh for veh in vehicles}

    for i, sample in samples.items():
        print("Sample #" + str(i))
        for key, val in sample.items():
            if val != 0 and key[:12] == "SortieCount_":
                print("  " + str(vehicle_dict[int(key[12])].type) + (" (Copy)" if vehicle_dict[int(key[12])].type != int(key[12]) else "       ") + " -> " + key[14])

def import_data(job_file, results_file, target_env):
    # Load JSONS
    with open(job_file) as f:
        job_json = json.load(f)

    with open(results_file) as f:
        results_json = json.load(f)

    pois = [Poi("Poi_" + str(i), i, 1) for i in range(3)]

    vehicle_dict = {int(key): Vehicle("Veh_" + key, int(key), int(key), 0, {}, {}, {}) for key, veh in job_json["vehicles"].items()}

    # Fill out each vehicle's data
    for job_id, job_val in job_json["experiments"].items():
        if job_val["env_id"] != target_env:
            continue

        target_vehicle = vehicle_dict[job_val["veh_id"]]

        results_val = results_json[str(job_id)]

        for poi_id, poi_val in results_val.items():
            if poi_id == "error":
                assert poi_val == None
                continue

            target_vehicle.feasible_per_poi[int(poi_id)] = poi_val["elapsedTime"] != 120 #TODO Fix hack
            target_vehicle.survey_per_sortie_per_poi[int(poi_id)] = 2 #TODO Fix hardcoding
            target_vehicle.travel_time_per_sortie_per_poi[int(poi_id)] = poi_val["elapsedTime"]

    # Duplicate vehicles
    vehicle_list = []

    for veh in vehicle_dict.values():
        vehicle_list.append(veh)

        dup_veh = copy.deepcopy(veh)
        dup_veh.id = veh.id + len(vehicle_dict.values())

        vehicle_list.append(dup_veh)

    # Sanity check
    for veh in vehicle_list:
        assert list(veh.feasible_per_poi.keys()) == list(range(3))
        assert list(veh.survey_per_sortie_per_poi.keys()) == list(range(3))
        assert list(veh.travel_time_per_sortie_per_poi.keys()) == list(range(3))

    return vehicle_list, list(pois)

if __name__ == "__main__":
    random.seed("foobrbar")

    JOB_PATH = "input.json"
    RESULTS_PATH = "output.json"
    TARGET_ENV = 1
    MAX_VEHS = 25

    vehicles, pois = import_data(JOB_PATH, RESULTS_PATH, TARGET_ENV)

    for veh in vehicles:
        print(veh)
    print(pois)

    if len(vehicles) > MAX_VEHS:
        vehicles = random.sample(vehicles, MAX_VEHS)

    print(len(vehicles))
    assert len(vehicles) <= MAX_VEHS

    plan(vehicles, pois)

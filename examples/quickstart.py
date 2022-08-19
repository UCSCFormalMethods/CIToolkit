from fractions import Fraction
import z3

from citoolkit.specifications.dfa import Dfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa
from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.labellingfunctions.labelling_z3_formula import LabellingZ3Formula
from citoolkit.costfunctions.cost_z3_formula import CostZ3Formula
from citoolkit.improvisers.ci import CI
from citoolkit.improvisers.lqci import LQCI, MELQCI
from citoolkit.improvisers.alqci import ALQCI
from citoolkit.improvisers.improviser import (
    InfeasibleImproviserError,
    InfeasibleRandomnessError,
    InfeasibleCostError,
    InfeasibleSoftConstraintError,
    InfeasibleLabelRandomnessError,
    InfeasibleWordRandomnessError,
)

# Create our hard constraint, a Dfa object accepting only binary strings of length 1 to 6.
hc_alphabet = {"0", "1"}
hc_states = {"D0", "D1", "D2", "D3", "D4", "D5", "D6", "Sink"}
hc_accepting_states = {"D1", "D2", "D3", "D4", "D5", "D6"}
hc_start_state = "D0"

hc_transitions = {}
for symbol in hc_alphabet:
    hc_transitions[("D0", symbol)] = "D1"
    hc_transitions[("D1", symbol)] = "D2"
    hc_transitions[("D2", symbol)] = "D3"
    hc_transitions[("D3", symbol)] = "D4"
    hc_transitions[("D4", symbol)] = "D5"
    hc_transitions[("D5", symbol)] = "D6"
    hc_transitions[("D6", symbol)] = "Sink"
    hc_transitions[("Sink", symbol)] = "Sink"

hard_constraint = Dfa(hc_alphabet, hc_states, hc_accepting_states, hc_start_state, hc_transitions)

# Create our soft constraint, a Dfa object accepting only binary strings whose integer value is less than 8.
sc_alphabet = {"0", "1"}
sc_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Small", "Large"}
sc_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Small"}
sc_start_state = "Seen0"

sc_transitions = {}
for symbol in sc_alphabet:
    sc_transitions[("Seen0", symbol)] = "Seen1"
    sc_transitions[("Seen1", symbol)] = "Seen2"
    sc_transitions[("Seen2", symbol)] = "Seen3"

sc_transitions["Seen3", "0"] = "Small"
sc_transitions["Seen3", "1"] = "Large"
sc_transitions["Small", "0"] = "Small"
sc_transitions["Small", "1"] = "Large"
sc_transitions["Large", "0"] = "Large"
sc_transitions["Large", "1"] = "Large"

soft_constraint = Dfa(sc_alphabet, sc_states, sc_accepting_states, sc_start_state, sc_transitions)

# Create a CI improviser for our problem (with infeasible bounds).
length_bounds = (1,6)
epsilon = 0.33
prob_bounds = (0.005, 0.1)

ci_improviser = CI(hard_constraint, soft_constraint, length_bounds)

try:
    ci_improviser.parameterize(epsilon, prob_bounds)
except InfeasibleSoftConstraintError:
    pass

# Create a CI improviser for our problem (with feasible bounds).
length_bounds = (1,6)
epsilon = 0.45
prob_bounds = (0.0025, 0.15)

ci_improviser = CI(hard_constraint, soft_constraint, length_bounds)

ci_improviser.parameterize(epsilon, prob_bounds)

print("Printing some words from our CI Improviser:")
for _ in range(10):
    print(ci_improviser.improvise())
print()

# Create a labelling function for our problem
# First create the Dfa for our label function
lf_alphabet = {"0", "1"}
lf_states = {"Seen0", "Seen1", "Seen2",\
             "Seen3", "Seen4", "Seen5",\
             "Seen6+"}
lf_accepting_states = lf_states

lf_start_state = "Seen0"

lf_transitions = {}

for state in lf_states:
    lf_transitions[state, "0"] = state

lf_transitions["Seen0", "1"] = "Seen1"
lf_transitions["Seen1", "1"] = "Seen2"
lf_transitions["Seen2", "1"] = "Seen3"
lf_transitions["Seen3", "1"] = "Seen4"
lf_transitions["Seen4", "1"] = "Seen5"
lf_transitions["Seen5", "1"] = "Seen6+"
lf_transitions["Seen6+", "1"] = "Seen6+"

lf_dfa = Dfa(lf_alphabet, lf_states,\
             lf_accepting_states, lf_start_state,\
             lf_transitions)

# Second create a mapping from all accepting
# states to the appropriate label.
lf_label_map = {}

lf_label_map["Seen0"] = "Seen0-2"
lf_label_map["Seen1"] = "Seen0-2"
lf_label_map["Seen2"] = "Seen0-2"
lf_label_map["Seen3"] = "Seen3-4"
lf_label_map["Seen4"] = "Seen3-4"
lf_label_map["Seen5"] = "Seen5-6"
lf_label_map["Seen6+"] = "Seen5-6"

# Finally create the LabellingDfa
label_func = LabellingDfa(lf_dfa, lf_label_map)

# Create a cost function for our problem.
# First create the Dfa for our cost function.
cf_alphabet = {"0","1"}
cf_states = {"Start", "Sink"}
cf_accepting_states = set()

for depth in range(1,7):
    for cost in range(0, 2**depth):
        # Each state has two numbers in it. The first indicates
        # the number of symbols it's seen so far and the second
        # indicates the total cost it has accumulated.
        cf_states.add(f"State_{depth}_{cost}")
        cf_accepting_states.add(f"State_{depth}_{cost}")

cf_start_state = "Start"
cf_transitions = {}

# Insert transitions to Start/Sink node
cf_transitions["Start", "0"] = "State_1_0"
cf_transitions["Start", "1"] = "State_1_1"
cf_transitions["Sink", "0"] = "Sink"
cf_transitions["Sink", "1"] = "Sink"

# Insert remaining transition.
for depth in range(1,7):
    for cost in range(0, 2**depth):
        orig_state = f"State_{depth}_{cost}"

        for symbol in ["0", "1"]:
            if depth == 6:
                dest_state = "Sink"
            elif symbol == "0":
                dest_state = f"State_{depth+1}_{cost}"
            else:
                dest_state = f"State_{depth+1}_{cost+2**depth}"

            cf_transitions[orig_state, symbol] = dest_state

cf_dfa = Dfa(cf_alphabet, cf_states,\
             cf_accepting_states, cf_start_state,\
             cf_transitions)

# Second create a mapping from all the accepting
# states to the appropriate cost
cf_cost_map = {}

for depth in range(1,7):
    for cost in range(0, 2**depth):
        cf_cost_map[f"State_{depth}_{cost}"] = cost

# Finally create the StaticCostDfa
cost_func =  StaticCostDfa(cf_dfa, cf_cost_map)

# Create an LQCI improviser.
length_bounds = (1,6)
cost_bound = 25
label_prob_bounds = (Fraction(1,5), Fraction(1,2))
word_prob_bounds = {
                    "Seen0-2":(Fraction(1,80),Fraction(1,3)), 
                    "Seen3-4":(Fraction(1,90),Fraction(1,4)), 
                    "Seen5-6":(Fraction(1,100),Fraction(1,5))
                   }

improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds)
improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds)

# Create a MELQCI improviser.
length_bounds = (1,6)
cost_bound = 25
label_prob_bounds = (Fraction(1,5), Fraction(1,2))

improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds)
improviser.parameterize(cost_bound, label_prob_bounds)

# ALQCI Encoding of our example problem
# Start by declaring variables
bitvec_val = z3.BitVec("bitvec_val", 6)
cost = z3.BitVec("cost", 6)
label = z3.BitVec("label", 2)
num_ones_var = z3.BitVec("num_ones", 6)

# Set cost equal to value
formula = bitvec_val == cost

# Set label depending on value
ones = [ z3.Extract(i, i, bitvec_val) for i in range(6) ]
one_vecs  = [ z3.Concat(z3.BitVecVal(0, 5), o) for o in ones ]

formula = z3.And(formula, num_ones_var == sum(one_vecs))

formula = z3.And(formula, (label == 0) == (z3.ULE(num_ones_var, 2)))
formula = z3.And(formula, (label == 1) == (z3.And(z3.UGE(num_ones_var, 3), z3.ULE(num_ones_var, 4))))
formula = z3.And(formula, (label == 2) == (z3.And(z3.UGE(num_ones_var, 5))))

lf_label_map = {}

lf_label_map["Seen0-2"] = 0
lf_label_map["Seen3-4"] = 1
lf_label_map["Seen5-6"] = 2

# Create hard constraint and cost/label functions
hard_constraint = Z3Formula(formula, {bitvec_val})
cost_func = CostZ3Formula(cost)
label_func = LabellingZ3Formula(lf_label_map, label)

bucket_ratio, counting_tol, sampling_tol, conf = 2, 0.8, 15, 0.2

cost_bound = 25
label_prob_bounds = (Fraction(1,5), Fraction(1,2))
word_prob_bounds = {
                    "Seen0-2":(Fraction(1,80),Fraction(1,3)), 
                    "Seen3-4":(Fraction(1,90),Fraction(1,4)), 
                    "Seen5-6":(Fraction(1,100),Fraction(1,5))
                   }

alqci_improviser = ALQCI(hard_constraint, cost_func, label_func, bucket_ratio, counting_tol, sampling_tol, conf)
alqci_improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds)

print("Printing some words from our ALQCI Improviser:")
for _ in range(10):
    print(alqci_improviser.improvise())
print()


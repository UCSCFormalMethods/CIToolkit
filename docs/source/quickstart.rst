..  _quickstart:

Quickstart
============
In this section, we will examine a very simple toy problem and how we can apply different control improvisation extensions to solve it. Note that all the code for this tutorial can be found in ``examples/quickstart.py``.

Problem statement
*****************
In our example we are interested in generating a diverse set of *words* over a binary alphabet, specifically words between 1 and 6 symbols in length such as ``0``, ``1``, ``101``, ``001011``, etc. We can also think of this problem as generating a diverse set of integers in the range 0-63, as the words can be interpreted as the binary encoding of our numbers. 

At it's core Control Improvisation is about balancing control over what is being generated with diversity over what is being generated, and so throughout this tutorial we will be looking to generate diverse strings subject to different kinds of constraint. Lets first look at how we can encode the above as a formal constraint.

DFA Specifications
******************
In Control Improvisation we need a way to formally express what exactly we want to generate, what we call our *hard constraint*. We can do this using a specification. Specifically, we will encode this requirement as a Deterministic Finite Automata (DFA). We show one DFA accepting that language below.

.. image:: images/HCDfa.png

To turn this into something the CIToolkit library can understand, we can create an equivalent member of the :mod:`Dfa<citoolkit.specifications.dfa.Dfa>` class which we'll call hard constraint, as shown below.

.. code-block::
   :caption: Create our hard constraint, a Dfa object accepting only binary strings of length 1 to 6.

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

	hard_constraint = Dfa(hc_alphabet, hc_states,\
		hc_accepting_states, hc_start_state,\
		hc_transitions)

The original Control Improvisation problem also allows us to work with a *soft constraint*, a specification which sampled words must satisfy with a certain probability. For the sake of this example, let's have our soft constraint specification accept only words with a binary value less than or equal to 8. A DFA accepting this language is shown below.

.. image:: images/SCDfa.png

And once again we will encode this as a :mod:`Dfa<citoolkit.specifications.dfa.Dfa>` object.

.. code-block::
   :caption: Create our soft constraint, a Dfa object accepting only binary strings whose integer value is less than 8.

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

	soft_constraint = Dfa(sc_alphabet, sc_states,\
		sc_accepting_states, sc_start_state,\
		sc_transitions)


Note that the Dfa objects can be combined using the and (&), or (|), negation (~), and difference (-) operations, as well as minimized among other useful included functions. For more details, see the :mod:`Dfa<citoolkit.specifications.dfa.Dfa>` documentation page and the :doc:`usage` page.

With these two specifications in hand, lets take a look at how we can now use them in a Control Improvisation problem.

Control Improvisation (CI)
**************************
Using our specifications we can formalize our Control Improvisation problem. As a quick overview, this library allows you ro formulate a Control Improvisation problem (along with several extensions of CI), determines its feasibility, and if it is feasibility allows you to sample from an *improvising distribution* that meets your constraints. Specifically for our example, we want to generate only word meeting our hard constraint, we want at least a third of the words we generate to meet our soft constraint, and we want all words to have probability between 0.005 and 0.1. We can use these parameters to create a new :mod:`CI<citoolkit.improvisers.ci.CI>` object, as shown below.

.. code-block::
   :caption: Create and parameterize an improviser for our Control Improvisation problem.

	length_bounds = (1,6)
	epsilon = 0.33
	prob_bounds = (0.005, 0.1)

	ci_improviser = CI(hard_constraint, soft_constraint, length_bounds)
	ci_improviser.parameterize(epsilon, prob_bounds)

However, if we actually run this code we get an :mod:`InfeasibleSoftConstraintError<citoolkit.improvisers.improviser.InfeasibleSoftConstraintError>` exception:	**citoolkit.improvisers.improviser.InfeasibleSoftConstraintError: Greedy construction does not satisfy soft constraint, meaning no improviser can. Maximum percentage of words satisfying soft constraint was 0.56.**

It looks like our original goals are not realizable, and conveniently our improviser will throw an error when we try to parameterize it saying exactly why it was infeasible. We have two options here, we can lower our epsilon so that the instance becomes feasible with the same distribution or we can tweak our randomness parameters to allow the improviser to assign more probability to words satisfying our soft constraint. As we look at more complicated generalizations of Control Improvisation though, the number of possible tweaks we can make will increase and this can introduce some problems. The vast majority of the time solving a Control Improvisation problem is spent counting internal specifications, and if we need to start from scratch each time we need to tweak our constraints things can get cumbersome. 

This is why creating a complete Control Improvisation improviser is split into two parts: the creation of the class and calling the ``parameterize`` method. Creating the class takes only the inputs needed to setup the basics of the problem, like the constraints which define which words can be sampled over. The ``parameterize`` function takes in numerical bounds and actually tries to determine if the problem is feasible, which will involve counting some of the internal specifications created with the object. By splitting the improviser creation into two functions, one can put the ``parameterize`` function call into a ``try-except`` block and try again when a certain set of parameters is infeasible, saving any internal work that was done. One can even pickle an improviser at any stage and reload it later to parameterize it, or re-parameterize an improviser with new bounds, all while avoiding redundant work. While the time saved is miniscule in a toy example like this one, the time savings can be *significant* on larger problems.

With this in mind lets try creating an improviser with looser bounds which are actually feasible and then sampling words from the improvising distribution, as shown below.

.. code-block::
   :caption: Create and parameterize an improviser for our Control Improvisation problem, this time with feasible paramaters.

	length_bounds = (1,6)
	epsilon = 0.45
	prob_bounds = (0.0025, 0.15)

	ci_improviser = CI(hard_constraint, soft_constraint, length_bounds)

	ci_improviser.parameterize(epsilon, prob_bounds)

	for _ in range(10):
 	   print(ci_improviser.improvise())

One possible output is shown below.

.. code-block::

	('0', '1', '0', '0', '0')
	('0',)
	('0', '1', '1', '0', '0')
	('1', '1', '1', '0', '0', '0')
	('0', '0', '1', '1', '0', '1')
	('1', '1', '1')
	('0', '1', '1', '0')
	('0', '1', '0')
	('1', '0', '0')
	('0', '0', '1', '1', '0', '0')

Great, we're generating words as expected! In the next section we'll look at a generalization of CI that allows for more powerful constraints, using *quantitative soft constraints* to represent costs and *randomness over labels* constraints to give even finer control over randomness.

Labelled Quantitative Control Improvisation (LQCI)
**************************************************

While Control Improvisation can encode many constraints naturally, there are some constraints that are difficult or impossible to encode in the original problem formulation.

The first class of constraints we'd like are *quantitative const constraints*. Specifically, we'd like to come up with some function that determines the cost of any word and then using this we'd like to be able to enforce an upper bound on the expected cost of our output distribution. This is something we can approximate using the current soft constraint, but the soft constraint gives no way to actually enforce an expected cost bound.

The second class of constraints we'd like *randomness over labels* constraints. With the original randomness constraints randomness is defined with respect to all other traces. While this works well to get general diversity, oftentimes there are attributes we are particularly concerned about and these may not be uniformly distributed among all traces. In this case, we would like to be able to provide some function that assigns a label to any word, and then we'd like to enforce randomness over these labels and over words in general. Labelled Quantitative Control Improvisation (LQCI) adds both these constraints, so lets look at how we can augment our toy example to make use of them.

Label Function
----------------
For the label constraint, lets say we want good diversity over the number of "1" symbols in our words. Specifically, lets say we want our labels to be whether a string has 0-2 "1" symbols, 3-4 "1" symbols, or 5-6 "1" symbols. The encoding is fairly straightforward and the implementation is included below.

.. code-block::
	:caption: Create a labelling function for our problem.

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

.. note::

	You might have noticed that some words might have a label but aren't accepted by the hard constraint. In LQCI only words that are accepted by the hard constraint, are assigned a label by the label function, and are assigned a cost by the cost function are considered valid words, so keep that in mind when encoding your constraints.


Cost Function
---------------
For the cost constraint, lets explicitly say that the cost is the integer value of the binary word. We can encode this using a Dfa, where each accepting state has a cost associated to it and the cost of a word is the cost of the accepting state it finishes on. We provide a native class encoding this cost function called :mod:`StaticCostDfa<citoolkit.costfunctions.static_cost_dfa.StaticCostDfa>`. Details on further cost functions can be found in :doc:`usage`. This encoding gets a little tricker as must essentially construct a binary tree Dfa to keep track of the cost at any point. The implementation is included below.

.. code-block::
   :caption: Create a cost function for our problem.

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

.. note::
	
	One thing to note with all the constraints above is the the CIToolkit will internally handle minimization, so don't worry too much about optimizing your encoding.

LQCI Improviser
----------------
Now that we have new functions encoding costs and labels, lets create an :mod:`Labelled Quantitative Control Improvisation (LQCI)<citoolkit.improvisers.lqci.LQCI>` improviser to actually solve our problem with some arbitrary bounds. The only differences are that instead of a soft constraints we can set a maximum expected cost via the cost bound and we now have two randomness constraints. The first is randomness over labels, controlled by passing the label probability bounds parameter which is a tuple with lower and upper bounds on the marginal probability of selecting a word with any particular label. The second is randomness over words, controlled by passing the words probability bounds parameter which is a dictionary mapping each label to a tuple containing lower and upper bounds on the conditional probability of selecting a word with that label. Let's see an example using the CIToolkit.

.. code-block::
   :caption: Create an LQCI Improviser

	length_bounds = (1,6)
	cost_bound = 25
	label_prob_bounds = (Fraction(1,5), Fraction(1,2))
	word_prob_bounds = {
	                    "Seen0-2":(Fraction(1,40),Fraction(1,3)), 
	                    "Seen3-4":(Fraction(1,42),Fraction(1,4)), 
	                    "Seen5-6":(Fraction(1,54),Fraction(1,5))
	                   }

	improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds)
	improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds)

While this new problem formulation gives us exact control, it can be a little cumbersome to come up with the word probability bounds constraint, especially if you have many labels. However, without these constraints there are many solutions that unnecessarily sacrifice randomness, and in many situations we don't care about particular bounds as long as the overall distribution is random. To solve this problem, we provide an alternative problem definition for Maximum Entropy LQCI (MELQCI), which forgoes the randomness over words constraint but requires that the improvising distribution be the maximum entropy distribution satisfying the constraints (one way to think of this is the most random distribution). This allows the improviser to worry about finding good word probability bounds so you don't have to. The process is exactly the same as above, except you should use the :mod:`Maximum Entropy Labelled Quantitative Control Improvisation (MELQCI)<citoolkit.improvisers.lqci.MELQCI>` class and can omit the word probability bounds parameter.

.. code-block::
   :caption: Create a MELQCI Improviser

	length_bounds = (1,6)
	cost_bound = 25
	label_prob_bounds = (Fraction(1,5), Fraction(1,2))

	improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds)
	improviser.parameterize(cost_bound, label_prob_bounds)

.. note::
	There are several parameters that can be passed during improviser creation and when calling the ``parameterize`` function. Most important is the ``num_threads`` flag which can allow the improviser to parallelize its computation. For more details see the documentation on the classes.


Approximate LQCI (ALQCI)
**************************************************

While the LQCI works great for small applications, we can see from the cost example above that it has a lot of trouble encoding specifications compactly. To solve this problem, we can turn towards more powerful specifications, but that does come with a trade off when it comes to efficiency: namely that we cannot efficiently count or sample from these more powerful specs. For this reason we consider two types of specs: exact specs like Dfas which we've been concerned about thus far, and approximate specs like boolean formulas which we will cover in this section. Exact specs support counting and sampling exactly, while for approximate specs we relax that requirement and merely count and sample approximately for the sake of efficiency. For this reason only exact specs can be used in CI/LQCI/MELQCI/etc... and only approximate specs can be used in Approximate LQCI.

The approximate specifications we'll use in this section are :mod:`Z3Formula<citoolkit.specifications.z3_formula.Z3Formula>`, which supports formulas using the theory of bitvectors in Z3. Internally these are represented using :mod:`BoolFormula<citoolkit.specifications.bool_formula.BoolFormula>` specifications but Z3 formulas are generally far easier to work with. 

There are analogous approximate cost and label formulas, :mod:`CostZ3Formula<citoolkit.costfunctions.cost_z3_formula.CostZ3Formula>` and :mod:`LabellingZ3Formula<citoolkit.labellingfunctions.labelling_z3_formula.LabellingZ3Formula>` respectively. The only other major differences with ALQCI are that only integer costs are supported and ALQCI supports exponentially many costs while still having polynomial (relative to an NP oracle) runtime, though it does this by having a maximum cost error which is no more than the bucket ratio parameter. The cost function's only parameter is a bitvector that represents cost in the hard constraint and the label function takes a similar bitvector variable encoding which label and a map from strings to integer values that bitvector value can hold.

Below is an example of our example problem from before encoded as an approximate constraint. For the sake of simplicity we'll assume that we only want to generate words of exactly length 6, but this could be avoided with a more intricate encoding.

.. code-block::
   :caption: ALQCI Encoding of our example problem

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

	improviser = ALQCI(hard_constraint, cost_func, label_func, bucket_ratio, counting_tol, sampling_tol, conf)
	improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds)

If we sample our improviser, we can see some of the values we might get. Note that the CIToolkit automatically converts bitvector values to integers.

.. code-block::

	{label: 0, cost: 2, bitvec_val: 2}
	{label: 0, cost: 2, bitvec_val: 2}
	{label: 0, cost: 0, bitvec_val: 0}
	{label: 0, cost: 1, bitvec_val: 1}
	{label: 2, cost: 61, bitvec_val: 61}
	{label: 1, cost: 11, bitvec_val: 11}
	{label: 1, cost: 14, bitvec_val: 14}
	{label: 0, cost: 0, bitvec_val: 0}
	{label: 0, cost: 0, bitvec_val: 0}
	{label: 2, cost: 61, bitvec_val: 61}

Conclusion
**********

Hopefully this quickstart tutorial has helped give you an idea of what you can do with the CIToolkit! If you have any questions, please feel free to post an issue on GitHub or contact Eric Vin (evin@ucsc.edu).

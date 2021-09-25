from robotic_planning_exact import run_exact_experiments
from robotic_planning_approximate import run_approximate_experiments

if __name__ == '__main__':
    run_exact_experiments(True)
    run_exact_experiments(False)
    run_approximate_experiments(True)
    run_approximate_experiments(False)

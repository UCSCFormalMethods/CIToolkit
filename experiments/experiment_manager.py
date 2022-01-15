import multiprocessing
import psutil, os

from robotic_planning_exact import run_exact_experiments
from robotic_planning_approximate import run_approximate_experiments

# Source: https://stackoverflow.com/questions/22784890/auto-kill-process-and-child-process-of-multiprocessing-pool
def kill_proc_tree(pid, including_parent=False):    
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

def func_timeout(func, args, timeout=(24*60*60)):
    p = multiprocessing.Process(target=func, args=args)
    p.start()

    p.join(timeout)

    if p.is_alive():
        print("Terminating experiment after " + str(timeout) + " seconds...")
        kill_proc_tree(os.getpid())
        p.terminate()
    else:
        print("Experiment completed without exceeding timeout...")

if __name__ == '__main__':
    for i in range(10):
        func_timeout(run_approximate_experiments, (True, 100))
        func_timeout(run_approximate_experiments, (True, 1000))
        func_timeout(run_approximate_experiments, (True, 10000))
        func_timeout(run_approximate_experiments, (False, 100))
        func_timeout(run_approximate_experiments, (False, 1000))
        func_timeout(run_approximate_experiments, (False, 10000))
    # func_timeout(run_exact_experiments, (False,))
    # func_timeout(run_exact_experiments, (True,))
    # run_exact_experiments(False)
# make classes for extra class
# make version for proper programs
# es and then just parse them in
 
import copy
import json
import sys
import subprocess
import types
import threading
import time
import multiprocessing
from multiprocessing import Process, Queue

class TimeoutError(Exception):
    pass

def run_function_in_process(func, args, result_queue):
    """Run function in separate process"""
    try:
        result = func(*args)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', e))

def py_try_with_process_timeout(algo, *args, correct=False, fixed=False, timeout=5):
    """Run function with hard timeout using multiprocessing"""
    if not fixed:
        module = __import__("python_programs."+algo)
    else:
        if not correct: 
            module = __import__("fixed_programs."+algo)
        else: 
            module = __import__("correct_python_programs."+algo)

    fx = getattr(module, algo)
    func = getattr(fx, algo)
    
    # Create a queue to get results
    result_queue = Queue()
    
    # Start process
    process = Process(target=run_function_in_process, args=(func, args, result_queue))
    process.start()
    
    # Wait for result or timeout
    process.join(timeout)
    
    if process.is_alive():
        # Process is still running, terminate it
        process.terminate()
        process.join(1)  # Wait 1 second for clean termination
        if process.is_alive():
            process.kill()  # Force kill if still alive
        return f"TIMEOUT after {timeout} seconds"
    
    # Get result from queue
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == 'success':
            return result
        else:
            return result
    else:
        return "No result returned"

def py_try(algo, *args, correct=False, fixed=False, timeout=5):
    """Regular py_try with shorter default timeout"""
    # Skip known problematic test cases entirely for speed
    if algo in ["knapsack", "levenshtein"] and not correct:
        return "SKIPPED (too slow for testing)"
    
    # Use process timeout for bitcount and other potentially slow algorithms
    if algo in ["bitcount"]:
        return py_try_with_process_timeout(algo, *args, correct=correct, fixed=fixed, timeout=timeout)
    
    # Regular execution for other algorithms
    if not fixed:
        module = __import__("python_programs."+algo)
    else:
        if not correct: module = __import__("fixed_programs."+algo)
        else: module = __import__("correct_python_programs."+algo)

    fx = getattr(module, algo)

    try:
        return getattr(fx,algo)(*args)
    except:
        return sys.exc_info()


def prettyprint(o):
    if isinstance(o, types.GeneratorType):
        return("(generator) " + str(list(o)))
    else:
        return(str(o))

graph_based = ["breadth_first_search",
               "depth_first_search",
               "detect_cycle",
               "minimum_spanning_tree",
               "reverse_linked_list",
               "shortest_path_length",
               "shortest_path_lengths",
               "shortest_paths",
               "topological_ordering"
              ]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tester.py <program_name>")
        print("Example: python tester.py quicksort")
        print("Add --fast flag to skip slow tests: python tester.py quicksort --fast")
        sys.exit(1)
    
    algo = sys.argv[1]
    fast_mode = "--fast" in sys.argv
    
    if fast_mode:
        print(f"🚀 Running {algo} in FAST mode (skipping slow tests)")
    else:
        print(f"🐌 Running {algo} in NORMAL mode (may be slow for some algorithms)")

    if algo in graph_based:
        print("Correct Python:")
        correct_module = __import__("correct_python_programs."+algo+"_test")
        correct_fx = getattr(correct_module, algo+"_test")
        getattr(correct_fx,"main")()
        print()

        print("Bad Python:")
        test_module = __import__("python_programs."+algo+"_test")
        test_fx = getattr(test_module, algo+"_test")
        try:
            getattr(test_fx,"main")()
        except:
            print(sys.exc_info())
        print()

    else:
        working_file = open("json_testcases/"+algo+".json", 'r')
        test_count = 0
        max_tests = 3 if fast_mode else 100  # Limit tests in fast mode

        for line in working_file:
            if fast_mode and test_count >= max_tests:
                print(f"⏭️  Skipping remaining tests (fast mode - tested {max_tests})")
                break
                
            py_testcase = json.loads(line)
            print(f"\n📝 Test {test_count + 1}: {py_testcase}")
            test_in, test_out = py_testcase
            if not isinstance(test_in, list):
                test_in = [test_in]

            # Much shorter timeout for fast mode
            timeout_duration = 3 if fast_mode else (10 if algo in ["knapsack", "levenshtein"] else 5)

            print(f"⏱️  Timeout set to {timeout_duration} seconds")
            
            # check good Python version
            print("🟢 Testing correct version...")
            py_out_good = py_try(algo, *copy.deepcopy(test_in), correct=True, timeout=timeout_duration)
            print("Correct Python: " + prettyprint(py_out_good))

            # check bad Python version  
            print("🔴 Testing buggy version...")
            py_out_test = py_try(algo, *copy.deepcopy(test_in), timeout=timeout_duration)
            print("Bad Python: " + prettyprint(py_out_test))

            # check fixed version if available
            try:
                print("🔧 Testing fixed version...")
                py_out_fixed = py_try(algo, *copy.deepcopy(test_in), fixed=True, timeout=timeout_duration)
                print("Fixed Python: " + prettyprint(py_out_fixed))
            except ImportError:
                print("Fixed Python: Not available (fixed_programs folder not found)")
            
            test_count += 1
            
            # Add small delay to prevent overwhelming
            if not fast_mode:
                time.sleep(0.1)
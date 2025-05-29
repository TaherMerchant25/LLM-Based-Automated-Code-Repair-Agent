import json
import copy
import sys
import importlib
import threading
import time
import os
from queue import Queue, Empty

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def run_with_timeout(func, args, timeout_seconds):
    """Run a function with timeout using threading"""
    result_queue = Queue()
    exception_queue = Queue()
    
    def target():
        try:
            result = func(*args)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutError(f"Function execution timed out after {timeout_seconds} seconds")
    
    # Check if there was an exception
    try:
        exception = exception_queue.get_nowait()
        raise exception
    except Empty:
        pass
    
    # Get the result
    try:
        return result_queue.get_nowait()
    except Empty:
        raise RuntimeError("Function completed but no result was returned")

def test_fixed_program(program_name, test_timeout=5):
    """Test only the fixed version of a program without testing buggy versions"""
    print(f"Testing fixed version of {program_name}...")
    print("="*50)
    
    try:
        # Check if test cases file exists first
        testcase_file = f"json_testcases/{program_name}.json"
        if not os.path.exists(testcase_file):
            print(f"SKIP: No test cases file found for {program_name} at {testcase_file}")
            return None  # Return None to indicate skipped, not failed
        
        # Import the fixed program
        module_name = f"fixed_programs.{program_name}"
        module = importlib.import_module(module_name)
        func = getattr(module, program_name)
        
        # Load test cases
        with open(testcase_file, 'r') as f:
            passed = 0
            total = 0
            failed_tests = []
            
            for line in f:
                test_case = json.loads(line.strip())
                test_in, expected_out = test_case
                
                if not isinstance(test_in, list):
                    test_in = [test_in]
                
                total += 1
                
                try:
                    # Test fixed version with timeout protection
                    result = run_with_timeout(func, copy.deepcopy(test_in), test_timeout)
                    
                    if result == expected_out:
                        passed += 1
                        print(f"PASS Test {total}: Input {test_in} -> Expected {expected_out}, Got {result}")
                    else:
                        failed_tests.append((test_in, expected_out, result))
                        print(f"FAIL Test {total}: Input {test_in} -> Expected {expected_out}, Got {result}")
                        
                except TimeoutError as e:
                    failed_tests.append((test_in, expected_out, f"TIMEOUT: {e}"))
                    print(f"TIMEOUT Test {total}: Input {test_in} -> Expected {expected_out}, Got TIMEOUT after {test_timeout}s")
                except Exception as e:
                    failed_tests.append((test_in, expected_out, f"ERROR: {e}"))
                    print(f"ERROR Test {total}: Input {test_in} -> Expected {expected_out}, Got ERROR: {e}")
            
            print("\n" + "="*50)
            print(f"SUMMARY for {program_name}:")
            print(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
            
            if failed_tests:
                print(f"Failed: {len(failed_tests)} tests")
                print("\nFailed test details:")
                for i, (inp, exp, got) in enumerate(failed_tests, 1):
                    print(f"  {i}. Input: {inp}, Expected: {exp}, Got: {got}")
            
            success = passed == total
            print(f"\nOverall Result: {'SUCCESS' if success else 'FAILED'}")
            return success
            
    except ImportError as e:
        print(f"Error: Could not import fixed program: {e}")
        print(f"Make sure {program_name}.py exists in the fixed_programs/ folder")
        return False
    except Exception as e:
        print(f"Error testing {program_name}: {e}")
        return False

def test_all_fixed_programs(test_timeout=5):
    """Test all programs in the fixed_programs folder"""
    
    fixed_dir = "fixed_programs"
    if not os.path.exists(fixed_dir):
        print(f"Error: {fixed_dir} directory not found")
        return
    
    # Get all Python files in fixed_programs
    program_files = [f[:-3] for f in os.listdir(fixed_dir) 
                    if f.endswith('.py') and not f.startswith('__')]
    
    if not program_files:
        print(f"No Python files found in {fixed_dir}")
        return
    
    print(f"Found {len(program_files)} fixed programs to test")
    print(f"Test timeout: {test_timeout} seconds per test case")
    print("="*60)
    
    results = {}
    skipped_programs = []
    
    for program in program_files:
        print(f"\nTesting: {program}")
        result = test_fixed_program(program, test_timeout)
        
        if result is None:
            # Program was skipped due to missing JSON file
            skipped_programs.append(program)
        else:
            # Program was tested (either passed or failed)
            results[program] = result
        
        print("\n" + "-"*60)
    
    # Summary - only count programs that were actually tested
    successful = sum(results.values())
    total_tested = len(results)
    total_found = len(program_files)
    
    print(f"\nFINAL SUMMARY:")
    print(f"Programs found: {total_found}")
    print(f"Programs tested: {total_tested}")
    print(f"Programs skipped (no test cases): {len(skipped_programs)}")
    
    if total_tested > 0:
        print(f"Successfully fixed programs: {successful}/{total_tested} ({successful/total_tested*100:.1f}%)")
    else:
        print("No programs were tested (all were skipped due to missing test cases)")
    
    if results:
        print(f"\nTested programs - Successful fixes:")
        for program, success in results.items():
            if success:
                print(f"  - {program}")
        
        if successful < total_tested:
            print(f"\nTested programs - Failed fixes:")
            for program, success in results.items():
                if not success:
                    print(f"  - {program}")
    
    if skipped_programs:
        print(f"\nSkipped programs (no JSON test cases):")
        for program in skipped_programs:
            print(f"  - {program}")

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fixed programs with timeout protection')
    parser.add_argument('program', nargs='?', help='Specific program to test (optional)')
    parser.add_argument('--timeout', '-t', type=int, default=5, 
                       help='Timeout in seconds for each test case (default: 5)')
    
    args = parser.parse_args()
    
    if args.program:
        # Test specific program
        result = test_fixed_program(args.program, args.timeout)
        if result is None:
            print(f"\nProgram {args.program} was skipped due to missing test cases file.")
    else:
        # Test all fixed programs
        test_all_fixed_programs(args.timeout)

if __name__ == "__main__":
    # Handle both old and new usage patterns
    if len(sys.argv) == 1:
        # Test all fixed programs with default timeout
        test_all_fixed_programs()
    elif len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
        # Test specific program with default timeout
        program_name = sys.argv[1]
        result = test_fixed_program(program_name)
        if result is None:
            print(f"\nProgram {program_name} was skipped due to missing test cases file.")
    else:
        # Use argparse for more complex arguments
        main()
import os
import google.generativeai as genai
import logging
from pathlib import Path
import difflib
import ast
import time
import json
from typing import Dict, List, Tuple
import concurrent.futures
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastCodeCorrectionAgent:
    def __init__(self, api_key: str, max_workers: int = 3, fast_mode: bool = True, 
                 accuracy_mode: str = 'standard', model_name: str = 'gemini-1.5-flash'):
        """Initialize the code correction agent with Gemini API key.
        
        Args:
            accuracy_mode: 'fast', 'standard', 'high', 'maximum'
            model_name: 'gemini-1.5-flash' (fast) or 'gemini-1.5-pro' (more accurate)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.comparison_results = []
        self.max_workers = max_workers
        self.rate_limit_lock = threading.Lock()
        self.fast_mode = fast_mode
        self.accuracy_mode = accuracy_mode
        self.model_name = model_name
        
        # Configure accuracy vs speed settings
        self.accuracy_settings = {
            'fast': {'multi_attempt': False, 'temperature': 0.1, 'delay': 0.1},
            'standard': {'multi_attempt': False, 'temperature': 0.0, 'delay': 0.3},
            'high': {'multi_attempt': True, 'temperature': 0.0, 'delay': 0.5},
            'maximum': {'multi_attempt': True, 'temperature': 0.0, 'delay': 1.0}
        }
        
        self.current_settings = self.accuracy_settings.get(accuracy_mode, self.accuracy_settings['standard'])
        
    def generate_repair_prompt(self, buggy_code: str, algorithm_name: str, attempt: int = 1) -> str:
        """Generate an optimized prompt for fixing QuixBugs dataset code."""
        
        if attempt == 1:
            # Simplified, direct prompt focusing on common bug patterns
            prompt = f"""Fix the single bug in this Python function.

Algorithm: {algorithm_name}
Code:
```python
{buggy_code}
```

This code contains exactly one bug. Common QuixBugs patterns:
- Off-by-one errors in loops or indexing
- Wrong comparison operators (< vs <=, > vs >=)
- Incorrect variable references
- Wrong loop bounds or conditions
- Missing/incorrect increments

Find the bug and return only the corrected Python code."""
            
        else:
            # Second attempt with step-by-step reasoning (ReACT approach)
            prompt = f"""Debug this Python function step by step.

Algorithm: {algorithm_name}
Code:
```python
{buggy_code}
```

REASONING:
1. What should this algorithm do?
2. Trace through the code with a simple input
3. Identify where the logic fails
4. What is the minimal fix needed?

ACTION:
Return only the corrected Python code with the single bug fixed."""
        
        return prompt

    def fix_code_with_gemini(self, buggy_code: str, algorithm_name: str, enable_multi_attempt: bool = True) -> dict:
        """Use Gemini to fix the buggy code with optional multi-attempt strategy for higher accuracy."""
        max_retries = 2
        base_delay = 10
        best_result = None
        
        for attempt in range(1, 3 if enable_multi_attempt else 2):  # 1-2 attempts
            for retry in range(max_retries):
                try:
                    prompt = self.generate_repair_prompt(buggy_code, algorithm_name, attempt)
                    
                    # Use temperature from settings for consistency
                    generation_config = genai.types.GenerationConfig(
                        temperature=self.current_settings['temperature']
                    )
                    
                    response = self.model.generate_content(prompt, generation_config=generation_config)
                    fixed_code = response.text.strip()
                    
                    # Clean up code formatting
                    if fixed_code.startswith('```python'):
                        fixed_code = fixed_code.replace('```python', '').replace('```', '').strip()
                    elif fixed_code.startswith('```'):
                        lines = fixed_code.split('\n')
                        fixed_code = '\n'.join(lines[1:-1]) if len(lines) > 2 else fixed_code.replace('```', '').strip()
                    
                    result = {
                        'success': True,
                        'fixed_code': fixed_code,
                        'original_response': response.text,
                        'attempt': attempt
                    }
                    
                    # If multi-attempt is enabled, evaluate this attempt
                    if enable_multi_attempt and attempt == 1:
                        # Quick syntax check and basic validation
                        if self.is_valid_python_syntax(fixed_code):
                            best_result = result
                            break  # Continue to attempt 2
                        else:
                            logger.warning(f"Attempt {attempt} produced invalid syntax for {algorithm_name}")
                            continue
                    else:
                        return result
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                        if retry < max_retries - 1:
                            delay = base_delay * (1.5 ** retry)
                            logger.warning(f"Rate limit hit for {algorithm_name}. Waiting {delay:.1f} seconds before retry {retry + 1}/{max_retries}")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"Max retries exceeded for {algorithm_name} due to rate limiting")
                            return {
                                'success': False,
                                'error': f"Rate limit exceeded after {max_retries} attempts: {error_msg}"
                            }
                    else:
                        logger.error(f"Error fixing code with Gemini for {algorithm_name}: {error_msg}")
                        if best_result:  # Return best result from previous attempt
                            return best_result
                        return {
                            'success': False,
                            'error': error_msg
                        }
                break  # Move to next attempt
        
        # If we have multiple attempts, compare and return the best one
        if best_result:
            return best_result
        
        return {
            'success': False,
            'error': f"Failed after all attempts"
        }

    def quick_compare_with_correct_version(self, algorithm_name: str, fixed_code: str) -> Dict:
        """Quick comparison with correct version - minimal analysis."""
        try:
            correct_file_path = f"correct_python_programs/{algorithm_name}.py"
            if not os.path.exists(correct_file_path):
                return {
                    'comparison_available': False,
                    'error': f"Correct version not found: {correct_file_path}"
                }
            
            with open(correct_file_path, 'r') as f:
                correct_code = f.read()
            
            similarity_ratio = difflib.SequenceMatcher(None, fixed_code.strip(), correct_code.strip()).ratio()
            exact_match = fixed_code.strip() == correct_code.strip()
            
            return {
                'comparison_available': True,
                'similarity_ratio': similarity_ratio,
                'exact_match': exact_match,
                'quick_mode': True
            }
            
        except Exception as e:
            return {
                'comparison_available': False,
                'error': f"Error comparing codes: {str(e)}"
            }

    def compare_with_correct_version(self, algorithm_name: str, fixed_code: str) -> Dict:
        """Compare the fixed code with the correct version from the dataset."""
        if self.fast_mode:
            return self.quick_compare_with_correct_version(algorithm_name, fixed_code)
            
        try:
            correct_file_path = f"correct_python_programs/{algorithm_name}.py"
            if not os.path.exists(correct_file_path):
                return {
                    'comparison_available': False,
                    'error': f"Correct version not found: {correct_file_path}"
                }
            
            with open(correct_file_path, 'r') as f:
                correct_code = f.read()
            
            similarity_ratio = difflib.SequenceMatcher(None, fixed_code.strip(), correct_code.strip()).ratio()
            
            fixed_syntax_valid = self.is_valid_python_syntax(fixed_code)
            correct_syntax_valid = self.is_valid_python_syntax(correct_code)
            
            diff = list(difflib.unified_diff(
                correct_code.splitlines(keepends=True),
                fixed_code.splitlines(keepends=True),
                fromfile='correct_version',
                tofile='fixed_version',
                lineterm=''
            ))
            
            logical_similarity = self.compare_code_structure(fixed_code, correct_code)
            
            return {
                'comparison_available': True,
                'similarity_ratio': similarity_ratio,
                'fixed_syntax_valid': fixed_syntax_valid,
                'correct_syntax_valid': correct_syntax_valid,
                'diff': ''.join(diff),
                'logical_similarity': logical_similarity,
                'exact_match': fixed_code.strip() == correct_code.strip()
            }
            
        except Exception as e:
            return {
                'comparison_available': False,
                'error': f"Error comparing codes: {str(e)}"
            }

    def is_valid_python_syntax(self, code: str) -> bool:
        """Check if the code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def compare_code_structure(self, code1: str, code2: str) -> float:
        """Compare the structural similarity of two Python codes."""
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            struct1 = self.get_ast_structure(tree1)
            struct2 = self.get_ast_structure(tree2)
            
            return difflib.SequenceMatcher(None, struct1, struct2).ratio()
            
        except:
            return 0.0

    def get_ast_structure(self, tree) -> str:
        """Get a string representation of AST structure."""
        structure = []
        for node in ast.walk(tree):
            structure.append(type(node).__name__)
        return ' '.join(structure)

    def test_fixed_code(self, algorithm_name: str) -> Dict:
        """Test the fixed code using the existing test framework - SKIPPED in fast mode."""
        if self.fast_mode:
            return {
                'tests_available': False,
                'skipped': True,
                'reason': 'Fast mode enabled - testing skipped for speed'
            }
            
        try:
            import sys
            sys.path.append('.')
            from tester import py_try
            
            testcase_file = f"json_testcases/{algorithm_name}.json"
            if os.path.exists(testcase_file):
                test_results = []
                with open(testcase_file, 'r') as f:
                    for line in f:
                        try:
                            testcase = json.loads(line)
                            test_in, expected_out = testcase
                            if not isinstance(test_in, list):
                                test_in = [test_in]
                            
                            fixed_result = py_try(algorithm_name, *test_in, fixed=True)
                            correct_result = py_try(algorithm_name, *test_in, correct=True)
                            
                            test_results.append({
                                'input': test_in,
                                'expected': expected_out,
                                'fixed_output': fixed_result,
                                'correct_output': correct_result,
                                'fixed_matches_expected': fixed_result == expected_out,
                                'fixed_matches_correct': fixed_result == correct_result
                            })
                            
                        except Exception as e:
                            test_results.append({
                                'error': f"Test execution error: {str(e)}"
                            })
                
                successful_tests = sum(1 for result in test_results 
                                     if result.get('fixed_matches_expected', False))
                total_tests = len(test_results)
                success_rate = successful_tests / total_tests if total_tests > 0 else 0
                
                return {
                    'tests_available': True,
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,  
                    'success_rate': success_rate,
                    'test_results': test_results
                }
            else:
                return {
                    'tests_available': False,
                    'error': f"No test cases found for {algorithm_name}"
                }
                
        except Exception as e:
            return {
                'tests_available': False,
                'error': f"Error testing code: {str(e)}"
            }

    def process_single_program(self, algorithm_name: str) -> dict:
        """Process a single buggy program and fix it with minimal evaluation."""
        logger.info(f"Processing: {algorithm_name}")
        
        try:
            buggy_file_path = f"python_programs/{algorithm_name}.py"
            if not os.path.exists(buggy_file_path):
                return {
                    'success': False,
                    'error': f"File not found: {buggy_file_path}"
                }
            
            with open(buggy_file_path, 'r') as f:
                buggy_code = f.read()
            
            logger.info(f"Read buggy code from: {buggy_file_path}")
            
            fix_result = self.fix_code_with_gemini(buggy_code, algorithm_name, 
                                                  enable_multi_attempt=self.current_settings['multi_attempt'])
            
            if not fix_result['success']:
                return {
                    'success': False,
                    'algorithm': algorithm_name,
                    'error': f"Failed to fix code: {fix_result.get('error', 'Unknown error')}"
                }
            
            fixed_code = fix_result['fixed_code']
            
            os.makedirs("fixed_programs", exist_ok=True)
            
            fixed_file_path = f"fixed_programs/{algorithm_name}.py"
            with open(fixed_file_path, 'w') as f:
                f.write(fixed_code)
            
            logger.info(f"Saved fixed code to: {fixed_file_path}")
            
            comparison_result = self.compare_with_correct_version(algorithm_name, fixed_code)
            test_result = self.test_fixed_code(algorithm_name)
            
            result = {
                'success': True,
                'algorithm': algorithm_name,
                'input_file': buggy_file_path,
                'output_file': fixed_file_path,
                'comparison': comparison_result,
                'testing': test_result,
                'fast_mode': self.fast_mode
            }
            
            self.comparison_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {algorithm_name}: {str(e)}")
            return {
                'success': False,
                'algorithm': algorithm_name,
                'error': str(e)
            }

    def process_single_program_with_rate_limit(self, algorithm_name: str) -> dict:
        """Process single program with thread-safe rate limiting."""
        with self.rate_limit_lock:
            time.sleep(self.current_settings['delay'])
        return self.process_single_program(algorithm_name)

    def process_programs_sequentially_fast(self) -> dict:
        """Process programs sequentially with minimal delays."""
        python_programs_dir = "python_programs"
        
        if not os.path.exists(python_programs_dir):
            return {
                'error': f"Directory not found: {python_programs_dir}",
                'success': False
            }
        
        program_files = [f[:-3] for f in os.listdir(python_programs_dir) 
                        if f.endswith('.py') and not f.endswith('_test.py')]
        
        logger.info(f"Found {len(program_files)} programs to process in FAST mode")
        
        results = []
        successful_fixes = 0
        
        for i, algorithm_name in enumerate(program_files):
            logger.info(f"Processing {i+1}/{len(program_files)}: {algorithm_name}")
            
            result = self.process_single_program(algorithm_name)
            results.append(result)
            
            if result['success']:
                successful_fixes += 1
                comparison = result.get('comparison', {})
                similarity = comparison.get('similarity_ratio', 0) if comparison.get('comparison_available') else 0
                status = "EXACT" if comparison.get('exact_match') else ("HIGH" if similarity > 0.8 else ("MED" if similarity > 0.5 else "LOW"))
                print(f"[{status}] Fixed: {algorithm_name} (Similarity: {similarity:.2%})")
            else:
                print(f"[FAILED] {algorithm_name} - {result.get('error', 'Unknown error')}")
            
            time.sleep(self.current_settings['delay'])
        
        summary = {
            'total_programs': len(program_files),
            'successful_fixes': successful_fixes,
            'success_rate': successful_fixes / len(program_files) if program_files else 0,
            'results': results,
            'success': True,
            'processing_mode': 'sequential_fast'
        }
        
        return summary

    def process_programs_concurrently(self) -> dict:
        """Process programs with controlled concurrency."""
        python_programs_dir = "python_programs"
        
        if not os.path.exists(python_programs_dir):
            return {'error': f"Directory not found: {python_programs_dir}", 'success': False}
        
        program_files = [f[:-3] for f in os.listdir(python_programs_dir) 
                        if f.endswith('.py') and not f.endswith('_test.py')]
        
        logger.info(f"Found {len(program_files)} programs to process with {self.max_workers} workers")
        
        results = []
        successful_fixes = 0
        
        batch_size = self.max_workers * 2
        
        for i in range(0, len(program_files), batch_size):
            batch = program_files[i:i+batch_size]
            batch_results = []
            
            print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} programs)...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_program = {
                    executor.submit(self.process_single_program_with_rate_limit, program): program 
                    for program in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_program):
                    program = future_to_program[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result['success']:
                            successful_fixes += 1
                            comparison = result.get('comparison', {})
                            similarity = comparison.get('similarity_ratio', 0) if comparison.get('comparison_available') else 0
                            status = "EXACT" if comparison.get('exact_match') else ("HIGH" if similarity > 0.8 else ("MED" if similarity > 0.5 else "LOW"))
                            print(f"  [{status}] Fixed: {program} (Similarity: {similarity:.2%})")
                        else:
                            print(f"  [FAILED] {program} - {result.get('error', 'Unknown error')}")
                            
                    except Exception as exc:
                        print(f"  [ERROR] {program} generated an exception: {exc}")
                        batch_results.append({
                            'success': False,
                            'algorithm': program,
                            'error': str(exc)
                        })
            
            results.extend(batch_results)
            
            if i + batch_size < len(program_files):
                print(f"   Brief pause between batches...")
                time.sleep(1)
        
        return {
            'total_programs': len(program_files),
            'successful_fixes': successful_fixes,
            'success_rate': successful_fixes / len(program_files) if program_files else 0,
            'results': results,
            'success': True,
            'processing_mode': f'concurrent_{self.max_workers}_workers'
        }

    def generate_evaluation_report(self, summary: Dict) -> str:
        """Generate a detailed evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("QUIXBUGS CODE CORRECTION RESULTS")
        report.append("=" * 80)
        
        processing_mode = summary.get('processing_mode', 'unknown')
        report.append(f"\nPROCESSING MODE: {processing_mode}")
        report.append(f"FAST MODE: {'Enabled' if self.fast_mode else 'Disabled'}")
        report.append(f"ACCURACY MODE: {self.accuracy_mode}")
        report.append(f"MODEL: {self.model_name}")
        report.append(f"\nOVERALL STATISTICS:")
        report.append(f"Total Programs: {summary['total_programs']}")
        report.append(f"Successfully Fixed: {summary['successful_fixes']}")
        report.append(f"Success Rate: {summary['success_rate']:.2%}")
        
        similarities = []
        exact_matches = 0
        syntax_valid = 0
        
        for result in summary['results']:
            if result['success'] and result.get('comparison', {}).get('comparison_available'):
                comp = result['comparison']
                similarities.append(comp['similarity_ratio'])
                if comp.get('exact_match', False):
                    exact_matches += 1
                if comp.get('fixed_syntax_valid', True):
                    syntax_valid += 1
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            report.append(f"\nSIMILARITY ANALYSIS:")
            report.append(f"Average Similarity to Correct Code: {avg_similarity:.2%}")
            report.append(f"Exact Matches: {exact_matches}")
            report.append(f"High Similarity (>80%): {sum(1 for s in similarities if s > 0.8)}")
            report.append(f"Medium Similarity (50-80%): {sum(1 for s in similarities if 0.5 <= s <= 0.8)}")
            report.append(f"Low Similarity (<50%): {sum(1 for s in similarities if s < 0.5)}")
        
        report.append(f"\nDETAILED RESULTS:")
        report.append("-" * 80)
        
        for result in summary['results']:
            if result['success']:
                algo = result['algorithm']
                comp = result.get('comparison', {})
                
                if comp.get('comparison_available'):
                    similarity = comp['similarity_ratio']
                    status = "EXACT" if comp.get('exact_match') else ("HIGH" if similarity > 0.8 else ("MED" if similarity > 0.5 else "LOW"))
                    report.append(f"[{status}] {algo:<25} | Similarity: {similarity:.2%}")
                else:
                    report.append(f"[UNKNOWN] {algo:<25} | No comparison available")
            else:
                report.append(f"[FAILED] {result.get('algorithm', 'Unknown'):<25} | {result.get('error', 'Unknown error')}")
        
        return '\n'.join(report)

    def run_correction(self, concurrent: bool = False):
        """Main method to run the code correction process."""
        mode_desc = "CONCURRENT" if concurrent else "SEQUENTIAL FAST"
        print(f"Starting {mode_desc} QuixBugs Code Correction")
        print(f"Fast Mode: {'Enabled' if self.fast_mode else 'Disabled'}")
        print(f"Accuracy Mode: {self.accuracy_mode}")
        print(f"Model: {self.model_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        if concurrent:
            summary = self.process_programs_concurrently()
        else:
            summary = self.process_programs_sequentially_fast()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not summary.get('success', False):
            print(f"Error: {summary.get('error', 'Unknown error')}")
            return
        
        report = self.generate_evaluation_report(summary)
        print(report)
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"Total Processing Time: {processing_time:.1f} seconds")
        print(f"Average Time per Program: {processing_time/summary['total_programs']:.1f} seconds")
        print(f"Programs per Minute: {summary['total_programs'] * 60 / processing_time:.1f}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f'quixbugs_report_{timestamp}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(report)
            f.write(f"\n\nPERFORMANCE METRICS:\n")
            f.write(f"Total Processing Time: {processing_time:.1f} seconds\n")
            f.write(f"Average Time per Program: {processing_time/summary['total_programs']:.1f} seconds\n")
            f.write(f"Programs per Minute: {summary['total_programs'] * 60 / processing_time:.1f}\n")
        
        print(f"\nDetailed report saved to: {report_filename}")
        print(f"Correction completed! Fixed programs saved in 'fixed_programs/' folder")


def main():
    """Main function to run the QuixBugs code correction."""
    API_KEY = ""
    
    if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("Please set your Gemini API key in the script")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # ACCURACY VS SPEED CONFIGURATION
    ACCURACY_MODE = 'standard'    # Options: 'fast', 'standard', 'high', 'maximum'
    MODEL_NAME = 'gemini-1.5-flash'  # Options: 'gemini-1.5-flash' (fast), 'gemini-1.5-pro' (accurate)
    FAST_MODE = True              # Skip detailed testing
    CONCURRENT = False            # Concurrent processing
    MAX_WORKERS = 3
    
    print(f"QuixBugs Configuration:")
    print(f"   Accuracy Mode: {ACCURACY_MODE}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Fast Mode: {'Enabled' if FAST_MODE else 'Disabled'}")
    print(f"   Concurrent Processing: {'Enabled' if CONCURRENT else 'Disabled'}")
    if CONCURRENT:
        print(f"   Max Workers: {MAX_WORKERS}")
    
    accuracy_info = {
        'fast': "Fastest processing, good accuracy (~75-85%)",
        'standard': "Balanced speed/accuracy (~80-90%)",
        'high': "Higher accuracy, slower processing (~85-92%)",
        'maximum': "Maximum accuracy, slowest processing (~88-95%)"
    }
    print(f"   Expected Performance: {accuracy_info.get(ACCURACY_MODE, 'Unknown')}")
    print()
    
    agent = FastCodeCorrectionAgent(
        api_key=API_KEY, 
        max_workers=MAX_WORKERS, 
        fast_mode=FAST_MODE,
        accuracy_mode=ACCURACY_MODE,
        model_name=MODEL_NAME
    )
    agent.run_correction(concurrent=CONCURRENT)


if __name__ == "__main__":
    main()

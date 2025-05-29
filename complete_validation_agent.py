import os
import logging
import json
import ast
import difflib
from pathlib import Path
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Tuple
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1024

class SimilarityScorer:
    """Calculate similarity between fixed code and correct reference code"""
    
    @staticmethod
    def calculate_similarity(fixed_code: str, correct_code: str) -> Dict[str, float]:
        """Calculate various similarity metrics between two code snippets"""
        
        # Normalize code for comparison
        fixed_normalized = SimilarityScorer._normalize_code(fixed_code)
        correct_normalized = SimilarityScorer._normalize_code(correct_code)
        
        # Calculate different similarity metrics
        similarity_metrics = {}
        
        # 1. Character-level similarity using difflib
        char_similarity = difflib.SequenceMatcher(None, fixed_normalized, correct_normalized).ratio()
        similarity_metrics['character_similarity'] = char_similarity
        
        # 2. Line-by-line similarity
        fixed_lines = [line.strip() for line in fixed_normalized.split('\n') if line.strip()]
        correct_lines = [line.strip() for line in correct_normalized.split('\n') if line.strip()]
        
        line_similarity = difflib.SequenceMatcher(None, fixed_lines, correct_lines).ratio()
        similarity_metrics['line_similarity'] = line_similarity
        
        # 3. Token-level similarity (split by common separators)
        fixed_tokens = SimilarityScorer._tokenize_code(fixed_normalized)
        correct_tokens = SimilarityScorer._tokenize_code(correct_normalized)
        
        token_similarity = difflib.SequenceMatcher(None, fixed_tokens, correct_tokens).ratio()
        similarity_metrics['token_similarity'] = token_similarity
        
        # 4. AST-based structural similarity (if both are valid Python)
        try:
            fixed_ast = ast.parse(fixed_code)
            correct_ast = ast.parse(correct_code)
            
            fixed_structure = SimilarityScorer._extract_ast_structure(fixed_ast)
            correct_structure = SimilarityScorer._extract_ast_structure(correct_ast)
            
            structure_similarity = difflib.SequenceMatcher(None, fixed_structure, correct_structure).ratio()
            similarity_metrics['structure_similarity'] = structure_similarity
        except:
            similarity_metrics['structure_similarity'] = 0.0
        
        # 5. Overall weighted similarity
        weights = {
            'character_similarity': 0.2,
            'line_similarity': 0.3,
            'token_similarity': 0.3,
            'structure_similarity': 0.2
        }
        
        overall_similarity = sum(
            similarity_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        similarity_metrics['overall_similarity'] = overall_similarity
        
        return similarity_metrics
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code by removing extra whitespace and standardizing formatting"""
        # Remove comments
        lines = []
        for line in code.split('\n'):
            # Remove inline comments but keep string literals
            if '#' in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        line = line[:i]
                        break
            lines.append(line.strip())
        
        # Remove empty lines and join
        normalized = '\n'.join(line for line in lines if line)
        return normalized
    
    @staticmethod
    def _tokenize_code(code: str) -> List[str]:
        """Tokenize code into meaningful tokens"""
        # Split by common Python separators
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [token for token in tokens if token.strip()]
    
    @staticmethod
    def _extract_ast_structure(tree) -> List[str]:
        """Extract structural elements from AST"""
        structure = []
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            structure.append(node_type)
            
            # Add more specific information for certain nodes
            if isinstance(node, ast.FunctionDef):
                structure.append(f"func:{node.name}")
            elif isinstance(node, ast.Name):
                structure.append(f"name:{node.id}")
            elif isinstance(node, ast.Constant):
                structure.append(f"const:{type(node.value).__name__}")
        
        return structure

class CodeValidationAgent:
    def __init__(self, api_key: str, config: Config = None):
        if config is None:
            config = Config()
        
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            google_api_key=api_key
        )
        
    def validate_code(self, original_code: str, fixed_code: str, algorithm_name: str, 
                     bug_analysis: str = None) -> dict:
        """Validate the fixed code and provide final corrections if needed"""
        
        # First check syntax
        syntax_valid = self._check_syntax(fixed_code)
        
        context = f"\n\nOriginal Bug Analysis:\n{bug_analysis}" if bug_analysis else ""
        
        prompt = f"""You are an expert software engineer specializing in code validation and quality assurance. 
You have extensive experience validating Python code fixes, particularly for the QuixBugs dataset. 
You excel at identifying remaining issues in supposedly fixed code and providing final corrections 
to ensure algorithmic correctness.

Validate the following fixed Python code for algorithm '{algorithm_name}' and ensure it's completely correct.

Original Buggy Code:
```python
{original_code}
```

Fixed Code to Validate:
```python
{fixed_code}
```
{context}

VALIDATION CHECKLIST:
1. Syntax correctness ✓ (Pre-checked: {'Valid' if syntax_valid else 'Invalid'})
2. Logic correctness - Does the algorithm logic make sense?
3. Bug fix verification - Is the original bug actually fixed?
4. Edge case handling - Consider boundary conditions
5. Variable usage - All variables properly defined and used
6. Return behavior - Function returns expected type/value
7. Algorithm integrity - Original algorithm intent preserved

YOUR TASK:
- If the code is completely correct and the bug is properly fixed, respond with "VALIDATION: PASSED"
- If there are any issues, provide the corrected version of the code

OUTPUT FORMAT:
VALIDATION: [PASSED/FAILED]
ISSUES: [List any problems found, or "None" if passed]
CORRECTED_CODE: [Only include this section if validation failed - provide the fully corrected code]

Focus on ensuring the algorithm works correctly for all intended use cases."""
        
        try:
            # Get response from LLM
            response = self.llm.invoke(prompt)
            validation_text = response.content
            
            # Parse validation response
            passed = "VALIDATION: PASSED" in validation_text
            
            # Extract corrected code if provided
            final_code = fixed_code
            if "CORRECTED_CODE:" in validation_text:
                parts = validation_text.split("CORRECTED_CODE:")
                if len(parts) > 1:
                    corrected_section = parts[1].strip()
                    final_code = self._clean_generated_code(corrected_section)
            
            return {
                'success': True,
                'validation_passed': passed,
                'syntax_valid': syntax_valid,
                'final_code': final_code,
                'validation_response': validation_text,
                'algorithm': algorithm_name
            }
            
        except Exception as e:
            logger.error(f"Code validation failed for {algorithm_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'syntax_valid': syntax_valid,
                'final_code': fixed_code,
                'algorithm': algorithm_name
            }
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and extract Python code from response"""
        # Remove markdown code blocks
        if '```python' in code:
            parts = code.split('```python')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                return code_part.strip()
        elif '```' in code:
            parts = code.split('```')
            if len(parts) >= 3:
                code_blocks = [parts[i] for i in range(1, len(parts), 2)]
                code_part = max(code_blocks, key=len)
                return code_part.strip()
        
        # Extract function definition
        lines = code.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                code_lines.append(line)
            elif in_function:
                if line.strip() == '' or line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                else:
                    code_lines.append(line)
        
        return '\n'.join(code_lines).strip() if code_lines else code.strip()

class CodeValidationWorker:
    def __init__(self, api_key: str, input_dir: str = "python_programs", 
                 repair_dir: str = "repair_results", validation_dir: str = "validation_results",
                 fixed_programs_dir: str = "fixed_programs", correct_programs_dir: str = "correct_python_programs"):
        self.input_dir = input_dir
        self.repair_dir = repair_dir
        self.validation_dir = validation_dir
        self.fixed_programs_dir = fixed_programs_dir
        self.correct_programs_dir = correct_programs_dir
        
        # Create output directories
        os.makedirs(validation_dir, exist_ok=True)
        os.makedirs(fixed_programs_dir, exist_ok=True)
        
        # Initialize agent
        self.agent = CodeValidationAgent(api_key)
        
        # Initialize result tracking
        self.results_summary = []
        
    def get_ready_files(self):
        """Get list of files that have repair results but haven't been validated yet"""
        ready_files = []
        
        if not os.path.exists(self.repair_dir):
            return ready_files
            
        # Find files with completed repairs
        for filename in os.listdir(self.repair_dir):
            if filename.endswith('_repair.json'):
                algorithm_name = filename.replace('_repair.json', '')
                
                # Check if validation result already exists
                validation_file = os.path.join(self.validation_dir, f"{algorithm_name}_validation.json")
                if not os.path.exists(validation_file):
                    
                    # Check if repair was successful
                    repair_file = os.path.join(self.repair_dir, filename)
                    try:
                        with open(repair_file, 'r') as f:
                            repair_data = json.load(f)
                        if repair_data.get('success', False):
                            ready_files.append(algorithm_name)
                    except:
                        logger.warning(f"Could not read repair file: {repair_file}")
        
        return ready_files
    
    def load_analysis(self, algorithm_name: str):
        """Load bug analysis for an algorithm"""
        analysis_file = os.path.join("analysis_results", f"{algorithm_name}_analysis.json")
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            return analysis_data.get('analysis', '')
        except Exception as e:
            logger.warning(f"Could not load analysis for {algorithm_name}: {e}")
            return None
    
    def load_fixed_code(self, algorithm_name: str):
        """Load fixed code from repair results"""
        repair_file = os.path.join(self.repair_dir, f"{algorithm_name}_repair.json")
        try:
            with open(repair_file, 'r') as f:
                repair_data = json.load(f)
            return repair_data.get('fixed_code', '')
        except Exception as e:
            logger.error(f"Could not load fixed code for {algorithm_name}: {e}")
            return None
    
    def load_correct_code(self, algorithm_name: str):
        """Load correct reference code if available"""
        if not os.path.exists(self.correct_programs_dir):
            return None
            
        correct_file = os.path.join(self.correct_programs_dir, f"{algorithm_name}.py")
        try:
            with open(correct_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load correct code for {algorithm_name}: {e}")
            return None
    
    def save_text_results(self, algorithm_name: str, result: dict, similarity_metrics: dict = None):
        """Save validation results to text file"""
        text_file = os.path.join(self.validation_dir, f"{algorithm_name}_validation.txt")
        
        with open(text_file, 'w') as f:
            f.write(f"CODE VALIDATION REPORT\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 50}\n\n")
            
            f.write(f"VALIDATION STATUS: {'PASSED' if result.get('validation_passed', False) else 'FAILED'}\n")
            f.write(f"SYNTAX VALID: {'YES' if result.get('syntax_valid', False) else 'NO'}\n")
            f.write(f"PROCESSING SUCCESS: {'YES' if result.get('success', False) else 'NO'}\n\n")
            
            if not result.get('success', False):
                f.write(f"ERROR: {result.get('error', 'Unknown error')}\n\n")
            
            f.write(f"VALIDATION RESPONSE:\n")
            f.write(f"{'-' * 30}\n")
            f.write(f"{result.get('validation_response', 'No response available')}\n\n")
            
            if similarity_metrics:
                f.write(f"SIMILARITY METRICS (vs Correct Code):\n")
                f.write(f"{'-' * 30}\n")
                for metric, value in similarity_metrics.items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
                f.write(f"\n")
            
            f.write(f"FINAL CODE:\n")
            f.write(f"{'-' * 30}\n")
            f.write(result.get('final_code', 'No code available'))
            f.write(f"\n\n{'=' * 50}\n")
    
    def process_file(self, algorithm_name: str):
        """Process a single algorithm file"""
        logger.info(f"Validating {algorithm_name}")
        
        # Read original buggy code
        input_file = os.path.join(self.input_dir, f"{algorithm_name}.py")
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
            
        with open(input_file, 'r') as f:
            original_code = f.read()
        
        # Load fixed code
        fixed_code = self.load_fixed_code(algorithm_name)
        if not fixed_code:
            logger.error(f"Could not load fixed code for {algorithm_name}")
            return False
        
        # Load bug analysis
        bug_analysis = self.load_analysis(algorithm_name)
        
        # Validate code
        result = self.agent.validate_code(original_code, fixed_code, algorithm_name, bug_analysis)
        
        # Calculate similarity with correct code if available
        similarity_metrics = None
        correct_code = self.load_correct_code(algorithm_name)
        if correct_code:
            similarity_metrics = SimilarityScorer.calculate_similarity(
                result['final_code'], correct_code
            )
            result['similarity_metrics'] = similarity_metrics
        
        # Save validation result (JSON)
        output_file = os.path.join(self.validation_dir, f"{algorithm_name}_validation.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save validation result (Text)
        self.save_text_results(algorithm_name, result, similarity_metrics)
        
        # Save final fixed code
        if result['success']:
            fixed_code_file = os.path.join(self.fixed_programs_dir, f"{algorithm_name}.py")
            with open(fixed_code_file, 'w') as f:
                f.write(result['final_code'])
            
            validation_status = "PASSED" if result.get('validation_passed', False) else "NEEDS_REVIEW"
            logger.info(f"✓ Validation complete for {algorithm_name} - {validation_status}")
            
            # Add to results summary
            self.results_summary.append({
                'algorithm': algorithm_name,
                'status': validation_status,
                'syntax_valid': result.get('syntax_valid', False),
                'similarity': similarity_metrics.get('overall_similarity', 0.0) if similarity_metrics else 0.0
            })
            
            return True
        else:
            logger.error(f"✗ Validation failed for {algorithm_name}: {result.get('error', 'Unknown error')}")
            
            # Add to results summary
            self.results_summary.append({
                'algorithm': algorithm_name,
                'status': 'FAILED',
                'syntax_valid': result.get('syntax_valid', False),
                'similarity': 0.0,
                'error': result.get('error', 'Unknown error')
            })
            
            return False
    
    def save_summary_report(self, total_processed: int, successful: int, failed: int, passed_validations: int):
        """Save overall summary report to text file"""
        summary_file = os.path.join(self.validation_dir, "validation_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"CODE VALIDATION SUMMARY REPORT\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")
            
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"{'-' * 30}\n")
            f.write(f"Total Files Processed: {total_processed}\n")
            f.write(f"Successful Validations: {successful}\n")
            f.write(f"Failed Validations: {failed}\n")
            f.write(f"Passed Final Validation: {passed_validations}\n")
            f.write(f"Success Rate: {successful/max(total_processed, 1)*100:.1f}%\n")
            f.write(f"Pass Rate: {passed_validations/max(total_processed, 1)*100:.1f}%\n\n")
            
            f.write(f"DETAILED RESULTS:\n")
            f.write(f"{'-' * 30}\n")
            
            for result in self.results_summary:
                f.write(f"Algorithm: {result['algorithm']}\n")
                f.write(f"  Status: {result['status']}\n")
                f.write(f"  Syntax Valid: {'YES' if result['syntax_valid'] else 'NO'}\n")
                f.write(f"  Similarity Score: {result['similarity']:.4f}\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                f.write(f"\n")
            
            f.write(f"{'=' * 60}\n")
            f.write(f"Fixed programs saved to: {self.fixed_programs_dir}/\n")
            f.write(f"Validation results saved to: {self.validation_dir}/\n")
    
    def run(self):
        """Run the code validation worker"""
        print("Starting Code Validation Agent (Gemini 1.5 Flash)")
        print("=" * 60)
        
        ready_files = self.get_ready_files()
        if not ready_files:
            print("No files ready for validation! Run code repair first.")
            return
            
        print(f"Found {len(ready_files)} files ready for validation")
        
        successful = 0
        failed = 0
        passed_validations = 0
        
        for i, algorithm_name in enumerate(ready_files):
            print(f"Processing {i+1}/{len(ready_files)}: {algorithm_name}")
            
            if self.process_file(algorithm_name):
                successful += 1
                
                # Check if validation passed
                validation_file = os.path.join(self.validation_dir, f"{algorithm_name}_validation.json")
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                    if validation_data.get('validation_passed', False):
                        passed_validations += 1
                except:
                    pass
            else:
                failed += 1
        
        # Save summary report
        self.save_summary_report(len(ready_files), successful, failed, passed_validations)
        
        print("\n" + "=" * 60)
        print("CODE VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total processed: {len(ready_files)}")
        print(f"Successful validations: {successful}")
        print(f"Failed validations: {failed}")
        print(f"Passed final validation: {passed_validations}")
        print(f"Success rate: {successful/len(ready_files)*100:.1f}%")
        print(f"Pass rate: {passed_validations/len(ready_files)*100:.1f}%")
        print(f"Fixed programs saved to: {self.fixed_programs_dir}/")
        print(f"Summary report saved to: {os.path.join(self.validation_dir, 'validation_summary.txt')}")

def main():
    # Gemini API keys (using the first one by default)
    GEMINI_API_KEYS = [
        "AIzaSyBE--FDVdwSZJNZlVQ6CG1y2HoNYJ8Se4k",
        "AIzaSyAJ6oxQy3HdJHdiohj2NktSDCiLZxYaQRs",
        "AIzaSyDKuD8bm8WPvjfYd2UsXP7rG13ozqWl0MI"
    ]
    
    # Use the first API key, you can rotate or add logic to switch between them
    API_KEY = GEMINI_API_KEYS[0]
    
    worker = CodeValidationWorker(API_KEY)
    worker.run()

if __name__ == "__main__":
    main()
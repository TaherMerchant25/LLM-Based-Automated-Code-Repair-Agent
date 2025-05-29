import os
import google.generativeai as genai
import logging
import time
import json
from pathlib import Path
import random
import threading
from queue import Queue
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 15
    base_delay: float = 4.0
    max_delay: float = 30.0
    jitter_range: float = 1.0

class APIRateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = Queue()
        self.lock = threading.Lock()
        self.consecutive_errors = 0
        self.last_request_time = 0
        
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            
            # Clean old request times
            while not self.request_times.empty():
                if now - self.request_times.queue[0] > 60:
                    self.request_times.get()
                else:
                    break
            
            # Check rate limit
            if self.request_times.qsize() >= self.config.requests_per_minute:
                wait_time = 60 - (now - self.request_times.queue[0])
                if wait_time > 0:
                    logger.info(f"Rate limit approaching, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
            
            # Calculate delay with backoff
            delay = self.config.base_delay
            if self.consecutive_errors > 0:
                delay = min(
                    self.config.base_delay * (2 ** self.consecutive_errors),
                    self.config.max_delay
                )
            
            # Add jitter
            jitter = random.uniform(-self.config.jitter_range, self.config.jitter_range)
            total_delay = max(0, delay + jitter)
            
            # Ensure minimum time between requests
            time_since_last = now - self.last_request_time
            if time_since_last < total_delay:
                sleep_time = total_delay - time_since_last
                logger.debug(f"Waiting {sleep_time:.1f}s for rate limiting")
                time.sleep(sleep_time)
            
            self.request_times.put(time.time())
            self.last_request_time = time.time()
    
    def record_success(self):
        with self.lock:
            self.consecutive_errors = max(0, self.consecutive_errors - 1)
    
    def record_error(self, is_rate_limit_error: bool = False):
        with self.lock:
            if is_rate_limit_error:
                self.consecutive_errors += 2
            else:
                self.consecutive_errors += 1
            logger.warning(f"API error recorded. Consecutive errors: {self.consecutive_errors}")

class CodeRepairAgent:
    def __init__(self, api_key: str, rate_limiter: APIRateLimiter, model_name: str = 'gemini-1.5-flash'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.rate_limiter = rate_limiter
        self.agent_name = "Code Repair Agent"
        
    def repair_code(self, buggy_code: str, algorithm_name: str, bug_analysis: str = None) -> dict:
        self.rate_limiter.wait_if_needed()
        
        context = f"\nBug Analysis:\n{bug_analysis}" if bug_analysis else ""
        
        prompt = f"""You are a code repair specialist. Fix the bug in this Python code with minimal changes.

Algorithm: {algorithm_name}
Original Code:
```python
{buggy_code}
```
{context}

REPAIR REQUIREMENTS:
1. Make the MINIMAL change necessary to fix the bug
2. Preserve original structure, variable names, and formatting
3. Ensure the fix addresses the root cause
4. Maintain algorithmic correctness
5. Do NOT add comments or unnecessary modifications

Common fixes for QuixBugs patterns:
- range(n) → range(n-1) or range(n+1)
- < → <= or > → >= 
- arr[i] → arr[i-1] or arr[i+1]
- while condition → correct termination logic
- variable_name → correct_variable_name

OUTPUT: Return ONLY the corrected Python code, no explanations."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    candidate_count=1,
                    max_output_tokens=1024
                )
            )
            
            self.rate_limiter.record_success()
            fixed_code = self._clean_generated_code(response.text.strip())
            
            return {
                'success': True,
                'fixed_code': fixed_code,
                'original_response': response.text,
                'agent': self.agent_name,
                'algorithm': algorithm_name
            }
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(term in error_msg.lower() for term in ['429', 'quota', 'rate', 'limit'])
            self.rate_limiter.record_error(is_rate_limit)
            
            logger.error(f"Code repair failed for {algorithm_name}: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'agent': self.agent_name,
                'algorithm': algorithm_name
            }
    
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

class CodeRepairWorker:
    def __init__(self, api_key: str, input_dir: str = "python_programs", 
                 analysis_dir: str = "analysis_results", output_dir: str = "fixed_programs"):
        self.input_dir = input_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure rate limiter
        rate_config = RateLimitConfig(requests_per_minute=12, base_delay=5.0)
        self.rate_limiter = APIRateLimiter(rate_config)
        
        # Initialize agent
        self.agent = CodeRepairAgent(api_key, self.rate_limiter)
        
    def get_ready_files(self):
        """Get list of files that have analysis results but haven't been repaired yet"""
        ready_files = []
        
        if not os.path.exists(self.analysis_dir):
            return ready_files
            
        # Find files with completed analysis
        for filename in os.listdir(self.analysis_dir):
            if filename.endswith('_analysis.json'):
                algorithm_name = filename.replace('_analysis.json', '')
                
                # Check if repair result already exists
                repair_file = os.path.join(self.output_dir, f"{algorithm_name}_repair.json")
                if not os.path.exists(repair_file):
                    
                    # Check if analysis was successful
                    analysis_file = os.path.join(self.analysis_dir, filename)
                    try:
                        with open(analysis_file, 'r') as f:
                            analysis_data = json.load(f)
                        if analysis_data.get('success', False):
                            ready_files.append(algorithm_name)
                    except:
                        logger.warning(f"Could not read analysis file: {analysis_file}")
        
        return ready_files
    
    def load_analysis(self, algorithm_name: str):
        """Load bug analysis for an algorithm"""
        analysis_file = os.path.join(self.analysis_dir, f"{algorithm_name}_analysis.json")
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            return analysis_data.get('analysis', '')
        except Exception as e:
            logger.warning(f"Could not load analysis for {algorithm_name}: {e}")
            return None
        
    def process_file(self, algorithm_name: str):
        """Process a single algorithm file"""
        logger.info(f"Repairing {algorithm_name}")
        
        # Read buggy code
        input_file = os.path.join(self.input_dir, f"{algorithm_name}.py")
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
            
        with open(input_file, 'r') as f:
            buggy_code = f.read()
        
        # Load bug analysis
        bug_analysis = self.load_analysis(algorithm_name)
        
        # Repair code
        result = self.agent.repair_code(buggy_code, algorithm_name, bug_analysis)
        
        # Save result
        output_file = os.path.join(self.output_dir, f"{algorithm_name}_repair.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save fixed code separately
        if result['success']:
            fixed_code_file = os.path.join(self.output_dir, f"{algorithm_name}_fixed.py")
            with open(fixed_code_file, 'w') as f:
                f.write(result['fixed_code'])
            logger.info(f"✓ Repair complete for {algorithm_name}")
            return True
        else:
            logger.error(f"✗ Repair failed for {algorithm_name}: {result.get('error', 'Unknown error')}")
            return False
    
    def run(self):
        """Run the code repair worker"""
        print("Starting Code Repair Agent")
        print("=" * 50)
        
        ready_files = self.get_ready_files()
        if not ready_files:
            print("No files ready for repair! Run bug analysis first.")
            return
            
        print(f"Found {len(ready_files)} files ready for repair")
        
        successful = 0
        failed = 0
        
        for i, algorithm_name in enumerate(ready_files):
            print(f"Processing {i+1}/{len(ready_files)}: {algorithm_name}")
            
            if self.process_file(algorithm_name):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "=" * 50)
        print("CODE REPAIR RESULTS")
        print("=" * 50)
        print(f"Total processed: {len(ready_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(ready_files)*100:.1f}%")

def main():
    # REPLACE WITH YOUR API KEY
    API_KEY_2 = ""  # Replace with your second API key
    
    if API_KEY_2 == "YOUR_GEMINI_API_KEY_2":
        print("Please set your Gemini API key in the script")
        return
    
    worker = CodeRepairWorker(API_KEY_2)
    worker.run()

if __name__ == "__main__":
    main()

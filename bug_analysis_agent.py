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

class BugAnalysisAgent:
    def __init__(self, api_key: str, rate_limiter: APIRateLimiter, model_name: str = 'gemini-1.5-flash'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.rate_limiter = rate_limiter
        self.agent_name = "Bug Analysis Agent"
        
    def analyze_bug(self, buggy_code: str, algorithm_name: str) -> dict:
        self.rate_limiter.wait_if_needed()
        
        prompt = f"""You are a bug analysis expert specializing in the QuixBugs dataset.

Algorithm: {algorithm_name}
Code to analyze:
```python
{buggy_code}
```

ANALYSIS TASK:
1. Understand what this algorithm should accomplish
2. Trace through the code logic step-by-step
3. Identify the specific bug location and type
4. Explain why it's incorrect and what the correct behavior should be

Focus on these common QuixBugs bug patterns:
- Off-by-one errors in array indexing or loop bounds
- Wrong comparison operators (< vs <=, > vs >=, == vs !=)
- Incorrect variable references or scope issues
- Wrong loop termination conditions
- Missing or incorrect increments/decrements
- Logical operator errors (and vs or)

REQUIRED OUTPUT FORMAT:
1. ALGORITHM PURPOSE: [Brief description]
2. BUG LOCATION: [Specific line number or code section]
3. BUG TYPE: [Classification of the bug]
4. PROBLEM DESCRIPTION: [Why it's wrong]
5. EXPECTED BEHAVIOR: [What should happen instead]
6. CONFIDENCE: [High/Medium/Low]

Be precise and focus only on the primary bug."""

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
            
            return {
                'success': True,
                'analysis': response.text.strip(),
                'agent': self.agent_name,
                'algorithm': algorithm_name
            }
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(term in error_msg.lower() for term in ['429', 'quota', 'rate', 'limit'])
            self.rate_limiter.record_error(is_rate_limit)
            
            logger.error(f"Bug analysis failed for {algorithm_name}: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'agent': self.agent_name,
                'algorithm': algorithm_name
            }

class BugAnalysisWorker:
    def __init__(self, api_key: str, input_dir: str = "python_programs", 
                 output_dir: str = "analysis_results"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure rate limiter
        rate_config = RateLimitConfig(requests_per_minute=12, base_delay=5.0)
        self.rate_limiter = APIRateLimiter(rate_config)
        
        # Initialize agent
        self.agent = BugAnalysisAgent(api_key, self.rate_limiter)
        
    def get_pending_files(self):
        """Get list of files that haven't been analyzed yet"""
        if not os.path.exists(self.input_dir):
            return []
            
        input_files = [f[:-3] for f in os.listdir(self.input_dir) 
                      if f.endswith('.py') and not f.endswith('_test.py')]
        
        # Filter out already processed files
        pending_files = []
        for filename in input_files:
            output_file = os.path.join(self.output_dir, f"{filename}_analysis.json")
            if not os.path.exists(output_file):
                pending_files.append(filename)
        
        return pending_files
        
    def process_file(self, algorithm_name: str):
        """Process a single algorithm file"""
        logger.info(f"Analyzing {algorithm_name}")
        
        # Read buggy code
        input_file = os.path.join(self.input_dir, f"{algorithm_name}.py")
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
            
        with open(input_file, 'r') as f:
            buggy_code = f.read()
        
        # Analyze bug
        result = self.agent.analyze_bug(buggy_code, algorithm_name)
        
        # Save result
        output_file = os.path.join(self.output_dir, f"{algorithm_name}_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if result['success']:
            logger.info(f"✓ Analysis complete for {algorithm_name}")
            return True
        else:
            logger.error(f"✗ Analysis failed for {algorithm_name}: {result.get('error', 'Unknown error')}")
            return False
    
    def run(self):
        """Run the bug analysis worker"""
        print("Starting Bug Analysis Agent")
        print("=" * 50)
        
        pending_files = self.get_pending_files()
        if not pending_files:
            print("No pending files to analyze!")
            return
            
        print(f"Found {len(pending_files)} files to analyze")
        
        successful = 0
        failed = 0
        
        for i, algorithm_name in enumerate(pending_files):
            print(f"Processing {i+1}/{len(pending_files)}: {algorithm_name}")
            
            if self.process_file(algorithm_name):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "=" * 50)
        print("BUG ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total processed: {len(pending_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(pending_files)*100:.1f}%")

def main():
    # REPLACE WITH YOUR API KEY
    API_KEY_1 = ""  # Replace with your first API key
    
    if API_KEY_1 == "YOUR_GEMINI_API_KEY_1":
        print("Please set your Gemini API key in the script")
        return
    
    worker = BugAnalysisWorker(API_KEY_1)
    worker.run()

if __name__ == "__main__":
    main()

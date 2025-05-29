import os
import sys
import time
import argparse
from pathlib import Path

from bug_analysis_agent import BugAnalysisWorker
from code_repair_agent import CodeRepairWorker  
from complete_validation_agent import CodeValidationWorker

class BugFixingPipeline:
    def __init__(self, api_keys: dict, input_dir: str = "python_programs"):
        self.api_keys = api_keys
        self.input_dir = input_dir
        
        required_keys = ['analysis', 'repair', 'validation']
        for key in required_keys:
            if key not in api_keys or not api_keys[key] or api_keys[key].startswith("YOUR_"):
                raise ValueError(f"Please provide a valid API key for {key}")
        
        self.analysis_worker = BugAnalysisWorker(api_keys['analysis'], input_dir)
        self.repair_worker = CodeRepairWorker(api_keys['repair'], input_dir)
        self.validation_worker = CodeValidationWorker(api_keys['validation'], input_dir)
    
    def run_analysis_phase(self):
        print("\nPHASE 1: BUG ANALYSIS")
        print("=" * 60)
        
        pending_files = self.analysis_worker.get_pending_files()
        if not pending_files:
            print("All files already analyzed")
            return True
        
        print(f"Analyzing {len(pending_files)} programs...")
        self.analysis_worker.run()
        return True
    
    def run_repair_phase(self):
        print("\nPHASE 2: CODE REPAIR")
        print("=" * 60)
        
        ready_files = self.repair_worker.get_ready_files()
        if not ready_files:
            print("All files already repaired or no analysis results available")
            return True
        
        print(f"Repairing {len(ready_files)} programs...")
        self.repair_worker.run()
        return True
    
    def run_validation_phase(self):
        print("\nPHASE 3: CODE VALIDATION")
        print("=" * 60)
        
        ready_files = self.validation_worker.get_ready_files()
        if not ready_files:
            print("All files already validated or no repair results available")
            return True
        
        print(f"Validating {len(ready_files)} programs...")
        self.validation_worker.run()
        return True
    
    def run_full_pipeline(self):
        print("STARTING QUIXBUGS BUG FIXING PIPELINE")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Using API keys: Analysis, Repair, Validation")
        
        start_time = time.time()
        
        try:
            if not self.run_analysis_phase():
                print("Analysis phase failed")
                return False
            
            if not self.run_repair_phase():
                print("Repair phase failed")
                return False
            
            if not self.run_validation_phase():
                print("Validation phase failed")
                return False
            
            elapsed = time.time() - start_time
            print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Total time: {elapsed:.1f} seconds")
            print(f"Fixed programs saved to: fixed_programs/")
            print(f"Run 'python test_fixed_only.py' to test the fixes")
            
            return True
            
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            return False
    
    def get_pipeline_status(self):
        status = {
            'analysis': {
                'pending': len(self.analysis_worker.get_pending_files()),
                'completed': 0
            },
            'repair': {
                'ready': len(self.repair_worker.get_ready_files()),
                'completed': 0
            },
            'validation': {
                'ready': len(self.validation_worker.get_ready_files()),
                'completed': 0
            }
        }
        
        for phase, directory in [('analysis', 'analysis_results'), 
                                ('repair', 'repair_results'),
                                ('validation', 'validation_results')]:
            if os.path.exists(directory):
                completed = len([f for f in os.listdir(directory) 
                               if f.endswith(f'_{phase.split("_")[0]}.json')])
                status[phase]['completed'] = completed
        
        return status

def print_status(pipeline):
    status = pipeline.get_pipeline_status()
    
    print("\nPIPELINE STATUS")
    print("=" * 40)
    print(f"Analysis:   {status['analysis']['completed']} completed, {status['analysis']['pending']} pending")
    print(f"Repair:     {status['repair']['completed']} completed, {status['repair']['ready']} ready")
    print(f"Validation: {status['validation']['completed']} completed, {status['validation']['ready']} ready")
    
    fixed_dir = "fixed_programs"
    if os.path.exists(fixed_dir):
        fixed_count = len([f for f in os.listdir(fixed_dir) if f.endswith('.py')])
        print(f"Fixed programs: {fixed_count} files in {fixed_dir}/")

def get_default_api_keys():
    keys = {}
    
    try:
        keys['analysis'] = "AIzaSyB8U-1zJJHQ_UU_6nDH-ktR7r1zxsaUEyo"
        keys['repair'] = "AIzaSyAJ6oxQy3HdJHdiohj2NktSDCiLZxYaQRs"
        keys['validation'] = "AIzaSyDKuD8bm8WPvjfYd2UsXP7rG13ozqWl0MI"
    except:
        pass
    
    return keys

def main():
    parser = argparse.ArgumentParser(description='QuixBugs Bug Fixing Pipeline')
    parser.add_argument('--input-dir', default='python_programs',
                       help='Directory containing buggy Python programs')
    parser.add_argument('--analysis-key', 
                       help='Gemini API key for bug analysis')
    parser.add_argument('--repair-key',
                       help='Gemini API key for code repair')  
    parser.add_argument('--validation-key',
                       help='Gemini API key for code validation')
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status and exit')
    parser.add_argument('--phase', choices=['analysis', 'repair', 'validation'],
                       help='Run only a specific phase')
    
    args = parser.parse_args()
    
    default_keys = get_default_api_keys()
    
    api_keys = {
        'analysis': (args.analysis_key or 
                    os.getenv('GEMINI_API_KEY_1') or 
                    default_keys.get('analysis', '')),
        'repair': (args.repair_key or 
                  os.getenv('GEMINI_API_KEY_2') or 
                  default_keys.get('repair', '')),
        'validation': (args.validation_key or 
                      os.getenv('GEMINI_API_KEY_3') or 
                      default_keys.get('validation', ''))
    }
    
    invalid_keys = []
    for key_name, key_value in api_keys.items():
        if not key_value or key_value.startswith("YOUR_") or len(key_value) < 10:
            invalid_keys.append(key_name)
    
    if invalid_keys:
        print(f"Configuration error: Invalid API keys for: {', '.join(invalid_keys)}")
        print("\nSet API keys using:")
        print("   - Command line: --analysis-key KEY1 --repair-key KEY2 --validation-key KEY3")
        print("   - Environment: GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3")
        print("   - Or edit the API_KEY variables in individual agent files")
        
        if default_keys:
            print(f"\nFound API keys in agent files:")
            for key_name, key_value in default_keys.items():
                if key_value and len(key_value) > 10:
                    print(f"   - {key_name}: {key_value[:20]}...")
                else:
                    print(f"   - {key_name}: INVALID")
        
        sys.exit(1)
    
    try:
        pipeline = BugFixingPipeline(api_keys, args.input_dir)
        
        if args.status:
            print_status(pipeline)
            return
        
        if args.phase:
            if args.phase == 'analysis':
                pipeline.run_analysis_phase()
            elif args.phase == 'repair':
                pipeline.run_repair_phase()
            elif args.phase == 'validation':
                pipeline.run_validation_phase()
        else:
            success = pipeline.run_full_pipeline()
            sys.exit(0 if success else 1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
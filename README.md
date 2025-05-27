# 🔧 QuixBugs AutoFix: Large-Scale Bug Repair with Gemini AI

## 🧠 Project Overview

**QuixBugs AutoFix** is a high-throughput, AI-powered pipeline designed to automatically fix buggy implementations in the [QuixBugs dataset](https://github.com/jkoppel/QuixBugs). It leverages the Gemini 1.5 model to repair Python functions that contain exactly one bug, using an intelligent prompt system and rigorous test-driven validation.

---

## ❓ Why This Problem Matters

Despite rapid advances in software development, debugging remains time-consuming and costly. The QuixBugs dataset represents common bug patterns that challenge compilers, students, and even seasoned developers.

Solving this problem can:
- Automate tedious debugging tasks
- Improve productivity and code quality
- Serve as a benchmark for AI-assisted program repair
- Advance research in software engineering and LLM-based debugging

---

## ⚙️ Pipeline Architecture

```
+---------------------+
| Buggy Python File |
+----------+----------+
|
v
+----------+----------+
| Gemini Prompt Agent |
+----------+----------+
|
v
+---------------------+
| Generated Fix |
| - Multi-attempts |
| - Syntax checks |
+----------+----------+
|
v
+--------------------------+
| Test-Driven Evaluation |
| using test_fixed_only |
+----------+--------------+
|
v
+---------------------------+
| Comparison with Correct |
| Reference Implementation |
+---------------------------+
```

---

## 🤖 Why Gemini Model Fits This Problem

- **Code-Aware Reasoning:** Gemini understands Python deeply and handles edge cases in QuixBugs such as off-by-one errors, condition miswrites, and loop bounds.
- **Prompt Tunability:** It supports multi-attempt logic with increasing reasoning complexity (direct fix → step-by-step ReAct debugging).
- **Fast & Accurate:** Using `gemini-1.5-flash` in standard mode gives excellent accuracy with low latency, suitable for large-scale batch repairs.

---

## 📝 Prompt Template

### Attempt 1 (Fast, Pattern-Based)
```text
Fix the single bug in this Python function.

Algorithm: {algorithm_name}
Code:
```
python
{buggy_code}
This code contains exactly one bug. Common QuixBugs patterns:

Off-by-one errors in loops or indexing

Wrong comparison operators (< vs <=, > vs >=)

Incorrect variable references

Wrong loop bounds or conditions

Missing/incorrect increments

Find the bug and return only the corrected Python code.


### Attempt 2 (Detailed ReAct)
```text
Debug this Python function step by step.

Algorithm: {algorithm_name}
Code:
```
python
{buggy_code}
REASONING:

What should this algorithm do?

Trace through the code with a simple input

Identify where the logic fails

What is the minimal fix needed?

ACTION:
Return only the corrected Python code with the single bug fixed.


---

## ✅ Test-Driven Validation Workflow

Validation happens in **three stages**:

1. **Unit Testing:** Each fixed function is tested using predefined I/O pairs from `json_testcases/` using `test_fixed_only.py`.

2. **Functional Comparison:** The fixed output is compared against the `correct_python_programs/` ground truth using:
   - Exact match
   - Syntax validation
   - AST similarity
   - Line-level diffs

3. **Execution Sanity:** Edge case timeout and exception handling using multiprocessing from `tester.py`.

---

## 📈 Results (from last run)

- **Mode:** Sequential Fast
- **Model:** gemini-1.5-flash
- **Accuracy Mode:** Standard

| Metric                         | Value         |
|-------------------------------|---------------|
| Total Programs                | 41            |
| Successfully Fixed            | 40            |
| **Success Rate**              | **97.56%**    |
| Avg. Similarity (Correct Code)| 62.07%        |
| Exact Matches                 | 4             |
| High Similarity (>80%)        | 12            |
| Processing Speed              | 22.1 programs/min |

📝 For detailed output, see: `quixbugs_report_20250527_225337.txt`

---

## 🧪 Run It Yourself

### 1. Prerequisites
```bash
pip install google-generativeai
export GOOGLE_API_KEY="your_gemini_api_key"
```
### 2. Run Fixer
```bash
python code_corrector.py
```
### 3. Run Tests
```bash
python test_fixed_only.py          # all programs
python test_fixed_only.py gcd      # specific program
```
### 📁 Folder Structure
```yaml
.
├── python_programs/              # Buggy QuixBugs programs
├── fixed_programs/               # AI-generated fixes
├── correct_python_programs/     # Ground-truth correct solutions
├── json_testcases/              # I/O tests for each program
├── code_corrector.py            # Main LLM repair pipeline
├── test_fixed_only.py           # Standalone test validator
├── tester.py                    # Timeout-enabled executor
└── quixbugs_report_<timestamp>.txt  # Evaluation report
```
### 📬 Contact
```
Created by Taher Merchant.
Feel free to open issues or contribute with PRs!
```

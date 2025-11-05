#!/usr/bin/env python3
"""
MATH Dataset Loader - Competition-level math problems

The MATH dataset contains 12,500 competition mathematics problems.
These are MUCH harder than GSM8K!
"""

import json
import os
import re
import math
from pathlib import Path

def load_math_problems(num_problems=50, difficulty="Level 1", shuffle=True):
    """
    Load problems from MATH dataset.
    
    Args:
        num_problems: Number of problems to load
        difficulty: "Level 1" (easiest) to "Level 5" (hardest)
        shuffle: Whether to shuffle problems
    
    Returns:
        List of problem dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("Run: pip install datasets")
        return []
    
    print(f"📥 Loading MATH dataset (Difficulty: {difficulty})...")
    
    # Load from Hugging Face
    # qwedsacf/competition_math is the original MATH dataset
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    
    # Filter by difficulty level
    filtered = [p for p in dataset if p['level'] == difficulty]
    
    print(f"✅ Found {len(filtered)} problems at {difficulty}")
    
    if shuffle:
        import random
        random.shuffle(filtered)
    
    # Take requested number
    problems = filtered[:num_problems]
    
    # Format for our solver
    formatted_problems = []
    for i, problem in enumerate(problems):
        formatted_problems.append({
            'question': problem['problem'],
            'answer': extract_boxed_answer(problem['solution']),
            'full_solution': problem['solution'],
            'subject': problem.get('type', 'Unknown'),
            'level': problem['level']
        })
    
    return formatted_problems

def extract_boxed_answer(solution_text):
    """Extract answer from \\boxed{answer} format."""
    # Look for \boxed{...} with proper brace matching
    match = re.search(r'\\boxed\{', solution_text)
    if match:
        start = match.end()
        depth = 1
        i = start
        while i < len(solution_text) and depth > 0:
            if solution_text[i] == '{':
                depth += 1
            elif solution_text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return solution_text[start:i-1].strip()
    
    # Fallback: look for final number
    numbers = re.findall(r'-?\d+\.?\d*', solution_text)
    if numbers:
        return numbers[-1]
    
    return "UNABLE_TO_PARSE"

def check_answer(student_answer, correct_answer):
    """Check if student answer matches correct answer."""
    def normalize(ans):
        ans = str(ans).strip()
        ans = re.sub(r'\\text\{([^}]+)\}', r'\1', ans)
        ans = re.sub(r'\\boxed\{([^}]+)\}', r'\1', ans)
        ans = ans.replace('$', '').replace('\\\\', '').replace('\\', '')
        return ans.lower()
    
    student_norm = normalize(student_answer)
    correct_norm = normalize(correct_answer)
    
    if student_norm == correct_norm:
        return True
    
    # Try numerical comparison (handles decimals vs fractions)
    try:
        # Try to convert student answer to float
        student_val = float(student_answer)
        
        # Try to evaluate correct answer if it's a fraction
        if 'frac' in correct_norm:
            # Extract numerator and denominator from \frac{num}{den}
            frac_match = re.search(r'frac\{([^}]+)\}\{([^}]+)\}', correct_norm)
            if frac_match:
                num = eval(frac_match.group(1).replace('sqrt', 'math.sqrt').replace('pi', 'math.pi'))
                den = eval(frac_match.group(2).replace('sqrt', 'math.sqrt').replace('pi', 'math.pi'))
                correct_val = num / den
                return abs(student_val - correct_val) < 0.0001
        
        # Try direct float conversion
        correct_val = float(correct_answer)
        return abs(student_val - correct_val) < 0.0001
    except:
        pass
    
    # Try extracting numbers
    try:
        student_nums = re.findall(r'-?\d+\.?\d*', student_norm)
        correct_nums = re.findall(r'-?\d+\.?\d*', correct_norm)
        
        if student_nums and correct_nums:
            return student_nums[-1] == correct_nums[-1]
    except:
        pass
    
    return False

if __name__ == "__main__":
    print("Testing MATH dataset loader...")
    problems = load_math_problems(num_problems=5, difficulty="Level 1")
    
    if problems:
        print(f"\n✅ Loaded {len(problems)} problems!")
        print("\nSample:")
        print(f"Question: {problems[0]['question'][:150]}...")
        print(f"Answer: {problems[0]['answer']}")

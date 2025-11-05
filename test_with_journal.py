"""
Test Linguistic RL (In-Context Journal) vs Baseline

Compare:
- Baseline (no help)
- Linguistic RL (maintains strategy journal in-context)
"""

import json
import requests
from math_dataset_loader import load_math_problems, check_answer
from datetime import datetime

class JournalTester:
    """Test with in-context strategy journal"""
    
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name
        self.strategy_journal = []
    
    def call_model(self, prompt: str) -> str:
        """Call Ollama"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    
    def solve_with_journal(self, problem: str, problem_num: int) -> dict:
        """
        Solve with Linguistic RL - maintain strategy journal
        """
        # Build journal context
        journal_context = ""
        if self.strategy_journal:
            journal_context = "\n\nSTRATEGY JOURNAL (what you've learned):\n"
            for entry in self.strategy_journal[-5:]:  # Last 5 entries
                journal_context += f"- {entry}\n"
        
        prompt = f"""You are solving math problems and learning from each attempt.
{journal_context}

PROBLEM #{problem_num}: {problem}

Solve this problem, then reflect on what worked or what you learned.

Format:
ANSWER: [your answer in a box, e.g., \\boxed{{42}}]
REASONING: [your work]
REFLECTION: [what you learned or what strategy helped]
"""
        
        response = self.call_model(prompt)
        
        # Parse response
        import re
        
        answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        reflection_match = re.search(r'REFLECTION:\s*(.+?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        reflection = reflection_match.group(1).strip() if reflection_match else ""
        
        # Add to journal if reflection is useful
        if reflection and len(reflection) > 20:
            self.strategy_journal.append(reflection)
        
        return {
            'answer': answer,
            'response': response,
            'reflection': reflection
        }
    
    def solve_baseline(self, problem: str) -> dict:
        """
        Solve WITHOUT journal (baseline)
        """
        prompt = f"""Solve this math problem:

{problem}

Provide your answer in a box: \\boxed{{answer}}
"""
        
        response = self.call_model(prompt)
        
        # Extract answer
        import re
        answer_match = re.search(r'\\boxed\{([^}]+)\}', response)
        answer = answer_match.group(1) if answer_match else ""
        
        return {
            'answer': answer,
            'response': response
        }
    
    def run_test(self, num_problems: int = 20, use_journal: bool = True):
        """Run test"""
        
        method = "Linguistic RL (Journal)" if use_journal else "Baseline (No Journal)"
        print(f"\n{'='*70}")
        print(f"🧪 TESTING: {method}")
        print(f"{'='*70}\n")
        
        problems = load_math_problems(num_problems, "Level 2", shuffle=True)
        
        results = {
            'correct': 0,
            'total': num_problems,
            'method': method,
            'problems': []
        }
        
        for i, prob in enumerate(problems):
            print(f"Problem {i+1}/{num_problems}...", end=' ')
            
            if use_journal:
                result = self.solve_with_journal(prob['question'], i+1)
            else:
                result = self.solve_baseline(prob['question'])
            
            is_correct = check_answer(result['answer'], prob['answer'])
            
            if is_correct:
                results['correct'] += 1
                print("✓")
            else:
                print("✗")
            
            results['problems'].append({
                'question': prob['question'][:100] + "...",
                'correct_answer': prob['answer'],
                'model_answer': result['answer'],
                'correct': is_correct,
                'reflection': result.get('reflection', '')
            })
        
        accuracy = results['correct'] / num_problems * 100
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTS: {method}")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy:.1f}% ({results['correct']}/{num_problems})")
        print(f"Strategy Journal Entries: {len(self.strategy_journal)}")
        print(f"{'='*70}\n")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_{'journal' if use_journal else 'baseline'}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved to: {filename}\n")
        
        return results


def main():
    """
    Run comparison test
    """
    import sys
    
    num_problems = 20
    if len(sys.argv) > 1:
        num_problems = int(sys.argv[1])
    
    print("\n" + "="*70)
    print("🔬 LINGUISTIC RL COMPARISON TEST")
    print("="*70)
    print(f"Testing with {num_problems} problems")
    print("="*70 + "\n")
    
    # Test 1: Baseline
    print("TEST 1: Baseline (no journal)")
    tester_baseline = JournalTester()
    baseline_results = tester_baseline.run_test(num_problems, use_journal=False)
    
    # Test 2: With Journal
    print("TEST 2: Linguistic RL (with strategy journal)")
    tester_journal = JournalTester()
    journal_results = tester_journal.run_test(num_problems, use_journal=True)
    
    # Comparison
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    print(f"Baseline:      {baseline_results['correct']}/{num_problems} = {baseline_results['correct']/num_problems*100:.1f}%")
    print(f"Linguistic RL: {journal_results['correct']}/{num_problems} = {journal_results['correct']/num_problems*100:.1f}%")
    
    improvement = (journal_results['correct'] - baseline_results['correct']) / num_problems * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("✅ Linguistic RL helps!")
    elif improvement < 0:
        print("⚠️ Linguistic RL hurts (learning overhead?)")
    else:
        print("➖ No difference")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

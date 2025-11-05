# Linguistic RL for Stochastic Environments

**Teaching AI to thrive in uncertainty through natural language**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 The Core Problem

Traditional Reinforcement Learning fails in stochastic environments (financial markets, robotics, medical diagnosis) because:

- **Scalar rewards can't capture regime changes**
- **Models overfit to current conditions**
- **Catastrophic forgetting when environments shift**
- **No way to preserve multiple strategies simultaneously**

Existing solutions (ensemble models, explicit regime detectors, complex architectures) are brittle and don't generalize.

## 💡 The Breakthrough

**Frame the stochastic nature in natural language and let the AI figure out the solution.**

Instead of programming regime detection and strategy switching, we simply explain:

> *"You're in a stochastic environment. Markets change. When something stops working, don't delete it—ask if the regime changed. Build complementary approaches that work across different conditions."*

The model leverages its pretrained knowledge about regimes, diversification, and risk management to **design its own robust system**.

---

## 🔥 Why This Works

### LLMs Already Understand:
- Market regimes (bull, bear, range-bound, volatile)
- Overfitting and its dangers
- Portfolio theory and diversification
- Risk management principles
- Meta-reasoning about learning

### Natural Language Enables:
1. **Meta-reasoning**: Not just "do this" but "understand WHY"
2. **Composability**: Build on previous understanding
3. **Self-awareness**: Reason about own learning process
4. **Knowledge preservation**: "Don't delete what works, add to it"

---

## 📊 Example: Trading AI Evolution

### Generation 0 (Naive):
```
Journal: "Buying breakouts works! Made 5% this week."
Strategy: Simple breakout trading
```

### Generation 1 (Regime Change):
```
Journal: "Breakout strategy stopped working. Lost 3% this week.
Wait—the REGIME changed! Market went from trending to range-bound.
My strategy isn't broken, it's just not appropriate right now."

Action: Preserve breakout logic, develop mean-reversion for ranges
```

### Generation 2 (Building Portfolio):
```
Journal: "I now have complementary strategies:
- Breakout (for trending regimes)
- Mean reversion (for range-bound regimes)
Need regime detector. Testing ADX + volatility..."
```

### Generation N (Robust System):
```
Journal: "Multi-strategy system operational:
- 5 complementary strategies
- Regime detection (ADX + volatility + breadth)
- Dynamic strategy weighting
- Each strategy documents its win conditions
- System survives regime changes through adaptation"
```

**The AI taught itself regime-aware trading through natural language reflection!**

---

## 🧬 Self-Modifying Architecture

Building on [Linguistic RL](https://github.com/DRawson5570/linguistic-rl), we add autonomous self-modification:

1. **Test** current performance
2. **Detect** weaknesses via self-reflection  
3. **Generate** training data with insights
4. **Train** LoRA adapter with new knowledge
5. **Hot-swap** to improved weights
6. **Repeat** → Continuous evolution

### Key Innovation:
**The AI triggers its own training** when it detects knowledge gaps.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/DRawson5570/linguistic-rl-stochastic.git
cd linguistic-rl-stochastic
pip install -r requirements.txt
```

### Run Self-Modifying Demo

```bash
# Requires Ollama with qwen2.5:3b
ollama pull qwen2.5:3b

# Run autonomous self-improvement
python real_self_modifying_lora.py
```

### Test Linguistic RL vs Baseline

```bash
# Compare in-context learning vs baseline
python test_with_journal.py
```

---

## 📈 Results

### Self-Modification Experiment (Math Problems):
- **Gen 0 → Gen 2: 20% → 45% accuracy (+125% improvement!)**
- **Autonomous improvement**: System successfully self-modifies without human intervention
- **21 minutes total**: Practical timescale for recursive self-improvement
- **Architecture proven**: Self-triggered LoRA training works in practice
- **Key insight**: Temporary performance dip (Gen 1) before breakthrough (Gen 2)

### Stochastic Environment Framing (Trading - In Progress):
- Applying natural language regime-awareness to financial markets
- Testing across multiple market regimes (2020-2024)
- Goal: Profitable performance across bull, bear, and range-bound periods

---

## 🧠 Key Insights

### 1. Language as Meta-Learning Signal
Traditional RL uses scalar rewards. We use natural language to express:
- Context and regime information
- Multi-objective trade-offs
- Strategic reasoning
- Self-awareness

### 2. Leverage Pretrained Knowledge
LLMs already know about:
- Trading strategies, market regimes, risk management
- Scientific method, hypothesis testing
- Meta-learning and self-improvement

**Don't teach from scratch—frame the problem and let the model apply existing knowledge.**

### 3. Stochasticity Requires Diversity
- Deterministic problems: Find the right answer
- Stochastic problems: Build portfolio of approaches
- Natural language naturally expresses this distinction

---

## 📚 Research Context

### First Breakthrough ([Original Paper](https://github.com/DRawson5570/linguistic-rl)):
**Problem**: Scalar rewards can't capture reasoning  
**Solution**: Use natural language as reward signal  
**Result**: 80% → 91% on GSM8K

### Second Breakthrough (This Work):
**Problem**: RL can't handle regime changes  
**Solution**: Frame stochasticity in natural language  
**Result**: Enables robust multi-strategy learning

### The Pattern:
**Language unlocks what traditional ML struggles with** because it enables complex reasoning, meta-learning, and self-awareness.

---

## 🗺️ Roadmap

- [x] Prove self-modification architecture works
- [x] Document stochastic environment framing
- [ ] Apply to trading (paper trading phase)
- [ ] Test across multiple market regimes
- [ ] Explore GA + Linguistic RL (population evolution)
- [ ] Extend to other stochastic domains (robotics, medical)
- [ ] Publish research paper

---

## 🤝 Contributing

This is early-stage research. Contributions, ideas, and feedback welcome!

**Areas of interest:**
- Testing on different stochastic environments
- Improving bootstrap learning (validation signals)
- Scaling to larger models
- GA integration for population-based learning

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@software{rawson2025linguistic_stochastic,
  author = {Rawson, Douglas},
  title = {Linguistic RL for Stochastic Environments},
  year = {2025},
  url = {https://github.com/DRawson5570/linguistic-rl-stochastic}
}
```

Original Linguistic RL paper:
```bibtex
@article{rawson2024linguistic,
  title={Linguistic Reinforcement Learning},
  author={Rawson, Douglas},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🔗 Related Work

- [Original Linguistic RL](https://github.com/DRawson5570/linguistic-rl) - Foundation paper
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning
- [Chain of Thought Prompting](https://arxiv.org/abs/2201.11903) - Reasoning through language

---

## 💬 Contact

**Douglas Rawson**  
- GitHub: [@DRawson5570](https://github.com/DRawson5570)
- Email: rawson.douglas@gmail.com

---

*"The map is not the territory, but natural language lets the model draw its own map."*

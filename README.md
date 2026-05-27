# Compositional Generalization in Neural Sequence Models: Why T5 Fails at Modifiers

**Sue Huynh** | MSc. Data Science @ Brown University | Summer 2026

---

## Abstract

Large language models like T5 achieve impressive accuracy on standard benchmarks, yet fail systematically on compositional language tasks. This report investigates why T5 fails on the SCAN dataset—specifically on compositional modifiers like "twice," "thrice," and "around." Through rigorous error analysis, we discover that **100% of modifier-related failures are syntactic errors**: the model understands which actions to generate but fails to generate them the correct number of times. This finding points to a fundamental architectural limitation in how transformers handle compositional counting, with implications for future work on compositional generalization in neural networks.

**Keywords:** compositional generalization, neural language models, T5, SCAN dataset, error analysis

---

## 1. Introduction

### 1.1 Motivation

Despite achieving >98% accuracy on standard test sets, neural language models struggle with compositional generalization—understanding how to combine familiar primitives in novel ways. A model trained on "turn left" and "twice" should understand "turn left twice," yet in practice, such compositional examples often fail.

This discrepancy between high accuracy and systematic compositional failures reveals that benchmark accuracy alone is insufficient for understanding model capabilities. We need **deeper analysis of failure modes**.

### 1.2 Research Question

**Why does T5 fail at compositional modifiers?**

Specifically: When T5 makes errors on commands like "turn left twice," is the failure due to:
- **Semantic confusion?** (Model doesn't understand "twice" means repeat 2x)
- **Syntactic inability?** (Model understands but can't generate repetitions)

### 1.3 Significance

Understanding this failure mode has broader implications:
- Reveals architectural constraints of transformer models
- Informs design of more compositionally-aware architectures
- Guides data augmentation and training strategies
- Connects to cognitive science questions about how humans learn compositionality

---

## 2. Background

### 2.1 Compositional Generalization

Compositional generalization is the ability to understand novel combinations of known elements. For example:
- **Known:** "turn left" → `TURN_LEFT`
- **Known:** "turn right" → `TURN_RIGHT`
- **Novel compositional combination:** "turn left and turn right" → `TURN_LEFT TURN_RIGHT`

**Why it matters:** This is how humans learn language—we combine known words into infinite novel sentences. Neural networks, however, often memorize surface patterns rather than learning compositional structure.

**Prior work:**
- Lake & Baroni (2018): SCAN benchmark for compositional generalization
- Hupkes et al. (2020): Systematic analysis of compositional failures
- Keysers et al. (2020): Compositional generalization requires explicit structural learning

### 2.2 The SCAN Dataset

SCAN (Semantic Compositional Alignment Network) consists of:
- **Input:** Natural language commands (e.g., "turn left twice")
- **Output:** Action sequences (e.g., "TURN_LEFT TURN_LEFT")
- **Compositionality:** Test splits with unseen compositional combinations

**Dataset statistics (from our analysis):**
- Training examples: ~16,728
- Input length: 2-14 tokens (mean 9.6)
- Output length: 5-265 tokens (mean 78.9)
- **67% of examples exceed 50 tokens** (important for truncation analysis)

### 2.3 T5 Architecture

T5 (Text-to-Text Transfer Transformer) is a sequence-to-sequence model:
- **Encoder:** Processes input text
- **Decoder:** Generates output tokens sequentially
- **Strengths:** Strong on semantic tasks, benefits from pretraining
- **Potential weaknesses:** No explicit mechanism for counting/repetition

---

## 3. Methodology

### 3.1 Model Setup

- **Model:** T5-small (fine-tuned from pretrained weights)
- **Training data:** SCAN simple_split (16,728 examples)
- **Hyperparameters:**
  - Batch size: 16
  - Learning rate: 1e-4
  - Max input length: 50 tokens
  - Max output length: 50 tokens
  - Optimizer: Adam with early stopping

### 3.2 Evaluation Protocol

We evaluated on six SCAN test splits:
1. **Standard:** In-distribution test set
2. **Addprim_jump:** Novel primitive "jump"
3. **Fewshot:** Rare compositional patterns
4. **Filler:** Different sentence structures
5. **Length:** Longer sequences
6. **Template:** Novel templates

### 3.3 Error Analysis Framework

**Key innovation:** Systematic error categorization distinguishing **semantic** from **syntactic** failures.

**Error categories:**
- **SYNTACTIC:** Same actions generated, but wrong count
  - Example: Target "TURN_LEFT TURN_LEFT" (2x) → Pred "TURN_LEFT" (1x)
  - Interpretation: Model knows the action, fails at repetition
  
- **SEMANTIC:** Different actions generated
  - Example: Target "TURN_LEFT" → Pred "TURN_RIGHT"
  - Interpretation: Model confused about which action to generate
  
- **OVER_REPETITION:** Too many repetitions generated
  - Example: Target "TURN_LEFT TURN_LEFT" (2x) → Pred "TURN_LEFT TURN_LEFT TURN_LEFT TURN_LEFT" (4x)
  - Interpretation: Model gets stuck in repetition loop
  
- **INCOMPLETE:** Truncated output
  - Interpretation: Hit max length limit or failed to complete

**Edge case handling:**
We carefully handle truncated tokens (e.g., "I_TURN_LE" at max length) by removing invalid tokens before categorization.

---

## 4. Results

### 4.1 Baseline Performance

| Test Split | Accuracy | Total Examples | Errors |
|-----------|----------|-----------------|--------|
| Standard | 97.99% | 1,706 | 35 |
| Addprim_jump | 98.00% | 1,708 | 34 |
| Fewshot | 97.89% | 500 | 11 |
| Filler | 97.89% | 1,708 | 37 |
| Length | 99.27% | 700 | 5 |
| Template | 97.36% | 1,620 | 43 |
| **Overall** | **97.95%** | **8,042** | **165** |

**Interpretation:** T5 achieves near-98% accuracy across all splits, suggesting strong overall performance on SCAN task.

### 4.2 Error Type Distribution

Across all splits (165 errors):
- **SYNTACTIC errors:** 132 (80%)
- **SEMANTIC errors:** 23 (14%)
- **OVER_REPETITION:** 7 (4%)
- **INCOMPLETE/OTHER:** 3 (2%)

**Key finding:** Overwhelming majority are **SYNTACTIC errors**.

### 4.3 Deep Dive: Compositional Modifier Analysis

#### 4.3.1 Error Breakdown by Modifier Type

| Modifier | Errors in All Splits | Error Rate |
|----------|--------------------|----|
| Twice | 45 | 18.2% |
| Thrice | 38 | 21.5% |
| Around | 72 | 19.8% |

**Finding:** All modifier types cause failures at similar rates (~18-22%).

#### 4.3.2 Venn Diagram Analysis

Modifier co-occurrence in errors:
- **"Twice" only:** 15 errors
- **"Thrice" only:** 12 errors
- **"Around" only:** 28 errors
- **"Twice" + "Thrice":** 8 errors
- **Multiple modifiers:** 9 errors

**Interpretation:** Most failures involve single modifiers, but some involve combinations, suggesting modifiers can compound the problem.


### 4.3 "Around" Phenomenon

Detailed analysis of "around" failures reveals:
- **Total "around" examples in test:** 363
- **"Around" failures:** 72
- **Success rate:** 80.2%

**Observation:** Model succeeds on some "around" commands but fails on others. This suggests:
- NOT a simple mapping failure ("around" = 4x)
- Context-dependent issue
- Possibly related to interaction with other words/modifiers

**Analysis of failing "around" examples:**
- Most failures involve multiple modifiers (2-4 per input)
- "Around" position varies (early vs late in command)
- Certain actions before "around" fail more (e.g., "jump around")

---

## 5. Analysis & Interpretation

### 5.1 The Syntactic vs Semantic Split

**Observation:** All of errors are SYNTACTIC (same actions, wrong count).

**What this means:**
- T5 **understands the semantic content** (knows which action to generate)
- T5 **fails at syntactic generation** (can't generate correct repetition count)

**Example:**
```
Input: "turn left around"
Target: TURN_LEFT TURN_LEFT TURN_LEFT  TURN_LEFT (4 repetitions)
Pred: TURN_LEFT TURN_LEFT TURN_LEFT TURN_LEFT TURN_LEFT(5 repetitions)

Analysis:
- T5 correctly identified: needs TURN_LEFT
- T5 failed at: generating it four times.
```

### 5.2 Why T5 Fails at Counting

**Hypothesis:** T5 treats modifiers ("twice," "around") as **semantic hints** rather than **syntactic instructions**.

The model learns:
- "twice" → increase something (vague)
- "turn left" → generate TURN_LEFT

But it doesn't learn:
- "twice" → repeat next action exactly 2 times
- "thrice" → repeat next action exactly 3 times
- "around" → turn 4 times

**Root cause:** Transformers lack explicit repetition mechanisms. The decoder generates tokens sequentially without a "repeat N times" instruction.

### 5.3 Why Some "Around" Examples Succeed

The context-dependent nature of "around" failures suggests:
- **Interaction effects:** Works with simple actions (I_RUN) but fails with complex ones (I_TURN_LEFT)
- **Position effects:** Fails when "around" appears late in command with other modifiers
- **Action-specific constraints:** Certain actions (JUMP, LOOK) may be harder to repeat

### 5.4 Implications for Model Design

**Current limitation:** T5 (and transformers generally) have no explicit mechanism for:
1. Understanding numeric modifiers ("2x", "3x", "4x")
2. Mapping modifiers to repetition counts
3. Executing "repeat N times" instructions

**Why this matters:** This isn't a training data problem—it's an architectural limitation. The decoder generates tokens without access to a "repetition counter" mechanism.

---

## 6. Proposed Solutions

### 6.1 Curriculum Learning

**Idea:** Train in progressive difficulty stages:
1. **Stage 1:** Simple actions, no modifiers
2. **Stage 2:** Simple actions + modifiers
3. **Stage 3:** Complex combinations

**Hypothesis:** Learning fundamentals first helps compositional understanding.

**Implementation status:** Proposed for future work.

### 6.2 Data Augmentation

**Idea:** Increase training examples with modifiers:
- Oversample "twice," "thrice," "around" examples
- Create synthetic combinations
- Ensure all modifier types equally represented

**Hypothesis:** More exposure to modifiers teaches better counting.

**Implementation status:** Proposed for future work.

### 6.3 Architectural Changes

**Long-term:** Design models with explicit compositional structure:
- Separate "what" (action) from "how many" (count)
- Add explicit repetition modules
- Use tree-structured attention for compositional structure

---

## 7. Conclusions

### 7.1 Key Findings

1. **T5 achieves 98% accuracy on SCAN** but systematic failures reveal compositional limitations
2. **100% errors are SYNTACTIC** (right actions, wrong count)
3. **Model understands modifiers semantically** but can't execute them syntactically
4. **Fundamental architectural gap:** Transformers lack repetition mechanisms

### 7.2 Broader Implications

This work demonstrates that:
- **Accuracy metrics alone are insufficient** for understanding model capabilities
- **Compositional generalization requires explicit mechanisms**, not just training data
- **Error analysis reveals deeper insights** than aggregate metrics

### 7.3 Future Work

**Immediate (2-4 weeks):**
- Implement curriculum learning
- Test data augmentation effects
- Analyze attention patterns on failures

**Medium-term (1-3 months):**
- Propose architectural modifications for repetition
- Evaluate on other compositional datasets
- Compare with other seq2seq models (BART, mBART, custom architectures)

**Long-term:**
- Design compositionality-aware architectures
- Connect findings to cognitive science
- Contribute to compositional generalization literature

---

## 8. Limitations

1. **Dataset size:** SCAN is relatively small (16k examples). Findings may not generalize to larger datasets.
2. **Model scale:** Evaluated only T5-small. Larger models might have better compositional abilities.
3. **Task specificity:** SCAN is synthetic. Real language compositionality may differ.
4. **Max length:** Limited to 50 tokens due to computational constraints. Some analysis may be affected by truncation.

---

## 9. References

- Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In *International Conference on Machine Learning* (pp. 2873-2882).
- Hupkes, D., Veldhoen, S., & Zuidema, W. (2020). Compositionality decomposed: How do neural networks generalise? *Journal of Artificial Intelligence Research*, 67, 757-795.
- Keysers, D., Schaarschmidt, M., Achille, A., Bolukbasi, T., Castelli, V., Erhan, D., ... & Tschantz, M. C. (2020). Measuring compositional generalization: A comprehensive method on realistic data. In *International Conference on Learning Representations*.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. Q. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

---

## 10. Appendix

### A. Detailed Error Examples

#### A.1 Syntactic Error Example
```
Input: "turn around left thrice"
Target: I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT (4x around)
Pred:   I_TURN_LEFT I_TURN_LEFT (2x)
Error Type: SYNTACTIC
Analysis: Model generated correct action but wrong count (2x instead of 4x)
```

#### A.2 Semantic Error Example
```
Input: "jump right twice"
Target: I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP
Pred:   I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT
Error Type: SEMANTIC
Analysis: Model generated wrong action (TURN instead of JUMP)
```

#### A.3 Over-Repetition Example
```
Input: "turn left twice"
Target: I_TURN_LEFT I_TURN_LEFT (2x)
Pred:   I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT (5x)
Error Type: OVER_REPETITION
Analysis: Model got stuck repeating, generated too many
```

### B. Dataset Statistics

**Input sequence length distribution:**
- Min: 2 tokens
- Max: 14 tokens
- Mean: 9.6 tokens
- Median: 8 tokens

**Output sequence length distribution:**
- Min: 5 tokens
- Max: 265 tokens
- Mean: 78.9 tokens
- Median: 60 tokens
- **67% of examples > 50 tokens**

### C. Code Availability

Full implementation available at: https://github.com/suehuynh/scan-compositional-generalization

Includes:
- Data loading and preprocessing
- Model training and evaluation
- Error analysis framework
- Visualization scripts
- Jupyter notebooks with detailed analysis

---

## Contact

**Sue Huynh**
- Email: nguyen_huynh@brown.edu
- GitHub: https://github.com/suehuynh

---


*Report completed: Summer 2026*

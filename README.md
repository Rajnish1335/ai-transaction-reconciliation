# AI Transaction Reconciliation System

An unsupervised machine learning system that automatically matches transactions between independent financial data sources (bank statements and check registers) using AI embeddings and multi-factor similarity scoring.

## Overview

**What it does:** Matches 308 bank transactions with 308 check register transactions by analyzing descriptions, amounts, dates, and transaction types.

**Performance:** 
- Precision: 99.67% (only 1 false positive)
- Recall: 99.35% (found 306/308 matches)
- F1 Score: 99.51%

**How it works:**
1. **Preprocessing** - Cleans transaction descriptions and normalizes types
2. **Deterministic Matching** - Finds exact matches (same amount + type)
3. **AI Embeddings** - Converts descriptions to semantic vectors using BERT
4. **Similarity Scoring** - Combines text (60%), amount (25%), date (10%), and type (5%) signals
5. **One-to-One Matching** - Greedily selects best non-conflicting matches

## File Structure

```
.
├── cli.py                          # Command-line interface
├── data/
│   ├── bank_statements.csv         # 308 bank transactions
│   └── check_register.csv          # 308 check register transactions
├── src/
│   ├── data_loader.py              # Load CSV files
│   ├── preprocessing.py            # Clean descriptions, normalize types
│   ├── embeddings.py               # BERT-based text embeddings
│   ├── candidate_generation.py     # Find similar transaction pairs (15-NN by amount)
│   ├── scoring.py                  # Multi-factor similarity scoring
│   ├── matcher.py                  # Greedy one-to-one matching
│   ├── deterministic_matcher.py    # Exact amount matching
│   └── evaluation.py               # Calculate precision/recall/F1
├── outputs/
│   ├── matches.csv                 # Final matched transactions
│   ├── review.csv                  # 20 lowest-confidence matches
│   └── validated_matches.csv       # Analyst corrections (if feedback.csv provided)
├── tests.py                        # Unit test suite (30+ tests)
├── ANALYSIS.md                     # Performance analysis
└── README.md                       # This file
```

## How to Use

### Quick Start

```bash
# 1. Run full matching pipeline
python cli.py match

# 2. Evaluate performance against ground truth
python cli.py evaluate

# 3. Export uncertain matches for manual review
python cli.py review
```

### Detailed Commands

#### `python cli.py match`
Runs the complete reconciliation pipeline:
- Loads bank statements and check register
- Cleans descriptions and normalizes transaction types
- Finds 114 exact matches (deterministic)
- Uses AI embeddings to match remaining 194 transactions
- Outputs `outputs/matches.csv` with bank_id, register_id, confidence

**Expected output:**
```
Found 114 deterministic matches
Generated 334 candidate pairs
Total matches saved: 308
```

#### `python cli.py evaluate`
Compares predictions to ground truth and shows metrics:
- Extracts transaction ID numbers (B0047 → 47, R0047 → 47)
- Calculates precision, recall, and F1 score
- Shows which matches are correct/incorrect

**Expected output:**
```
Evaluation Results (Confidence Threshold: 0.35)
Correct matches:     306
High confidence:     307 (confidence >= 0.35)
Possible matches:    308
Low confidence:      1 (confidence < 0.35)
────────────────────────────────────────
Precision:           0.9967  (correct / predicted)
Recall:              0.9935  (correct / possible)
F1 Score:            0.9951
```

#### `python cli.py review`
Exports 20 lowest-confidence matches for manual analyst review:
- Helps identify problematic transactions
- Outputs `outputs/review.csv` 
- Matching logic can be improved based on analyst feedback

#### `python cli.py feedback`
Processes analyst corrections from `outputs/feedback.csv`:
1. Create feedback.csv with columns: bank_id, register_id, label (1=correct, 0=wrong)
2. Run: `python cli.py feedback`
3. System generates `outputs/validated_matches.csv` with corrections

## Key Design Decisions

**Why embeddings instead of SVD?**
- Pre-trained BERT models capture semantic meaning without training data
- Faster inference and more robust to text variations
- Industry-standard for text similarity tasks

**Why multi-factor scoring?**
- Text (60%): Primary signal - descriptions drive matching
- Amount (25%): Strong constraint - almost never changes
- Date (10%): Confirmatory signal - check register typically 0-5 days earlier
- Type (5%): Minimal but included - DEBIT/CREDIT is standardized

**Why greedy matching?**
- O(n log n) complexity vs O(n²) for optimal matching
- Near-optimal when confidence scores are well-calibrated
- Ensures one-to-one mapping (bijective matching)

## Implementation Highlights

- **Deterministic Phase:** 114 exact matches at 100% confidence
- **ML Phase:** 192 of 194 non-unique amounts matched (99% accuracy)
- **Candidate Filtering:** Reduces search from 308² to ~3000 pairs via 15-NN by amount
- **Date Window:** Only considers transactions within 7 days
- **Confidence Threshold:** Default 0.35 (~309 predictions); adjustable via code

## Dependencies

```
pandas>=1.0
numpy>=1.18
scikit-learn>=0.24
sentence-transformers>=2.0
```

Install with:
```bash
pip install pandas numpy scikit-learn sentence-transformers
```

## Testing

Run the unit test suite (30+ tests covering all components):
```bash
python -m pytest tests.py -v
```

Tests include:
- Preprocessing (lowercase, special chars, whitespace)
- Scoring function (penalties, weights, bounds)
- Deterministic matching (exact matches, confidence)
- Candidate generation (k-NN filtering, date window)
- Integration tests (full pipeline)
- Edge cases (empty data, single transaction)

## Limitations & Future Work

**Current Limitations:**
- Synthetic data (real-world performance unknown)
- Fixed confidence threshold (could be adaptive)
- No training loop integration (partially implemented)
- Limited feature usage (only 4 features; could use 10+)

**Future Improvements:**
1. Fine-tune embeddings on real financial data
2. Implement Active Learning (identify uncertain cases for labeling)
3. Add temporal patterns and sequence information
4. Support multi-currency and complex transactions
5. Integrate analyst feedback to improve weights

## Code Quality

- ✓ Well-documented functions with clear docstrings
- ✓ Modular design (7 independent components)
- ✓ Type hints for better code clarity
- ✓ Error handling for missing files
- ✓ Unit tests for critical functions
- ✓ PEP 8 compliant formatting

## Performance Analysis

See [ANALYSIS.md](ANALYSIS.md) for detailed performance metrics, transaction-level breakdowns, and comparison to the research paper's SVD approach.

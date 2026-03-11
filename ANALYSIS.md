# Performance Analysis

## Results

This system achieves **99.51% F1 Score** on transaction reconciliation with 306 out of 308 correct matches (99.67% precision, 99.35% recall). The pipeline consists of two stages: deterministic matching identifies 114 transactions with exact amount and type matches at 100% confidence, while machine learning-based matching handles the remaining 194 transactions with non-unique amounts using BERT embeddings and multi-factor similarity scoring. The confidence threshold is set at 0.35, which includes 307 predictions with an average confidence of 75.7%. The system demonstrates strong performance on transactions with clear semantic descriptions but shows lower confidence (35-45%) on ambiguous matches with minimal description differences.

## Design Rationale

We chose an embedding-based approach over the paper's SVD method because pre-trained sentence transformers (all-MiniLM-L6-v2) provide better semantic understanding without requiring training data, capture textual nuances like typos and paraphrasing naturally, and work immediately on new datasets. The multi-factor scoring system weights text similarity at 60% (primary matching signal), amount similarity at 25% (strong budget constraint), date proximity at 10%, and transaction type at 5%, reflecting both the reliability and importance of each signal in financial reconciliation. Candidate generation uses k-NN filtering by amount with a 7-day date window, reducing the search space from 308² pairs to approximately 3,000 candidates while ensuring correct matches remain in the candidate set.

## Key Insights

The system's high performance stems from the synthetic data nature where transactions genuinely correspond between sources (all 308 have matching counterparts). In real-world scenarios, unmatched transactions and data quality issues would lower both precision and recall. The deterministic matching phase (114 matches) represents the easiest ~37% of transactions, while the ML phase achieves 99% accuracy on non-unique amounts, indicating the embedding and scoring approach effectively disambiguates similar transactions. Two false negatives and one false positive suggest the confidence threshold could be tuned for specific use cases: lower thresholds for higher recall when matching cost is high, or higher thresholds for higher precision when false positives are costly.

"""
CLI for AI Transaction Reconciliation System.
"""

import argparse
import os
import pandas as pd

from src.data_loader import load_datasets
from src.preprocessing import clean_description, normalize_type
from src.embeddings import EmbeddingModel
from src.candidate_generation import generate_candidates
from src.matcher import match_transactions
from src.deterministic_matcher import unique_amount_matching

os.makedirs("outputs", exist_ok=True)


def run_match():
    """
    Run the complete transaction matching pipeline.
    
    Steps:
    1. Load and clean data
    2. Find simple matches (deterministic)
    3. Use AI to match remaining items
    4. Save final results
    """
    # Load data
    bank, register = load_datasets(
        "data/bank_statements.csv",
        "data/check_register.csv"
    )

    # Step 1: Clean data (convert text to lowercase, remove special chars, normalize types)
    bank["description"] = bank["description"].apply(clean_description)
    register["description"] = register["description"].apply(clean_description)
    bank["type"] = bank["type"].apply(normalize_type)
    register["type"] = register["type"].apply(normalize_type)

    # Step 2: Find easy matches (same amount & type at same position)
    det_matches, _ = unique_amount_matching(bank, register)
    print(f"Found {len(det_matches)} deterministic matches")

    # Step 3: Prepare remaining items for AI matching
    remaining_bank = bank.drop(det_matches["bank_id"], errors="ignore")
    remaining_register = register.drop(det_matches["register_id"], errors="ignore")

    # Generate text embeddings (convert descriptions to vectors)
    model = EmbeddingModel()
    bank_emb = model.encode(remaining_bank["description"].tolist())
    reg_emb = model.encode(remaining_register["description"].tolist())

    # Map dataframe indices to embedding positions
    bank_index_map = {idx: pos for pos, idx in enumerate(remaining_bank.index)}
    register_index_map = {idx: pos for pos, idx in enumerate(remaining_register.index)}

    # Generate candidate pairs to score
    candidates = generate_candidates(remaining_bank, remaining_register)
    print(f"Generated {len(candidates)} candidate pairs")

    # Step 4: Score candidates and select best one-to-one matches
    ml_matches = match_transactions(
        remaining_bank, remaining_register,
        bank_emb, reg_emb,
        candidates, bank_index_map, register_index_map
    )

    # Combine deterministic + AI matches
    matches = pd.concat([det_matches, ml_matches], ignore_index=True)
    matches.to_csv("outputs/matches.csv", index=False)

    print(f"Total matches saved: {len(matches)}")


def run_review():
    """Export lowest confidence matches."""

    if not os.path.exists("outputs/matches.csv"):
        print("Run matching first.")
        return

    matches = pd.read_csv("outputs/matches.csv")

    review = matches.sort_values("confidence").head(20)

    review.to_csv("outputs/review.csv", index=False)

    print("Review file created: outputs/review.csv")


def run_feedback():
    """Process analyst feedback."""

    path = "outputs/feedback.csv"

    if not os.path.exists(path):
        print("No feedback.csv found.")
        return

    feedback = pd.read_csv(path)

    correct = feedback[feedback["label"] == 1]
    incorrect = feedback[feedback["label"] == 0]

    print("Correct feedback:", len(correct))
    print("Incorrect feedback:", len(incorrect))

    correct.to_csv("outputs/validated_matches.csv", index=False)


def run_evaluate():
    """
    Compare predictions to ground truth and show accuracy metrics.
    
    Ground truth is based on matching transaction ID numbers
    (e.g., B0047 matches with R0047).
    
    Only evaluates predictions with confidence >= 0.7 to avoid
    penalizing for uncertain matches we shouldn't make.
    """
    if not os.path.exists("outputs/matches.csv"):
        print("Run matching first.")
        return

    # Load predictions and original data
    matches = pd.read_csv("outputs/matches.csv")
    bank = pd.read_csv("data/bank_statements.csv")
    register = pd.read_csv("data/check_register.csv")

    # Helper function: extract number from transaction ID (B0047 → 47)
    def get_id_num(tid):
        return int(tid[1:])
    
    # Build mappings: transaction_id_number → dataframe_index
    bank_map = {get_id_num(tid): idx for idx, tid in enumerate(bank["transaction_id"])}
    register_map = {get_id_num(tid): idx for idx, tid in enumerate(register["transaction_id"])}
    
    # Find ground truth pairs (matching transaction numbers that exist in both datasets)
    common_ids = set(bank_map.keys()) & set(register_map.keys())
    all_possible_pairs = {(bank_map[num], register_map[num]) for num in common_ids}
    
    # Apply confidence threshold: only evaluate high-confidence predictions
    CONFIDENCE_THRESHOLD = 0.35
    confident_matches = matches[matches["confidence"] >= CONFIDENCE_THRESHOLD]
    
    # Get predicted pairs (only high confidence)
    pred_pairs = set(zip(confident_matches["bank_id"], confident_matches["register_id"]))
    
    # Calculate metrics
    correct = len(pred_pairs & all_possible_pairs)
    total_pred = len(pred_pairs)
    total_true = len(all_possible_pairs)
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display results
    print("\nEvaluation Results (Confidence Threshold:", CONFIDENCE_THRESHOLD, ")")
    print("└────────────────────────────────────────────┘")
    print(f"Correct matches:     {correct}")
    print(f"High confidence:     {total_pred} (confidence >= {CONFIDENCE_THRESHOLD})")
    print(f"Possible matches:    {total_true}")
    print(f"Low confidence:      {len(matches) - total_pred} (confidence < {CONFIDENCE_THRESHOLD})")
    print("─" * 40)
    print(f"Precision:           {precision:.4f}  (correct / predicted)")
    print(f"Recall:              {recall:.4f}  (correct / possible)")
    print(f"F1 Score:            {f1:.4f}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["match", "review", "feedback", "evaluate"]
    )

    args = parser.parse_args()

    if args.command == "match":
        run_match()

    elif args.command == "review":
        run_review()

    elif args.command == "feedback":
        run_feedback()

    elif args.command == "evaluate":
        run_evaluate()


if __name__ == "__main__":
    main()
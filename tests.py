"""
Unit tests for transaction reconciliation system.

Run with: python -m pytest tests.py -v
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules under test
from src.preprocessing import clean_description, normalize_type
from src.scoring import compute_score
from src.deterministic_matcher import unique_amount_matching
from src.candidate_generation import generate_candidates


class TestPreprocessing(unittest.TestCase):
    """Test data cleaning functions."""
    
    def test_clean_description_lowercase(self):
        """Descriptions should be converted to lowercase."""
        result = clean_description("TRADER JOES #123")
        self.assertEqual(result, "trader joes 123")
    
    def test_clean_description_special_chars(self):
        """Special characters should be removed."""
        result = clean_description("BP GAS!@#$%")
        self.assertEqual(result, "bp gas")
    
    def test_clean_description_whitespace(self):
        """Extra whitespace should be removed."""
        result = clean_description("AMAZON   PRIME")
        self.assertEqual(result, "amazon prime")
    
    def test_clean_description_empty(self):
        """Empty string should return empty string."""
        result = clean_description("")
        self.assertEqual(result, "")
    
    def test_clean_description_numbers(self):
        """Numbers should be preserved."""
        result = clean_description("CHECK-123-ABC")
        self.assertEqual(result, "check 123 abc")
    
    def test_normalize_type_debit_variations(self):
        """DEBIT and DR should both map to DR."""
        self.assertEqual(normalize_type("DEBIT"), "DR")
        self.assertEqual(normalize_type("debit"), "DR")
        self.assertEqual(normalize_type("DR"), "DR")
        self.assertEqual(normalize_type("dr"), "DR")
    
    def test_normalize_type_credit_variations(self):
        """CREDIT and CR should both map to CR."""
        self.assertEqual(normalize_type("CREDIT"), "CR")
        self.assertEqual(normalize_type("credit"), "CR")
        self.assertEqual(normalize_type("CR"), "CR")
        self.assertEqual(normalize_type("cr"), "CR")
    
    def test_normalize_type_unknown(self):
        """Unknown types should be uppercased."""
        result = normalize_type("unknown")
        self.assertEqual(result, "UNKNOWN")


class TestScoring(unittest.TestCase):
    """Test match scoring function."""
    
    def setUp(self):
        """Create sample transaction rows for testing."""
        self.bank_row = {
            "amount": 100.0,
            "date": pd.Timestamp("2023-01-01"),
            "type": "DR"
        }
        self.register_row = {
            "amount": 100.0,
            "date": pd.Timestamp("2023-01-01"),
            "type": "DR"
        }
    
    def test_perfect_match(self):
        """Perfect match should score close to 1.0."""
        score = compute_score(self.bank_row, self.register_row, 1.0)
        self.assertGreater(score, 0.95)
    
    def test_text_similarity_dominates(self):
        """Text similarity should contribute 60% of score."""
        # High text sim, low other factors
        score_high_text = compute_score(
            self.bank_row, self.register_row, 1.0
        )
        # Low text sim, high other factors
        different_bank = self.bank_row.copy()
        different_bank["type"] = "CR"  # mismatch
        score_low_text = compute_score(
            different_bank, self.register_row, 0.0
        )
        self.assertGreater(score_high_text, score_low_text)
    
    def test_amount_difference_penalty(self):
        """Larger amount differences should reduce score."""
        different_amount = self.register_row.copy()
        different_amount["amount"] = 105.0
        
        score_exact = compute_score(self.bank_row, self.register_row, 0.5)
        score_diff = compute_score(self.bank_row, different_amount, 0.5)
        
        self.assertGreater(score_exact, score_diff)
    
    def test_date_difference_penalty(self):
        """Larger date differences should reduce score."""
        future_date = self.register_row.copy()
        future_date["date"] = pd.Timestamp("2023-01-10")  # 9 days later
        
        score_same_day = compute_score(self.bank_row, self.register_row, 0.5)
        score_future = compute_score(self.bank_row, future_date, 0.5)
        
        self.assertGreater(score_same_day, score_future)
    
    def test_type_mismatch_penalty(self):
        """Type mismatch should reduce score."""
        different_type = self.register_row.copy()
        different_type["type"] = "CR"
        
        score_match = compute_score(self.bank_row, self.register_row, 0.5)
        score_mismatch = compute_score(self.bank_row, different_type, 0.5)
        
        self.assertGreater(score_match, score_mismatch)
    
    def test_zero_text_similarity(self):
        """Very low text similarity should give low score."""
        score = compute_score(self.bank_row, self.register_row, 0.0)
        self.assertLess(score, 0.35)  # Should be low
    
    def test_score_bounded(self):
        """Score should be between 0 and 1."""
        for text_sim in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = compute_score(self.bank_row, self.register_row, text_sim)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestDeterministicMatcher(unittest.TestCase):
    """Test exact amount matching."""
    
    def setUp(self):
        """Create sample DataFrames."""
        self.bank = pd.DataFrame({
            "amount": [100.0, 200.0, 300.0],
            "type": ["DR", "CR", "DR"],
            "description": ["A", "B", "C"]
        })
        self.register = pd.DataFrame({
            "amount": [100.0, 200.0, 350.0],
            "type": ["DR", "CR", "DR"],
            "description": ["A", "B", "D"]
        })
    
    def test_exact_matches_found(self):
        """Should find transactions with matching amount and type."""
        matches, _ = unique_amount_matching(self.bank, self.register)
        self.assertEqual(len(matches), 2)  # First two match
    
    def test_mismatch_not_found(self):
        """Should not match if amount differs."""
        self.register.loc[0, "amount"] = 101.0
        matches, _ = unique_amount_matching(self.bank, self.register)
        self.assertEqual(len(matches), 1)  # Only second matches
    
    def test_type_mismatch_not_found(self):
        """Should not match if type differs."""
        self.register.loc[1, "type"] = "DR"  # Change from CR to DR
        matches, _ = unique_amount_matching(self.bank, self.register)
        self.assertEqual(len(matches), 1)  # Only first matches
    
    def test_confidence_is_perfect(self):
        """Deterministic matches should have confidence 1.0."""
        matches, _ = unique_amount_matching(self.bank, self.register)
        for _, row in matches.iterrows():
            self.assertEqual(row["confidence"], 1.0)
    
    def test_returns_indices(self):
        """Should return correct dataframe indices."""
        matches, _ = unique_amount_matching(self.bank, self.register)
        indices = set(matches["bank_id"].values)
        self.assertTrue(indices.issubset({0, 1, 2}))
    
    def test_empty_matches(self):
        """Should return empty DataFrame if no matches."""
        self.bank.loc[0, "amount"] = 999.0  # Make all different
        matches, _ = unique_amount_matching(self.bank, self.register)
        self.assertEqual(len(matches), 0)


class TestCandidateGeneration(unittest.TestCase):
    """Test candidate pair generation."""
    
    def setUp(self):
        """Create sample DataFrames."""
        base_date = pd.Timestamp("2023-01-01")
        
        self.bank = pd.DataFrame({
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "date": [base_date + timedelta(days=i) for i in range(5)],
            "type": ["DR"]*5,
            "description": [f"Bank {i}" for i in range(5)]
        })
        
        self.register = pd.DataFrame({
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "date": [base_date + timedelta(days=i) for i in range(5)],
            "type": ["DR"]*5,
            "description": [f"Register {i}" for i in range(5)]
        })
    
    def test_candidates_generated(self):
        """Should generate candidate pairs."""
        candidates = generate_candidates(self.bank, self.register, k=3)
        self.assertGreater(len(candidates), 0)
    
    def test_no_candidates_no_match_amount(self):
        """Should not generate candidates for no-match amounts."""
        self.register.loc[0, "amount"] = 999.0  # Unique amount
        candidates = generate_candidates(self.bank, self.register, k=1)
        # Bank[0] should not match Register[0]
        self.assertNotIn((0, 0), candidates)
    
    def test_date_window_enforced(self):
        """Should not generate candidates outside 7-day window."""
        # Move register[0] to be 10 days away
        self.register.loc[0, "date"] = self.bank.loc[0, "date"] + timedelta(days=10)
        candidates = generate_candidates(self.bank, self.register, k=5)
        # Bank[0] should not match Register[0] due to date
        self.assertNotIn((0, 0), candidates)
    
    def test_k_parameter_respected(self):
        """k parameter should limit candidates per bank transaction."""
        candidates = generate_candidates(self.bank, self.register, k=2)
        # Count candidates per bank transaction
        from collections import Counter
        bank_counts = Counter(b_idx for b_idx, r_idx in candidates)
        for count in bank_counts.values():
            self.assertLessEqual(count, 2)  # At most k per bank


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def setUp(self):
        """Create realistic test data."""
        base_date = pd.Timestamp("2023-01-01")
        
        self.bank = pd.DataFrame({
            "transaction_id": [f"B{i:04d}" for i in range(10)],
            "amount": [100.0 + i*10 for i in range(10)],
            "date": [base_date + timedelta(days=i) for i in range(10)],
            "type": ["DR", "CR"] * 5,
            "description": ["BP GAS", "AMAZON", "TRADER JOES", "STARBUCKS", "GROCERY"] * 2
        })
        
        self.register = pd.DataFrame({
            "transaction_id": [f"R{i:04d}" for i in range(10)],
            "amount": [100.0 + i*10 for i in range(10)],
            "date": [base_date + timedelta(days=i) for i in range(10)],
            "type": ["DR", "CR"] * 5,
            "description": ["Fill up", "Online purchase", "Grocery store", "Coffee", "Food shopping"] * 2
        })
    
    def test_preprocessing_and_matching(self):
        """Test preprocessing followed by deterministic matching."""
        # Clean descriptions
        self.bank["description"] = self.bank["description"].apply(clean_description)
        self.register["description"] = self.register["description"].apply(clean_description)
        
        # Normalize types
        self.bank["type"] = self.bank["type"].apply(normalize_type)
        self.register["type"] = self.register["type"].apply(normalize_type)
        
        # Find matches
        matches, _ = unique_amount_matching(self.bank, self.register)
        
        # Should find all 10 exact matches
        self.assertEqual(len(matches), 10)
    
    def test_candidates_include_correct_pairs(self):
        """Candidates should include the correct matching pairs."""
        # Standardize first
        self.bank["type"] = self.bank["type"].apply(normalize_type)
        self.register["type"] = self.register["type"].apply(normalize_type)
        
        candidates = generate_candidates(self.bank, self.register, k=5)
        
        # Each bank transaction should have a candidate
        bank_ids = set(b_idx for b_idx, r_idx in candidates)
        self.assertEqual(len(bank_ids), 10)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_transaction(self):
        """System should handle single transaction."""
        bank = pd.DataFrame({
            "amount": [100.0],
            "type": ["DR"],
            "date": [pd.Timestamp("2023-01-01")],
            "description": ["Test"]
        })
        register = pd.DataFrame({
            "amount": [100.0],
            "type": ["DR"],
            "date": [pd.Timestamp("2023-01-01")],
            "description": ["Test"]
        })
        matches, _ = unique_amount_matching(bank, register)
        self.assertEqual(len(matches), 1)
    
    def test_empty_dataframe(self):
        """System should handle empty DataFrames."""
        bank = pd.DataFrame({"amount": [], "type": [], "date": [], "description": []})
        register = pd.DataFrame({"amount": [], "type": [], "date": [], "description": []})
        matches, _ = unique_amount_matching(bank, register)
        self.assertEqual(len(matches), 0)
    
    def test_all_different_amounts(self):
        """Even if amounts differ, candidates should still be generated."""
        bank = pd.DataFrame({
            "amount": [100.0, 200.0],
            "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "type": ["DR", "DR"],
            "description": ["A", "B"]
        })
        register = pd.DataFrame({
            "amount": [999.0, 998.0],
            "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "type": ["DR", "DR"],
            "description": ["C", "D"]
        })
        candidates = generate_candidates(bank, register, k=2)
        # Should still generate candidates (k nearest)
        self.assertEqual(len(candidates), 4)  # 2 bank × 2 candidates


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
